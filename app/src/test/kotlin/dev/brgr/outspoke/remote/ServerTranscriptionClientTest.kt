package dev.brgr.outspoke.remote

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import org.assertj.core.api.Assertions.assertThat
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/** Robolectric supplies a working `org.json`; the HTTP server is the JDK's. */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], manifest = Config.NONE)
class ServerTranscriptionClientTest {
    private class Recorded(val method: String, val path: String, val authorization: String?, val contentType: String?, val body: ByteArray)

    private lateinit var server: HttpServer
    private val requests = LinkedBlockingQueue<Recorded>()
    private val client = ServerTranscriptionClient(transcribeTimeoutMs = 1_000, modelsTimeoutMs = 1_000)
    private val endpoint get() = ServerEndpoint.parse("http://127.0.0.1:${server.address.port}")!!

    @Before
    fun start() {
        server = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)
        server.executor = Executors.newCachedThreadPool { Thread(it).apply { isDaemon = true } }
        server.start()
    }

    @After
    fun stop() = server.stop(0)

    private fun respond(path: String, status: Int, body: String, delayMs: Long = 0) {
        server.createContext(path) { exchange: HttpExchange ->
            requests.add(Recorded(
                exchange.requestMethod, exchange.requestURI.path,
                exchange.requestHeaders.getFirst("Authorization"), exchange.requestHeaders.getFirst("Content-Type"),
                exchange.requestBody.readBytes(),
            ))
            Thread.sleep(delayMs)
            val bytes = body.toByteArray()
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
    }

    private fun audio(): EncodedAudio {
        val file = File.createTempFile("recording-", ".ogg").apply { writeText("OggS fake opus"); deleteOnExit() }
        return EncodedAudio(file, "audio/ogg", "opus")
    }

    private fun failure(block: suspend () -> Unit): ServerTranscriptionException {
        val error = runCatching { runBlocking { block() } }.exceptionOrNull()
        assertThat(error).isInstanceOf(ServerTranscriptionException::class.java)
        return error as ServerTranscriptionException
    }

    @Test
    fun `uploads the recording as an OpenAI multipart request`() = runBlocking<Unit> {
        respond("/v1/audio/transcriptions", 200, """{"text": "Hello there."}""")
        val upload = audio()

        val text = client.transcribe(endpoint, "secret", "parakeet-unified", upload)

        assertThat(text).isEqualTo("Hello there.")
        val request = requests.poll(1, TimeUnit.SECONDS)!!
        assertThat(request.method).isEqualTo("POST")
        assertThat(request.path).isEqualTo("/v1/audio/transcriptions")
        assertThat(request.authorization).isEqualTo("Bearer secret")
        assertThat(request.contentType).startsWith("multipart/form-data; boundary=")
        val body = String(request.body, Charsets.ISO_8859_1)
        assertThat(body).contains(
            """Content-Disposition: form-data; name="file"; filename="${upload.file.name}"""",
            "Content-Type: audio/ogg",
            "OggS fake opus",
            "Content-Disposition: form-data; name=\"model\"\r\n\r\nparakeet-unified\r\n",
            "Content-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n",
            "Content-Disposition: form-data; name=\"language\"\r\n\r\nen\r\n",
        )
    }

    @Test
    fun `omits authorization without an API key`() = runBlocking<Unit> {
        respond("/v1/audio/transcriptions", 200, """{"text": ""}""")

        assertThat(client.transcribe(endpoint, " ", "m", audio())).isEmpty()
        assertThat(requests.poll(1, TimeUnit.SECONDS)!!.authorization).isNull()
    }

    @Test
    fun `lists models in server order`() = runBlocking<Unit> {
        respond("/v1/models", 200, """{"object":"list","data":[{"id":"parakeet-unified"},{"id":"whisper"}]}""")

        assertThat(client.listModels(endpoint, "secret")).containsExactly("parakeet-unified", "whisper")
        val request = requests.poll(1, TimeUnit.SECONDS)!!
        assertThat(request.method).isEqualTo("GET")
        assertThat(request.path).isEqualTo("/v1/models")
        assertThat(request.authorization).isEqualTo("Bearer secret")
    }

    @Test
    fun `an empty model list is reported`() {
        respond("/v1/models", 200, """{"object":"list","data":[]}""")

        assertThat(failure { client.listModels(endpoint, "") }.kind)
            .isEqualTo(ServerTranscriptionException.Kind.NO_MODELS)
    }

    @Test
    fun `a rejected key is reported as unauthorized`() {
        respond("/v1/models", 401, """{"error":{"message":"Invalid API key"}}""")

        val error = failure { client.listModels(endpoint, "wrong") }

        assertThat(error.kind).isEqualTo(ServerTranscriptionException.Kind.UNAUTHORIZED)
        assertThat(error.httpStatus).isEqualTo(401)
        assertThat(error.detail).isEqualTo("Invalid API key")
    }

    @Test
    fun `server errors carry status and detail, but the message stays loggable`() {
        respond("/v1/audio/transcriptions", 500, """{"detail":"decoder crashed on: some words"}""")

        val error = failure { client.transcribe(endpoint, "", "m", audio()) }

        assertThat(error.kind).isEqualTo(ServerTranscriptionException.Kind.HTTP)
        assertThat(error.httpStatus).isEqualTo(500)
        assertThat(error.detail).isEqualTo("decoder crashed on: some words")
        assertThat(error.message).isEqualTo("server request failed: HTTP 500")
    }

    @Test
    fun `a response without text is malformed`() {
        respond("/v1/audio/transcriptions", 200, """{"segments":[]}""")

        assertThat(failure { client.transcribe(endpoint, "", "m", audio()) }.kind)
            .isEqualTo(ServerTranscriptionException.Kind.BAD_RESPONSE)
    }

    @Test
    fun `a slow server times out`() {
        respond("/v1/audio/transcriptions", 200, """{"text":"late"}""", delayMs = 3_000)

        assertThat(failure { client.transcribe(endpoint, "", "m", audio()) }.kind)
            .isEqualTo(ServerTranscriptionException.Kind.TIMEOUT)
    }

    @Test
    fun `a closed port is unreachable`() {
        val port = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).use { it.localPort }
        val closed = ServerEndpoint.parse("http://127.0.0.1:$port")!!

        assertThat(failure { client.listModels(closed, "") }.kind)
            .isEqualTo(ServerTranscriptionException.Kind.UNREACHABLE)
    }

    @Test
    fun `parses transcription and model responses strictly`() {
        assertThat(ServerTranscriptionClient.parseTranscription("""{"text":"Hi."}""")).isEqualTo("Hi.")
        for (body in listOf("""{"text":null}""", """{"text":3}""", "{}", "not json", "")) {
            assertThat(runCatching { ServerTranscriptionClient.parseTranscription(body) }.exceptionOrNull())
                .describedAs(body)
                .isInstanceOf(ServerTranscriptionException::class.java)
        }
        assertThat(ServerTranscriptionClient.parseModels("""{"data":[{"id":"a"},{"id":""},{"name":"b"},3,{"id":"c"}]}"""))
            .containsExactly("a", "c")
        assertThat(runCatching { ServerTranscriptionClient.parseModels("""{"models":[]}""") }.exceptionOrNull())
            .isInstanceOf(ServerTranscriptionException::class.java)
    }

    @Test
    fun `extracts error details from common server shapes`() {
        assertThat(ServerTranscriptionClient.errorDetail("""{"error":{"message":"bad model"}}""")).isEqualTo("bad model")
        assertThat(ServerTranscriptionClient.errorDetail("""{"detail":"Not Found"}""")).isEqualTo("Not Found")
        assertThat(ServerTranscriptionClient.errorDetail("Bad Gateway")).isEqualTo("Bad Gateway")
        assertThat(ServerTranscriptionClient.errorDetail("")).isNull()
        assertThat(ServerTranscriptionClient.errorDetail("x".repeat(500))).hasSize(200)
    }
}

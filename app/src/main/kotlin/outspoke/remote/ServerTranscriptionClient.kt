package dev.brgr.outspoke.remote

import kotlinx.coroutines.suspendCancellableCoroutine
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.Response
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import java.io.InterruptedIOException
import java.net.UnknownServiceException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

private const val CONNECT_TIMEOUT_MS = 1_500L
private const val MODELS_TIMEOUT_MS = 5_000L
private const val TRANSCRIBE_TIMEOUT_MS = 30_000L
private const val MAX_RESPONSE_BYTES = 1_000_000L
private const val MAX_DETAIL_CHARS = 200

/**
 * A failed server request. The message names only the [kind] and HTTP status, so it is
 * safe to log; [detail] is the server's own error text, for display in settings only.
 */
class ServerTranscriptionException(
    val kind: Kind,
    val httpStatus: Int = 0,
    val detail: String? = null,
    cause: Throwable? = null,
) : IOException("server request failed: $kind${if (httpStatus != 0) " $httpStatus" else ""}", cause) {
    enum class Kind { INVALID_URL, UNREACHABLE, TIMEOUT, CLEARTEXT_BLOCKED, UNAUTHORIZED, HTTP, BAD_RESPONSE, NO_MODELS }
}

/** Speaks the OpenAI-compatible `/v1/models` and `/v1/audio/transcriptions` API. */
class ServerTranscriptionClient(
    httpClient: () -> OkHttpClient = {
        OkHttpClient.Builder().connectTimeout(CONNECT_TIMEOUT_MS, TimeUnit.MILLISECONDS).build()
    },
    private val modelsTimeoutMs: Long = MODELS_TIMEOUT_MS,
    private val transcribeTimeoutMs: Long = TRANSCRIBE_TIMEOUT_MS,
) {
    // The keyboard creates a client even when the feature is off.
    private val http by lazy(httpClient)

    /** Model ids in the order the server lists them. */
    suspend fun listModels(endpoint: ServerEndpoint, apiKey: String): List<String> {
        val request = Request.Builder().url(endpoint.models).authorization(apiKey).get().build()
        val models = parseModels(execute(request, modelsTimeoutMs))
        if (models.isEmpty()) throw ServerTranscriptionException(ServerTranscriptionException.Kind.NO_MODELS)
        return models
    }

    /** Uploads one complete recording and returns the server's text. */
    suspend fun transcribe(
        endpoint: ServerEndpoint,
        apiKey: String,
        model: String,
        audio: EncodedAudio,
        language: String = "en",
    ): String {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", audio.file.name, audio.file.asRequestBody(audio.mimeType.toMediaType()))
            .addFormDataPart("model", model)
            .addFormDataPart("response_format", "json")
            .addFormDataPart("language", language)
            .build()
        val request = Request.Builder().url(endpoint.transcriptions).authorization(apiKey).post(body).build()
        return parseTranscription(execute(request, transcribeTimeoutMs))
    }

    private fun Request.Builder.authorization(apiKey: String): Request.Builder =
        apply { if (apiKey.isNotBlank()) header("Authorization", "Bearer ${apiKey.trim()}") }

    private suspend fun execute(request: Request, timeoutMs: Long): String {
        val call = http.newCall(request)
        call.timeout().timeout(timeoutMs, TimeUnit.MILLISECONDS)
        val response = try {
            call.await()
        } catch (e: UnknownServiceException) {
            // Android refuses plain http when the network security policy forbids it.
            throw ServerTranscriptionException(ServerTranscriptionException.Kind.CLEARTEXT_BLOCKED, cause = e)
        } catch (e: InterruptedIOException) {
            throw ServerTranscriptionException(ServerTranscriptionException.Kind.TIMEOUT, cause = e)
        } catch (e: IOException) {
            throw ServerTranscriptionException(ServerTranscriptionException.Kind.UNREACHABLE, cause = e)
        }
        return response.use {
            val body = try {
                it.peekBody(MAX_RESPONSE_BYTES).string()
            } catch (e: InterruptedIOException) {
                throw ServerTranscriptionException(ServerTranscriptionException.Kind.TIMEOUT, cause = e)
            } catch (e: IOException) {
                throw ServerTranscriptionException(ServerTranscriptionException.Kind.UNREACHABLE, cause = e)
            }
            when {
                it.isSuccessful -> body
                it.code == 401 || it.code == 403 -> throw ServerTranscriptionException(
                    ServerTranscriptionException.Kind.UNAUTHORIZED, it.code, errorDetail(body),
                )
                else -> throw ServerTranscriptionException(
                    ServerTranscriptionException.Kind.HTTP, it.code, errorDetail(body),
                )
            }
        }
    }

    companion object {
        internal fun parseTranscription(body: String): String {
            val text = try {
                JSONObject(body).opt("text")
            } catch (e: JSONException) {
                throw ServerTranscriptionException(ServerTranscriptionException.Kind.BAD_RESPONSE, cause = e)
            }
            return text as? String
                ?: throw ServerTranscriptionException(ServerTranscriptionException.Kind.BAD_RESPONSE)
        }

        internal fun parseModels(body: String): List<String> {
            val data = try {
                JSONObject(body).optJSONArray("data")
            } catch (e: JSONException) {
                throw ServerTranscriptionException(ServerTranscriptionException.Kind.BAD_RESPONSE, cause = e)
            } ?: throw ServerTranscriptionException(ServerTranscriptionException.Kind.BAD_RESPONSE)
            return (0 until data.length()).mapNotNull { index ->
                (data.optJSONObject(index)?.opt("id") as? String)?.takeIf { it.isNotBlank() }
            }
        }

        /** OpenAI servers send `{"error":{"message":…}}`; FastAPI servers send `{"detail":…}`. */
        internal fun errorDetail(body: String): String? {
            val message = try {
                val json = JSONObject(body)
                json.optJSONObject("error")?.opt("message") as? String ?: json.opt("detail") as? String
            } catch (e: JSONException) {
                body
            }
            return message?.trim()?.take(MAX_DETAIL_CHARS)?.takeIf { it.isNotEmpty() }
        }
    }
}

/** Cancelling the coroutine cancels the HTTP call, so an abandoned upload stops sending. */
private suspend fun Call.await(): Response = suspendCancellableCoroutine { continuation ->
    continuation.invokeOnCancellation { cancel() }
    enqueue(object : Callback {
        override fun onFailure(call: Call, e: IOException) {
            continuation.resumeWithException(e)
        }

        override fun onResponse(call: Call, response: Response) {
            continuation.resume(response) { _, value, _ -> value.close() }
        }
    })
}

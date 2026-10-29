package dev.brgr.outspoke.remote

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class ServerEndpointTest {
    private fun v1(input: String) = ServerEndpoint.parse(input)?.toString()

    @Test
    fun `appends v1 to a server root`() {
        assertThat(v1("http://100.64.0.1:8000")).isEqualTo("http://100.64.0.1:8000/v1")
        assertThat(v1("http://100.64.0.1:8000/")).isEqualTo("http://100.64.0.1:8000/v1")
        assertThat(v1("https://asr.tail1234.ts.net/proxy")).isEqualTo("https://asr.tail1234.ts.net/proxy/v1")
    }

    @Test
    fun `keeps an existing v1 and drops trailing slashes, query and fragment`() {
        assertThat(v1("http://vps:8000/v1")).isEqualTo("http://vps:8000/v1")
        assertThat(v1("  http://vps:8000/v1/  ")).isEqualTo("http://vps:8000/v1")
        assertThat(v1("http://vps:8000/v1?x=1#top")).isEqualTo("http://vps:8000/v1")
    }

    @Test
    fun `assumes http for a bare host`() {
        assertThat(v1("vps:8000")).isEqualTo("https://vps:8000/v1")
        assertThat(v1("100.64.0.1:8000/v1")).isEqualTo("https://100.64.0.1:8000/v1")
    }

    @Test
    fun `accepts pasted endpoint URLs`() {
        assertThat(v1("http://vps:8000/v1/audio/transcriptions")).isEqualTo("http://vps:8000/v1")
        assertThat(v1("http://vps:8000/v1/models")).isEqualTo("http://vps:8000/v1")
    }

    @Test
    fun `derives endpoint URLs`() {
        val endpoint = ServerEndpoint.parse("http://vps:8000")!!
        assertThat(endpoint.models.toString()).isEqualTo("http://vps:8000/v1/models")
        assertThat(endpoint.transcriptions.toString()).isEqualTo("http://vps:8000/v1/audio/transcriptions")
    }

    @Test
    fun `rejects blank and non-http input`() {
        assertThat(v1("")).isNull()
        assertThat(v1("   ")).isNull()
        assertThat(v1("ftp://vps:21")).isNull()
        assertThat(v1("http://")).isNull()
        assertThat(v1("not a url")).isNull()
    }

    @Test
    fun `settings are off without a parsable URL`() {
        assertThat(ServerTranscriptionSettings().endpoint).isNull()
        assertThat(ServerTranscriptionSettings(baseUrl = "ftp://vps").endpoint).isNull()
        assertThat(ServerTranscriptionSettings(baseUrl = "vps:8000").endpoint).isNotNull()
    }
}

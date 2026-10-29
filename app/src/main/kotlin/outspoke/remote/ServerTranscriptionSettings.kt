package dev.brgr.outspoke.remote

/** How long the phone's ready text may wait for the server, from the end of capture. */
const val DEFAULT_SERVER_GRACE_MS = 0L
const val MAX_SERVER_GRACE_MS = 10_000L

/** Server transcription is off unless [baseUrl] parses as an HTTP(S) endpoint. */
data class ServerTranscriptionSettings(
    val baseUrl: String = "",
    /** Blank means the first model the server lists. */
    val model: String = "",
    val graceMs: Long = DEFAULT_SERVER_GRACE_MS,
) {
    val endpoint: ServerEndpoint? by lazy { ServerEndpoint.parse(baseUrl) }
}

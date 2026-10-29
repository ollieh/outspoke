package dev.brgr.outspoke.remote

import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

private val SCHEME = Regex("^[a-zA-Z][a-zA-Z0-9+.-]*://")

/** Root of an OpenAI-compatible API. [v1] always ends in `/v1`. */
class ServerEndpoint private constructor(val v1: HttpUrl) {
    val models: HttpUrl get() = v1.newBuilder().addPathSegment("models").build()
    val transcriptions: HttpUrl get() = v1.newBuilder().addPathSegments("audio/transcriptions").build()

    override fun equals(other: Any?) = other is ServerEndpoint && other.v1 == v1
    override fun hashCode() = v1.hashCode()
    override fun toString() = v1.toString()

    companion object {
        /**
         * Accepts what a user is likely to paste: a bare host (assumed `https`, since
         * the app refuses cleartext), a base URL with or without `/v1`, or a full
         * endpoint URL. Like OpenAI clients configured with a server root, `/v1` is
         * appended when missing. Returns null for blank or non-HTTP input.
         */
        fun parse(input: String): ServerEndpoint? {
            val trimmed = input.trim()
            if (trimmed.isEmpty()) return null
            val withScheme = if (SCHEME.containsMatchIn(trimmed)) trimmed else "https://$trimmed"
            val url = withScheme.toHttpUrlOrNull() ?: return null
            var segments = url.pathSegments.filter { it.isNotEmpty() }
            if (segments.takeLast(3) == listOf("v1", "audio", "transcriptions")) segments = segments.dropLast(2)
            if (segments.takeLast(2) == listOf("v1", "models")) segments = segments.dropLast(1)
            if (segments.lastOrNull() != "v1") segments = segments + "v1"
            val builder = url.newBuilder().query(null).fragment(null).encodedPath("/")
            segments.forEach { builder.addPathSegment(it) }
            return ServerEndpoint(builder.build())
        }
    }
}

package dev.brgr.outspoke.remote

import android.content.Context
import dev.brgr.outspoke.inference.RecordingTrace
import dev.brgr.outspoke.settings.preferences.ServerApiKeyStore
import dev.brgr.outspoke.settings.preferences.ServerTranscriptionPreferences
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.job
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Optional transcription on the user's own OpenAI-compatible server, raced against the
 * phone. Off unless a server URL is configured. The server model is English-only.
 */
class ServerTranscriber internal constructor(
    val settings: Flow<ServerTranscriptionSettings>,
    private val transcribe: suspend (ServerTranscriptionSettings, PcmRecording, Long) -> String,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
) {
    /**
     * Starts a race for one recording, or returns null when no server is configured or
     * the recording is not in English. Its upload is cancelled when [owner] completes.
     */
    internal fun startSession(
        owner: CoroutineScope,
        settings: ServerTranscriptionSettings?,
        recordingId: Long,
        language: String,
        formatNumbersAsDigits: Boolean,
        customWords: List<String>,
    ): ServerRaceSession? {
        if (settings?.endpoint == null) return null
        if (!language.startsWith("en")) {
            RecordingTrace.log { "server skipped session=$recordingId reason=language-$language" }
            return null
        }
        val scope = CoroutineScope(SupervisorJob() + dispatcher)
        owner.coroutineContext.job.invokeOnCompletion { scope.cancel() }
        val formatter = ServerTextFormatter(formatNumbersAsDigits, customWords)
        return ServerRaceSession(scope, settings.graceMs, recordingId, transcribe = { recording ->
            formatter.format(transcribe(settings, recording, recordingId))
        })
    }

    companion object {
        fun create(context: Context): ServerTranscriber {
            val app = context.applicationContext
            val keys = ServerApiKeyStore(app)
            val client = ServerTranscriptionClient()
            val encoder = RecordingEncoder(File(app.cacheDir, "server-upload"))
            // First listed model per server, when none is configured.
            val models = ConcurrentHashMap<ServerEndpoint, String>()
            return ServerTranscriber(ServerTranscriptionPreferences(app).settings, { settings, recording, recordingId ->
                val endpoint = checkNotNull(settings.endpoint)
                val encodeStart = monotonicMs()
                val audio = encoder.encode(recording)
                RecordingTrace.log {
                    "server encoded session=$recordingId codec=${audio.codec} bytes=${audio.file.length()} " +
                        "audioMs=${recording.durationMs} encodeMs=${monotonicMs() - encodeStart}"
                }
                try {
                    val apiKey = keys.get()
                    val model = settings.model.ifBlank {
                        models[endpoint] ?: client.listModels(endpoint, apiKey).first().also { models[endpoint] = it }
                    }
                    client.transcribe(endpoint, apiKey, model, audio, language = "en")
                } finally {
                    audio.file.delete()
                }
            })
        }
    }
}

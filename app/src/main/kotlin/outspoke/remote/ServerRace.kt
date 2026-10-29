package dev.brgr.outspoke.remote

import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.inference.RecordingTrace
import dev.brgr.outspoke.inference.TranscriptSource
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull

/** Bounds encoding plus upload; the HTTP call has its own, shorter limit. */
private const val UPLOAD_TIMEOUT_MS = 35_000L

internal fun monotonicMs(): Long = System.nanoTime() / 1_000_000

/** What the server returned and when, on the session clock. */
internal sealed interface ServerOutcome {
    val atMs: Long

    data class Text(val text: String, override val atMs: Long) : ServerOutcome
    data class Failed(val reason: String, override val atMs: Long) : ServerOutcome
}

/** [text] is what the keyboard inserts; null when neither side produced text. */
internal data class RaceDecision(val text: String?, val source: TranscriptSource?, val reason: String)

/**
 * The user's race rule. Capture ends at `stoppedAt`; the phone's text is ready at
 * `localReadyAt`. The server wins if its text arrives by `max(stoppedAt + grace, localReadyAt)`:
 * before the phone is ready it always wins, afterwards it has until the grace period ends.
 */
internal object ServerRacePolicy {
    fun deadlineMs(stoppedAtMs: Long, graceMs: Long, localReadyAtMs: Long): Long =
        maxOf(stoppedAtMs + graceMs, localReadyAtMs)

    /**
     * [server] is the outcome observed by the time the race was decided, or null if the
     * upload was still running. [localText] is null when the phone failed or heard nothing;
     * then any server text wins, however late.
     */
    fun decide(
        localText: String?,
        server: ServerOutcome?,
        stoppedAtMs: Long,
        graceMs: Long,
        localReadyAtMs: Long,
    ): RaceDecision {
        val serverText = (server as? ServerOutcome.Text)?.text?.takeIf { it.isNotBlank() }
        if (localText == null) {
            return if (serverText != null) RaceDecision(serverText, TranscriptSource.SERVER, "phone-failed")
            else RaceDecision(null, null, "both-failed")
        }
        if (serverText != null && server.atMs <= deadlineMs(stoppedAtMs, graceMs, localReadyAtMs)) {
            val reason = if (server.atMs <= localReadyAtMs) "server-first" else "server-in-grace"
            return RaceDecision(serverText, TranscriptSource.SERVER, reason)
        }
        val reason = when {
            server == null -> "server-late"
            server is ServerOutcome.Failed -> "server-error:${server.reason}"
            serverText == null -> "server-empty"
            else -> "server-late"
        }
        return RaceDecision(localText, TranscriptSource.PHONE, reason)
    }
}

/**
 * One recording's race between the phone and the server.
 *
 * The capture layer tees raw audio into [pcm]. When capture completes normally,
 * [onAudioComplete] uploads the whole recording at once, while the phone finishes
 * decoding. [choose] then applies [ServerRacePolicy]. All work runs in [scope], which
 * the owner cancels when the recording ends, so a losing upload never outlives it.
 */
internal class ServerRaceSession(
    private val scope: CoroutineScope,
    private val graceMs: Long,
    private val recordingId: Long,
    private val transcribe: suspend (PcmRecording) -> String,
    private val clock: () -> Long = ::monotonicMs,
    val pcm: RecordingPcmBuffer = RecordingPcmBuffer(),
) {
    private var stoppedAtMs = 0L
    private var request: Deferred<ServerOutcome>? = null
    private var skipped = "audio-incomplete"
    private var decided = false

    fun onAudioComplete() {
        val recording = synchronized(this) {
            if (decided || request != null) return
            stoppedAtMs = clock()
            if (pcm.overflowed) {
                skipped = "over-${MAX_RECORDING_SAMPLES / 16_000 / 60}-minutes"
                return
            }
            pcm.snapshot().takeIf { it.sampleCount > 0 } ?: run {
                skipped = "no-audio"
                return
            }
        }
        RecordingTrace.log { "server upload begin session=$recordingId samples=${recording.sampleCount} audioMs=${recording.durationMs}" }
        val upload = scope.async {
            try {
                val text = withTimeout(UPLOAD_TIMEOUT_MS) { transcribe(recording) }
                val at = clock()
                RecordingTrace.log { "server response session=$recordingId status=ok latencyMs=${at - stoppedAtMs} chars=${text.length}" }
                ServerOutcome.Text(text, at)
            } catch (e: Exception) {
                currentCoroutineContext().ensureActive()
                val reason = when (e) {
                    is ServerTranscriptionException -> e.kind.name.lowercase() + if (e.httpStatus != 0) "-${e.httpStatus}" else ""
                    is CancellationException -> "timeout"
                    else -> e.javaClass.simpleName
                }
                val at = clock()
                RecordingTrace.log { "server response session=$recordingId status=error reason=$reason latencyMs=${at - stoppedAtMs}" }
                ServerOutcome.Failed(reason, at)
            }
        }
        synchronized(this) { request = upload }
    }

    /**
     * Called once the phone's text is final: [localText], or null when the phone failed
     * or heard nothing. Waits only as long as the race rule allows, then abandons the upload.
     */
    suspend fun choose(localText: String?): RaceDecision {
        val readyAtMs = clock()
        val pending = synchronized(this) {
            decided = true
            request
        }
        if (pending == null) {
            val decision = RaceDecision(localText, localText?.let { TranscriptSource.PHONE }, "server-skipped:$skipped")
            RecordingTrace.log { "server race session=$recordingId winner=${decision.source} reason=${decision.reason}" }
            return decision
        }
        val outcome = try {
            // With no phone text, wait for the server; the upload itself is time-bounded.
            if (pending.isCompleted || localText == null) pending.await()
            else withTimeoutOrNull(ServerRacePolicy.deadlineMs(stoppedAtMs, graceMs, readyAtMs) - readyAtMs) {
                pending.await()
            }
        } catch (e: CancellationException) {
            currentCoroutineContext().ensureActive()
            ServerOutcome.Failed("cancelled", clock())
        }
        pending.cancel()
        val decision = ServerRacePolicy.decide(localText, outcome, stoppedAtMs, graceMs, readyAtMs)
        RecordingTrace.log {
            "server race session=$recordingId winner=${decision.source} reason=${decision.reason} " +
                "localReadyMs=${readyAtMs - stoppedAtMs} serverMs=${outcome?.let { it.atMs - stoppedAtMs }} " +
                "waitedMs=${clock() - readyAtMs} localChars=${localText?.length} chars=${decision.text?.length}"
        }
        return decision
    }
}

/** Starts the upload when capture completes normally; a cancelled capture uploads nothing. */
internal fun Flow<AudioChunk>.uploadWhenComplete(race: ServerRaceSession?): Flow<AudioChunk> =
    if (race == null) this else onCompletion { cause -> if (cause == null) race.onAudioComplete() }

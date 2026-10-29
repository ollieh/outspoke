package dev.brgr.outspoke.audio

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import dev.brgr.outspoke.R
import dev.brgr.outspoke.inference.RecordingTrace
import dev.brgr.outspoke.settings.preferences.AppPreferences
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.log10
import kotlin.math.sqrt
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "AudioCaptureManager"

/** 16 kHz mono - matches Parakeet V3's expected input format. */
private const val SAMPLE_RATE = 16_000

/** 30 ms window at 16 kHz = 480 samples per chunk. Matches Silero VAD's required frame size. */
private const val CHUNK_SAMPLES = 480

/**
 * AudioRecord ring buffer: 2 s of 16-bit mono. The platform minimum is ~60 ms, so a
 * short stall of the read thread (GC, CPU contention while a model loads) used to
 * overwrite unread audio silently. Blocking reads still return every 30 ms chunk.
 */
private const val RECORD_BUFFER_BYTES = SAMPLE_RATE * 2 * 2

/**
 * Maximum silence frames pumped through the VAD during the trailing drain phase.
 * 20 frames × 30 ms = 600 ms - generously above [RMSVadFilter]'s 15-frame / 450 ms hangover
 * so the hangover always expires before this cap is hit.
 */
private const val HANGOVER_DRAIN_SAFETY_FRAMES = 20

/**
 * dBFS level at which the waveform bar reads 0.0. Levels below this floor
 * (typical device ambient noise) are inaudible on the bar.
 */
private const val NOISE_FLOOR_DB = -60f

/**
 * Display gain applied to the normalised waveform amplitude after the dB mapping.
 * Purely cosmetic: [amplitude] only feeds the [WaveformBar] animation. Typical speech
 * (-50..-20 dBFS) otherwise lands at ~0.2..0.7, which reads as "low gain" on the bar;
 * 1.5× lifts it to ~0.3..1.0 with loud speech reaching full height.
 */
private const val WAVEFORM_DISPLAY_GAIN = 1.5f

// Capture source: MediaRecorder.AudioSource.DEFAULT — the vendor-recommended source.
// It arrives at a usable level with the platform's standard voice processing, so no
// application-side gain is applied.
//
// History: rawer sources (VOICE_RECOGNITION, UNPROCESSED) with application-side gain
// were tried in 2026-08 on the theory that the TDT decoder was level- or shape-
// sensitive. That theory was disproven: the nemo128 frontend normalises each window to
// a fixed level, so the decoder is level-invariant (the same waveform at 0.1x / 0.5x /
// 1.0x yields byte-identical features). The short-word ("thanks") reliability gap was
// a decode-loop bug in ParakeetEngine — the predictor state must freeze on blank and
// the initial target must be blank — now fixed. DEFAULT is good enough; per-mic
// calibration in settings remains an optional refinement.
//
// Exception - [rawMicCapture] preference: `AudioSource.UNPROCESSED` is available
// (via startCapture(rawSource = true)) to bypass the platform voice processing
// entirely. The default source's AEC cancels audio played from the device's own
// speaker out of the mic stream, which makes the speakerphone use case
// (e.g. transcribing a voicemail playing on speaker) transcribe as silence.
// The TDT decoder is level-invariant, so the rawer level of UNPROCESSED does
// not affect transcription quality.

/**
 * Manages a single [AudioRecord] session and emits captured audio as a cold
 * [Flow]<[AudioChunk]>. Runs entirely on [Dispatchers.IO].
 *
 * Typical usage:
 * ```kotlin
 * val job = scope.launch {
 *     manager.startCapture().collect { chunk -> } // forward to inference
 * }
 * job.cancel() // stops recording and releases AudioRecord
 * ```
 */
class AudioCaptureManager(private val context: Context) {

    /** Calibration-selected microphone, applied to each [AudioRecord] via setPreferredDevice. */
    private val prefs = AppPreferences(context.applicationContext)

    private val _amplitude = MutableStateFlow(0f)

    private class ActiveCapture(val recordingId: Long, val stop: AtomicBoolean)

    /**
     * The most recent [startCapture]. Its flag is set by [stopCapture] to end the read loop
     * on the next iteration. Using a flag instead of coroutine cancellation lets the [Flow]
     * complete normally so that the inference layer can emit a final result before tearing down.
     */
    @Volatile
    private var active: ActiveCapture? = null

    /**
     * Signals the active [startCapture] flow to exit its read loop and complete normally.
     * The flow will finish within one 30 ms chunk read cycle.
     */
    fun stopCapture() {
        val capture = active ?: return
        RecordingTrace.log { "capture stop requested session=${capture.recordingId}" }
        capture.stop.set(true)
    }

    /** Stops [recordingId] only if it is still the active capture, never a newer one. */
    fun stopCapture(recordingId: Long) {
        val capture = active?.takeIf { it.recordingId == recordingId } ?: return
        RecordingTrace.log { "capture stop requested session=$recordingId" }
        capture.stop.set(true)
    }

    /**
     * Normalised RMS amplitude of the most recently captured chunk, in the range [0.0, 1.0].
     * Resets to 0.0 when capture stops.
     */
    val amplitude: StateFlow<Float> = _amplitude.asStateFlow()

    /**
     * Starts microphone capture and emits each 30 ms [AudioChunk] downstream.
     *
     * **Prerequisites:** [PermissionHelper.hasRecordPermission] must return `true` before
     * calling this. If the permission is absent a [SecurityException] is thrown immediately
     * so the caller can transition the UI to an error state.
     *
     * The flow is cold - a new [AudioRecord] is created per collection. The [AudioRecord]
     * is always released in the `finally` block, even if the collector cancels mid-stream.
     *
     * A dedicated reader drains [AudioRecord] into an unbounded queue, so VAD work and a
     * slow collector never delay a read. Recording starts before the VAD model is created.
     *
     * @param onAudioStarted Called once, on the reader thread, when the microphone first
     *   delivers samples. Speech before this point was never captured.
     * @throws SecurityException if [android.Manifest.permission.RECORD_AUDIO] is not granted.
     * @throws IllegalStateException if [AudioRecord] fails to initialise.
     */
    // Permission is checked manually via PermissionHelper before AudioRecord is created.
    @SuppressLint("MissingPermission")
    fun startCapture(
        vadEnabled: Boolean = true,
        rawSource: Boolean = false,
        recordingId: Long = RecordingTrace.nextSessionId(),
        onAudioStarted: () -> Unit = {},
    ): Flow<AudioChunk> {
        // Allocate before collection so releasing the button during startup is not lost.
        val stopRequested = AtomicBoolean(false)
        active = ActiveCapture(recordingId, stopRequested)
        return channelFlow {
            if (!PermissionHelper.hasRecordPermission(context)) {
                throw SecurityException(
                    "RECORD_AUDIO permission is not granted. " +
                            "Open the Outspoke app to grant microphone access."
                )
            }

            if (stopRequested.get()) return@channelFlow

            val minBufferBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            val bufferBytes = maxOf(minBufferBytes, RECORD_BUFFER_BYTES)

            val source = if (rawSource) MediaRecorder.AudioSource.UNPROCESSED
            else MediaRecorder.AudioSource.DEFAULT

            val recorder = AudioRecord(
                source,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferBytes,
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                val state = recorder.state
                recorder.release()
                throw IllegalStateException("AudioRecord failed to initialise (state=$state)")
            }

            // Apply the calibration-selected microphone (no-op when unset or the device is gone).
            applyPreferredMic(recorder)

            val frames = Channel<ShortArray>(Channel.UNLIMITED)
            val deadObject = AtomicBoolean(false)
            launch {
                var capturedSamples = 0L
                try {
                    recorder.startRecording()
                    RecordingTrace.log { "capture start session=$recordingId recorder=${recorder.audioSessionId} vad=$vadEnabled raw=$rawSource" }
                    Log.d(TAG, "AudioRecord started - chunk=$CHUNK_SAMPLES samples, buf=$bufferBytes bytes")

                    while (isActive && !stopRequested.get()) {
                        val buffer = ShortArray(CHUNK_SAMPLES)
                        val read = recorder.read(buffer, 0, buffer.size)
                        when {
                            read > 0 -> {
                                if (capturedSamples == 0L) {
                                    RecordingTrace.log { "capture first audio session=$recordingId" }
                                    onAudioStarted()
                                }
                                capturedSamples += read
                                frames.send(if (read == buffer.size) buffer else buffer.copyOf(read))
                            }

                            read == AudioRecord.ERROR_DEAD_OBJECT -> {
                                // Hardware-level error - the audio subsystem was taken away.
                                Log.e(TAG, "AudioRecord ERROR_DEAD_OBJECT - stopping capture")
                                deadObject.set(true)
                                break
                            }

                            read < 0 -> {
                                // Non-fatal read error - log and skip this chunk.
                                Log.w(TAG, "AudioRecord read returned error code: $read")
                            }
                            // read == 0: no data yet; continue the loop
                        }
                    }

                    // When the user releases the record key, samples may still sit in the
                    // AudioRecord ring buffer. Drain them (non-blocking) so the last word
                    // is never cut off mid-phoneme.
                    if (stopRequested.get() && !deadObject.get() && isActive) {
                        while (true) {
                            val buffer = ShortArray(CHUNK_SAMPLES)
                            val drained = recorder.read(buffer, 0, buffer.size, AudioRecord.READ_NON_BLOCKING)
                            if (drained <= 0) break
                            capturedSamples += drained
                            frames.send(buffer.copyOf(drained))
                        }
                        RecordingTrace.log { "capture hardware drained session=$recordingId samples=$capturedSamples" }
                    }
                } finally {
                    frames.close()
                    RecordingTrace.log { "capture reader end session=$recordingId samples=$capturedSamples stop=${stopRequested.get()}" }
                    try {
                        recorder.stop()
                    } catch (e: IllegalStateException) {
                        Log.w(TAG, "AudioRecord stop failed", e)
                    }
                    recorder.release()
                    Log.d(TAG, "AudioRecord stopped and released")
                }
            }

            // Only create a filter when VAD is enabled; null means pass-through (VAD disabled).
            // The reader is already recording, so loading the VAD model costs no audio.
            val vad: VadFilter? = if (vadEnabled) {
                try {
                    val sileroBytes = context.resources.openRawResource(R.raw.silero_vad_v4).readBytes()
                    SileroVadFilter(modelBytes = sileroBytes, threshold = 0.3f)
                } catch (e: Exception) {
                    Log.w(TAG, "Silero VAD model not found in raw resources. Falling back to Energy VAD.", e)
                    RMSVadFilter(0.4f)
                }
            } else null.also { Log.d(TAG, "VAD disabled") }

            var processedSamples = 0L
            var forwardedSamples = 0L
            var nextReport = SAMPLE_RATE * 5L
            try {
                for (samples in frames) {
                    processedSamples += samples.size
                    val chunk = AudioChunk(samples = samples)
                    val rms = calculateRms(chunk.samples)
                    _amplitude.value = normaliseAmplitude(rms)

                    val toSend = vad?.process(chunk, rms) ?: listOf(chunk)
                    for (c in toSend) {
                        send(c)
                        forwardedSamples += c.samples.size
                    }
                    if (processedSamples >= nextReport) {
                        RecordingTrace.log { "capture progress session=$recordingId samples=$processedSamples forwarded=$forwardedSamples nonzero=${rms > 0f}" }
                        nextReport = processedSamples + SAMPLE_RATE * 5L
                    }
                }

                // The VAD hangover window (up to 450 ms) may not have expired yet. Feed
                // silence frames until VAD transitions back to SILENCE, so the hangover
                // tail is emitted before the flow completes.
                if (stopRequested.get() && !deadObject.get() && vad != null && vad.isSpeechActive) {
                    val silence = AudioChunk(ShortArray(CHUNK_SAMPLES))
                    var safetyFrames = HANGOVER_DRAIN_SAFETY_FRAMES
                    while (vad.isSpeechActive && safetyFrames-- > 0) {
                        val toSend = vad.process(silence, 0f)
                        for (c in toSend) {
                            send(c)
                            forwardedSamples += c.samples.size
                        }
                    }
                    Log.d(TAG, "VAD hangover drain complete (framesLeft=$safetyFrames)")
                }
            } finally {
                RecordingTrace.log { "capture end session=$recordingId samples=$processedSamples forwarded=$forwardedSamples stop=${stopRequested.get()}" }
                vad?.flush()
                vad?.close()
                _amplitude.value = 0f
            }
        }.flowOn(Dispatchers.IO)
    }

    /**
     * Computes the RMS amplitude of [samples] and normalises it to [0.0, 1.0] relative to
     * [Short.MAX_VALUE] (32 767).
     */
    private fun calculateRms(samples: ShortArray): Float {
        if (samples.isEmpty()) return 0f
        val sumOfSquares = samples.fold(0.0) { acc, s -> acc + s.toDouble() * s.toDouble() }
        val rms = sqrt(sumOfSquares / samples.size)
        return (rms / Short.MAX_VALUE.toDouble()).toFloat().coerceIn(0f, 1f)
    }

    /**
     * Applies the calibration-selected microphone to [recorder] via `setPreferredDevice`
     * (API 29+). No-op when no calibration has run ([preferredMicId] == 0), the API is too
     * old, or the persisted device is no longer present (e.g. after a reboot or cable change).
     */
    private suspend fun applyPreferredMic(recorder: AudioRecord) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val id = prefs.preferredMicId.first()
        if (id == 0) return
        val device = MicCalibrationManager(context).deviceForId(id) ?: run {
            Log.w(TAG, "Preferred mic id=$id not present — using system default")
            return
        }
        try {
            recorder.setPreferredDevice(device)
            Log.d(TAG, "Applied preferred mic id=$id")
        } catch (e: Exception) {
            Log.w(TAG, "setPreferredDevice failed for id=$id", e)
        }
    }

    /**
     * Returns the RMS for the waveform display, clamped to [0.0, 1.0].
     */
    private fun normaliseAmplitude(rms: Float): Float {
        if (rms <= 0f) return 0f
        // Full-scale-linear normalisation is useless for the waveform bar: normal
        // speech through a phone mic with AudioSource.DEFAULT (voice processing /
        // AGC applied) typically sits at ~1-5 % of full scale, so the bars barely
        // move. Map the dB scale instead: NOISE_FLOOR_DB dBFS -> 0.0, 0 dBFS -> 1.0,
        // which lifts the typical speech range (-50..-20 dBFS) to ~0.2..0.7.
        val db = 20f * log10(rms)
        return ((db - NOISE_FLOOR_DB) / (0f - NOISE_FLOOR_DB) * WAVEFORM_DISPLAY_GAIN).coerceIn(0f, 1f)
    }

}

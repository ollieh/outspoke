package dev.brgr.outspoke.inference

/** Represents the outcome of a single transcription attempt. */
sealed class TranscriptResult {

    /**
     * In-progress transcript from streaming inference.
     * The keyboard shows [text] in its recording preview, without editor writes.
     *
     * [confidence] is the engine's per-token geometric-mean probability in [0.0, 1.0]
     * (1.0 when the engine does not surface per-token log-probs). Used by
     * [dev.brgr.outspoke.inference.InferenceRepository] to suppress low-confidence
     * first-stride outputs that would otherwise displace real speech (cold-stride
     * hallucinations). Defaults to 1.0 so engines/tests that don't compute it are
     * treated as fully confident.
     */
    data class Partial(val text: String, val confidence: Float = 1.0f) : TranscriptResult()

    /**
     * Final, confirmed transcript for a completed utterance.
     * The keyboard accumulates [text] until the complete recording is ready to insert.
     *
     * [isUtteranceBoundary] is true when this Final was emitted mid-session by the VAD
     * silence-boundary handler. The recording session is still active and the keyboard
     * retains the utterance in its preview and continues listening.
     * When false (the default), this is a true session-ending Final.
     *
     * [confidence] is the engine's per-token geometric-mean probability in [0.0, 1.0]
     * (1.0 when not computed). See [Partial.confidence].
     */
    data class Final(
        val text: String,
        val isUtteranceBoundary: Boolean = false,
        val confidence: Float = 1.0f,
    ) : TranscriptResult()

    /** Inference failed. [cause] carries the underlying exception for logging/display. */
    data class Failure(val cause: Throwable) : TranscriptResult()

    /**
     * The rolling audio window was trimmed to prevent attention drift.
     *
     * RecordingTranscript keeps the dropped prefix in memory and reconciles the next
     * window's overlapping hypothesis. [stableWords] contains the confirmed stable prefix
     * when available; it is empty for force/silence trims. Legacy composing-text callers
     * use TextInjector.resetAfterTrim to maintain their field alignment.
     */
    data class WindowTrimmed(val stableWords: List<String> = emptyList()) : TranscriptResult()

    /**
     * The model saw audio but could not resolve a word — typically a short utterance
     * (a single word, or one buried in room noise) that fails the decode confidence gate.
     * No text is committed; the keyboard should surface a brief "didn't catch that" cue
     * instead of staying silent.
     */
    object NoSpeech : TranscriptResult()
}

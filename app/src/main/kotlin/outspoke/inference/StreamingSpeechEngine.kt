package dev.brgr.outspoke.inference

/** A model that owns its trained streaming window and decoder state. */
interface StreamingSpeechEngine : SpeechEngine {
    /** Sessions are serialized by InferenceRepository and must not outlive the engine. */
    fun newStream(): SpeechStream
}

/** One utterance. Input is mono PCM16 at 16 kHz; text is the accumulated hypothesis. */
interface SpeechStream : AutoCloseable {
    fun accept(samples: ShortArray): TranscriptResult.Partial?
    fun finish(): TranscriptResult.Final
}

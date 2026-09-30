package dev.brgr.outspoke.ui.keyboard

import dev.brgr.outspoke.ime.TranscriptAligner
import dev.brgr.outspoke.ime.TranscriptAligner.splitToWords

/** Recording-owned text. Silence completes an utterance, never an editor insertion. */
internal class RecordingTranscript {
    private val completed = mutableListOf<String>()
    private var current = emptyList<String>()
    private var windowTrimmed = false

    val text: String
        get() = (completed + current.joinToString(" ")).filter { it.isNotBlank() }.joinToString(" ")

    val hasFinalText: Boolean get() = completed.isNotEmpty()
    val finalText: String get() = completed.joinToString(" ")

    fun partial(text: String) {
        val fresh = text.splitToWords()
        current = if (windowTrimmed) mergeTrimmedWindow(current, fresh) else fresh
    }

    fun final(text: String) {
        partial(text)
        current.joinToString(" ").takeIf { it.isNotBlank() }?.let(completed::add)
        current = emptyList()
        windowTrimmed = false
    }

    fun windowTrimmed() {
        // A legacy window may drop audio without finalizing the utterance. Keep
        // its prefix in memory and reconcile the retained overlap on later results.
        if (current.isNotEmpty()) windowTrimmed = true
    }

    private fun mergeTrimmedWindow(previous: List<String>, fresh: List<String>): List<String> {
        if (previous.isEmpty()) return fresh
        if (fresh.isEmpty()) return previous
        for (overlap in minOf(previous.size, fresh.size) downTo 2) {
            if (previous.takeLast(overlap).zip(fresh.take(overlap)).all { (a, b) ->
                    TranscriptAligner.wordsMatch(a, b)
                }) {
                return previous.dropLast(overlap) + fresh
            }
        }
        val continuation = TranscriptAligner.findNewContent(previous, fresh)
        if (continuation.isNotEmpty()) return previous + continuation
        // A shorter repeated hypothesis adds nothing; unrelated output is kept
        // rather than losing words when a forced trim provides no usable overlap.
        if (fresh.size <= previous.size && previous.windowed(fresh.size).any { window ->
                window.zip(fresh).all { (a, b) -> TranscriptAligner.wordsMatch(a, b) }
            }) return previous
        return previous + fresh
    }
}

package dev.brgr.outspoke.remote

import dev.brgr.outspoke.inference.CustomWordCorrector
import dev.brgr.outspoke.inference.NumberNormaliser

/**
 * Applies the user's text preferences to the server's transcript.
 *
 * The server decodes the whole recording in one offline pass and returns punctuated,
 * capitalised English, so only preference-driven steps run, in the phone's order:
 * number formatting (when post-processing and digits are both on), then custom words.
 * The rest of the phone's cleanup exists for streaming artefacts the server cannot
 * produce (stutters and repeated phrases at chunk joins, periods at chunk ends,
 * stray leading punctuation) and would damage good punctuation: the spurious-period
 * filter deletes the period of a genuine short sentence, for example.
 */
internal class ServerTextFormatter(
    private val formatNumbersAsDigits: Boolean,
    customWords: List<String>,
) {
    private val corrector = CustomWordCorrector(customWords)

    fun format(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return ""
        val numbered = if (formatNumbersAsDigits) {
            NumberNormaliser.normalise(trimmed.split(Regex("\\s+")), language = "en").joinToString(" ")
        } else trimmed
        return corrector.correct(numbered)
    }
}

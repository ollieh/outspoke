package dev.brgr.outspoke.inference

import java.util.Locale
import dev.brgr.outspoke.ime.correction.DoubleMetaphone
import kotlin.math.abs

/** Local vocabulary correction. Compile once at recording start; never mutate mid-session. */
class CustomWordCorrector(words: Collection<String>) {
    private data class Term(val spelling: String, val key: String, val phonetic: String)
    private val terms = normalizeWords(words).map { Term(it, key(it), phonetic(key(it))) }

    fun correct(text: String): String {
        if (terms.isEmpty() || text.isBlank()) return text
        val words = Regex("\\S+").findAll(text).toList()
        val output = StringBuilder()
        var offset = 0
        var index = 0
        while (index < words.size) {
            var best: Term? = null
            var bestSize = 0
            var bestScore = 0.18
            var ambiguous = false
            for (size in minOf(3, words.size - index) downTo 1) {
                val span = words.subList(index, index + size)
                // Never join words across a comma, sentence boundary, or line break.
                if (span.dropLast(1).any { it.value.last().let { c -> !c.isLetterOrDigit() } }) continue
                val source = text.substring(span.first().range.first, span.last().range.last + 1)
                if ('\n' in source || '\r' in source) continue
                val sourceKey = key(source)
                if (sourceKey.isEmpty() || sourceKey.length > MAX_LENGTH) continue
                val sourcePhonetic = phonetic(sourceKey)
                for (term in terms) {
                    val score = score(sourceKey, sourcePhonetic, term)
                    if (score < bestScore) {
                        best = term
                        bestSize = size
                        bestScore = score
                        ambiguous = false
                    } else if (score == bestScore && best != null && term != best) {
                        ambiguous = true
                    }
                }
            }
            if (best == null || ambiguous) {
                index++
                continue
            }
            val first = words[index]
            val last = words[index + bestSize - 1]
            val leading = first.value.takeWhile { !it.isLetterOrDigit() }
            val trailing = last.value.takeLastWhile { !it.isLetterOrDigit() }
            output.append(text, offset, first.range.first)
            val savedPrefix = best.spelling.takeWhile { !it.isLetterOrDigit() }
            output.append(if (savedPrefix.isNotEmpty()) leading.removeSuffix(savedPrefix) else leading)
            output.append(best.spelling)
            // A saved term may itself end in punctuation, such as "C++".
            val savedSuffix = best.spelling.takeLastWhile { !it.isLetterOrDigit() }
            output.append(if (savedSuffix.isNotEmpty()) trailing.removePrefix(savedSuffix) else trailing)
            offset = last.range.last + 1
            index += bestSize
        }
        return output.append(text, offset, text.length).toString()
    }

    private fun score(source: String, sourcePhonetic: String, term: Term): Double {
        val target = term.key
        if (source == target) return 0.0
        // Short words and numbers are too easy to change accidentally.
        if (minOf(source.length, target.length) < 5 ||
            source.any { it.isDigit() } || target.any { it.isDigit() }
        ) return 1.0
        val length = maxOf(source.length, target.length)
        if (abs(source.length - target.length) > maxOf(length / 4, 2)) return 1.0
        val distance = editDistance(source, target)
        val phoneticWeight = if (distance <= 2 && sourcePhonetic.isNotEmpty() &&
            sourcePhonetic == term.phonetic) 0.3 else 1.0
        return distance.toDouble() / length * phoneticWeight
    }

    private fun editDistance(a: String, b: String): Int {
        var previous = IntArray(b.length + 1) { it }
        for (i in a.indices) {
            val current = IntArray(b.length + 1)
            current[0] = i + 1
            for (j in b.indices) {
                current[j + 1] = minOf(current[j] + 1, previous[j + 1] + 1,
                    previous[j] + if (a[i] == b[j]) 0 else 1)
            }
            previous = current
        }
        return previous[b.length]
    }

    companion object {
        const val MAX_WORDS = 100
        const val MAX_LENGTH = 50
        private fun key(word: String): String = word.filter { it.isLetterOrDigit() }.lowercase(Locale.ROOT)
        private fun phonetic(word: String): String = if (word.all { it in 'a'..'z' })
            DoubleMetaphone.encode(word).primary else ""

        fun normalizeWord(word: String): String = word.trim().replace(Regex("\\s+"), " ")

        fun normalizeWords(words: Collection<String>): List<String> = words.asSequence()
            .map(::normalizeWord)
            .filter { it.length <= MAX_LENGTH && it.any(Char::isLetter) && it.split(' ').size <= 3 }
            .distinctBy(::key)
            .take(MAX_WORDS)
            .toList()
    }
}

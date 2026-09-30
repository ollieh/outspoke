package dev.brgr.outspoke.inference

private const val MAX_RECOVERY_OVERLAP_TOKENS = 48

/**
 * Locate the continuation after an already-emitted suffix in a re-decoded context.
 * The context can start with a truncated or differently capitalised word, so the
 * matching suffix need not begin at its first token. Prefer the longest match.
 * A lone token or punctuation-only match is not enough evidence to append text.
 */
internal fun findStreamingRecoveryStart(
    previous: List<Int>,
    recovered: List<Int>,
    hasWord: (List<Int>) -> Boolean,
): Int? {
    if (previous.isEmpty()) return 0
    val maxOverlap = minOf(MAX_RECOVERY_OVERLAP_TOKENS, previous.size, recovered.size)
    for (count in maxOverlap downTo 2) {
        val suffix = previous.subList(previous.size - count, previous.size)
        if (!hasWord(suffix)) continue
        for (start in 0..recovered.size - count) {
            if (suffix == recovered.subList(start, start + count)) return start + count
        }
    }
    return null
}

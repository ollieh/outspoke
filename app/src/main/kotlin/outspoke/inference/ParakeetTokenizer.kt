package dev.brgr.outspoke.inference

private val SPACE_BEFORE_PUNCTUATION = Regex(" ([.,!?;:])")
private val REPEATED_SPACES = Regex(" {2,}")

/** Vocabulary entries are token text, not IDs. Preserve digits and standalone word boundaries. */
internal fun decodeParakeetTokens(tokenIds: List<Int>, vocabulary: Array<String>, blankId: Int): String =
    buildString {
        for (id in tokenIds) {
            if (id !in vocabulary.indices || id == blankId) continue
            append(vocabulary[id].replace('▁', ' '))
        }
    }.replace(SPACE_BEFORE_PUNCTUATION, "$1")
        .replace(REPEATED_SPACES, " ")
        .trim()

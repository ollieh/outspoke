package dev.brgr.outspoke.inference

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class ParakeetTokenizerTest {
    @Test
    fun `numeric tokens and separate word boundaries are preserved`() {
        val vocabulary = arrayOf("▁Number", "▁", "9", ".", "1", "0", "<blk>")
        assertThat(decodeParakeetTokens(listOf(0, 1, 2, 3), vocabulary, 6)).isEqualTo("Number 9.")
        assertThat(decodeParakeetTokens(listOf(0, 1, 4, 5, 3), vocabulary, 6)).isEqualTo("Number 10.")
    }

    @Test
    fun `digits within words ordinals and decimals are not stripped`() {
        val vocabulary = arrayOf("▁1980", "s", "▁6", "th", "▁3", ".", "1", "4", "<blk>")
        assertThat(decodeParakeetTokens((0..7).toList(), vocabulary, 8)).isEqualTo("1980s 6th 3.14")
    }

    @Test
    fun `blank and invalid tokens are ignored without losing spacing or punctuation`() {
        val vocabulary = arrayOf("▁hello", "▁", "world", "▁!", "<blk>")
        assertThat(decodeParakeetTokens(listOf(-1, 0, 4, 1, 2, 3, 99), vocabulary, 4))
            .isEqualTo("hello world!")
    }
}

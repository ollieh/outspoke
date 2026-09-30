package dev.brgr.outspoke.inference

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class StreamingTokenRecoveryTest {
    @Test
    fun `matching inside the context skips a changed leading token without repeating speech`() {
        val previous = listOf(1, 2, 3, 4)
        val recovered = listOf(9, 2, 3, 4, 5, 6)
        val start = findStreamingRecoveryStart(previous, recovered) { true }
        assertThat(previous + recovered.drop(start!!)).containsExactly(1, 2, 3, 4, 5, 6)
    }

    @Test
    fun `longest matching suffix wins over an earlier shorter match`() {
        assertThat(findStreamingRecoveryStart(listOf(1, 2, 3), listOf(2, 3, 9, 1, 2, 3, 4)) { true })
            .isEqualTo(6)
    }

    @Test
    fun `unmatched and single token anchors are rejected`() {
        assertThat(findStreamingRecoveryStart(listOf(1, 2), listOf(3, 4)) { true }).isNull()
        assertThat(findStreamingRecoveryStart(listOf(1, 2), listOf(2, 3)) { true }).isNull()
    }

    @Test
    fun `punctuation alone cannot anchor recovered words`() {
        assertThat(findStreamingRecoveryStart(listOf(1, 2), listOf(1, 2, 3)) { false }).isNull()
    }

    @Test
    fun `initially blank speech can be recovered in full`() {
        assertThat(findStreamingRecoveryStart(emptyList(), listOf(1, 2, 3)) { true }).isZero()
    }

    @Test
    fun `context containing only old speech adds nothing`() {
        val recovered = listOf(2, 3)
        val start = findStreamingRecoveryStart(listOf(1, 2, 3), recovered) { true }
        assertThat(recovered.drop(start!!)).isEmpty()
    }
}

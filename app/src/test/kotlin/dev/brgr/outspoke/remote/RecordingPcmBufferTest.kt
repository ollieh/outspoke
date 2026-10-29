package dev.brgr.outspoke.remote

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class RecordingPcmBufferTest {
    private fun ramp(from: Int, count: Int) = ShortArray(count) { (from + it).toShort() }

    @Test
    fun `keeps every captured sample in order across blocks`() {
        val buffer = RecordingPcmBuffer(maxSamples = 100, blockSamples = 4)
        buffer.onPcm(ramp(0, 3))
        buffer.onPcm(ramp(3, 6))
        buffer.onPcm(ramp(9, 1))

        val recording = buffer.snapshot()

        assertThat(recording.sampleCount).isEqualTo(10)
        assertThat(recording.blocks.map { it.size }).containsExactly(4, 4, 2)
        assertThat(recording.blocks.flatMap { it.toList() }).containsExactlyElementsOf(ramp(0, 10).toList())
        assertThat(buffer.overflowed).isFalse()
    }

    @Test
    fun `accepts audio up to the cap`() {
        val buffer = RecordingPcmBuffer(maxSamples = 8, blockSamples = 4)
        buffer.onPcm(ramp(0, 8))

        assertThat(buffer.overflowed).isFalse()
        assertThat(buffer.snapshot().sampleCount).isEqualTo(8)
    }

    @Test
    fun `releases the audio and stays overflowed past the cap`() {
        val buffer = RecordingPcmBuffer(maxSamples = 8, blockSamples = 4)
        buffer.onPcm(ramp(0, 6))
        buffer.onPcm(ramp(6, 3))
        buffer.onPcm(ramp(9, 1))

        assertThat(buffer.overflowed).isTrue()
        assertThat(buffer.snapshot().sampleCount).isZero()
    }

    @Test
    fun `the default cap holds fifteen minutes of 16 kHz audio`() {
        assertThat(MAX_RECORDING_SAMPLES).isEqualTo(15L * 60 * 16_000)
        assertThat(PcmRecording(listOf(ShortArray(16_000), ShortArray(8_000)), 16_000).durationMs).isEqualTo(1_500)
    }
}

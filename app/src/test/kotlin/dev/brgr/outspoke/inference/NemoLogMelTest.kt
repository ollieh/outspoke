package dev.brgr.outspoke.inference

import org.junit.Assert.*
import org.junit.Test

class NemoLogMelTest {
    @Test
    fun `features match kaldi native fbank reference including reflected edges`() {
        val expected = javaClass.classLoader!!.getResourceAsStream("audio/unified-fbank-golden.txt")!!
            .bufferedReader().useLines { lines ->
                lines.filter { it.matches(Regex("^[0-5] .*")) }
                    .map { line -> line.split(' ').drop(1).map(String::toFloat).toFloatArray() }.toList()
            }
        val frontend = NemoLogMel()
        val live = frontend.accept(ShortArray(1000) { ((it * 7919) % 60001 - 30000).toShort() })
        assertEquals(5, live.size)
        val actual = live + frontend.finish()
        assertEquals(6, actual.size)
        expected.zip(actual).forEach { (reference, frame) -> assertArrayEquals(reference, frame, 0.001f) }
    }

    @Test
    fun `feature frames are invariant to microphone chunk boundaries including final reflection`() {
        val pcm = ShortArray(10_123) { ((it * 7919) % 60001 - 30000).toShort() }
        val whole = NemoLogMel().let { it.accept(pcm) + it.finish() }
        for (chunkSize in listOf(1, 79, 160, 480, 997)) {
            val extractor = NemoLogMel()
            val streamed = pcm.indices.step(chunkSize).flatMap { start ->
                extractor.accept(pcm.copyOfRange(start, minOf(start + chunkSize, pcm.size)))
            } + extractor.finish()
            assertEquals((pcm.size + 80) / 160, streamed.size)
            whole.zip(streamed).forEach { (expected, actual) -> assertArrayEquals(expected, actual, 0f) }
        }
    }

    @Test
    fun `empty and very short recordings finish without invalid reflection or nonfinite features`() {
        for (size in listOf(0, 1, 79, 80, 159, 160, 279, 280)) {
            val extractor = NemoLogMel()
            val frames = extractor.accept(ShortArray(size)) + extractor.finish()
            assertEquals((size + 80) / 160, frames.size)
            assertTrue(frames.all { frame -> frame.size == 128 && frame.all { it.isFinite() } })
        }
    }
}

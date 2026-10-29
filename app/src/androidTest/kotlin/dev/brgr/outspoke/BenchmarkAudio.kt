package dev.brgr.outspoke

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/** Test-only WAV loading shared by the on-device benchmarks. */
internal object BenchmarkAudio {
    /** These fixtures are mono PCM16 WAVs; resample anything that is not 16 kHz. */
    fun readPcm(bytes: ByteArray): ShortArray {
        val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
            String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE")
        require(String(bytes, 12, 4, Charsets.US_ASCII) == "fmt " && header.getInt(16) == 16)
        require(header.getShort(20).toInt() == 1 && header.getShort(22).toInt() == 1 && header.getShort(34).toInt() == 16)
        require(String(bytes, 36, 4, Charsets.US_ASCII) == "data")
        val sampleRate = header.getInt(24)
        val length = header.getInt(40)
        require(length >= 0 && length % 2 == 0 && length <= bytes.size - 44)
        val pcm = ShortArray(length / 2) { header.getShort(44 + it * 2) }
        if (sampleRate == 16_000) return pcm
        val ratio = sampleRate / 16_000.0
        return ShortArray((pcm.size / ratio).roundToInt()) { index ->
            val position = index * ratio
            val left = position.toInt().coerceAtMost(pcm.lastIndex)
            val right = minOf(left + 1, pcm.lastIndex)
            (pcm[left] + (pcm[right] - pcm[left]) * (position - left)).roundToInt().toShort()
        }
    }
}

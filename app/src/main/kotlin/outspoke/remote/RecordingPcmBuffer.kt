package dev.brgr.outspoke.remote

import dev.brgr.outspoke.audio.PcmTap

/** 15 minutes of 16 kHz mono audio, about 29 MB of heap. */
internal const val MAX_RECORDING_SAMPLES = 16_000L * 60 * 15

/** Complete raw audio of one recording, in capture order. */
internal class PcmRecording(val blocks: List<ShortArray>, val sampleRate: Int) {
    val sampleCount: Long = blocks.sumOf { it.size.toLong() }
    val durationMs: Long get() = sampleCount * 1000 / sampleRate
}

/**
 * Keeps one recording's raw microphone audio for the server upload.
 *
 * Audio is stored in one-second blocks so a long recording never needs one large
 * contiguous allocation. Past [maxSamples] the buffer [overflowed] and releases its
 * audio: uploading only the beginning would let a shorter server transcript replace
 * the phone's complete one, so an overlong recording is transcribed on the phone only.
 */
internal class RecordingPcmBuffer(
    private val sampleRate: Int = 16_000,
    private val maxSamples: Long = MAX_RECORDING_SAMPLES,
    private val blockSamples: Int = sampleRate,
) : PcmTap {
    private val blocks = ArrayList<ShortArray>()
    private var lastBlockFill = 0
    private var stored = 0L

    @get:Synchronized
    var overflowed = false
        private set

    @Synchronized
    override fun onPcm(samples: ShortArray) {
        if (overflowed) return
        if (stored + samples.size > maxSamples) {
            overflowed = true
            blocks.clear()
            blocks.trimToSize()
            stored = 0
            return
        }
        var offset = 0
        while (offset < samples.size) {
            if (blocks.isEmpty() || lastBlockFill == blockSamples) {
                blocks.add(ShortArray(blockSamples))
                lastBlockFill = 0
            }
            val count = minOf(blockSamples - lastBlockFill, samples.size - offset)
            samples.copyInto(blocks.last(), lastBlockFill, offset, offset + count)
            lastBlockFill += count
            offset += count
        }
        stored += samples.size
    }

    /** The audio captured so far. Empty after an overflow. */
    @Synchronized
    fun snapshot(): PcmRecording {
        val complete = if (blocks.isEmpty()) emptyList()
        else blocks.dropLast(1) + blocks.last().copyOf(lastBlockFill)
        return PcmRecording(complete, sampleRate)
    }
}

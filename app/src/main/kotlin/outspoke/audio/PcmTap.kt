package dev.brgr.outspoke.audio

/**
 * Receives every block of 16 kHz mono PCM read from the microphone, before VAD gating,
 * so pauses are preserved. Called on the capture thread; [samples] is not reused by
 * the caller and may be retained.
 */
fun interface PcmTap {
    fun onPcm(samples: ShortArray)
}

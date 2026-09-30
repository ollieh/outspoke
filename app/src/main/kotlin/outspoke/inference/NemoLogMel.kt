package dev.brgr.outspoke.inference

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * NeMo's unnormalized 128-bin log-mel frontend, matching sherpa-onnx's
 * kaldi-native-fbank configuration. Normalization belongs to the encoder window,
 * after its missing left/right feature frames have been padded with zeros.
 */
internal class NemoLogMel {
    private var samples = FloatArray(0)
    private var sampleOffset = 0L
    private var sampleCount = 0L
    private var nextFrame = 0L
    private var finished = false
    private val real = DoubleArray(FFT_SIZE)
    private val imaginary = DoubleArray(FFT_SIZE)

    fun accept(pcm: ShortArray): List<FloatArray> {
        check(!finished) { "Feature stream is finished" }
        val joined = samples.copyOf(samples.size + pcm.size)
        for (i in pcm.indices) joined[samples.size + i] = pcm[i] / 32768f
        samples = joined
        sampleCount += pcm.size
        return extract(flush = false)
    }

    fun finish(): List<FloatArray> {
        check(!finished) { "Feature stream is finished" }
        finished = true
        return extract(flush = true).also { samples = FloatArray(0) }
    }

    private fun extract(flush: Boolean): List<FloatArray> {
        val result = ArrayList<FloatArray>()
        val frameCount = (sampleCount + HOP / 2) / HOP
        while (nextFrame < frameCount) {
            val start = nextFrame * HOP + HOP / 2 - WINDOW_SIZE / 2
            if (!flush && start + WINDOW_SIZE > sampleCount) break
            result += frame(start)
            nextFrame++
        }
        val keepFrom = maxOf(0L, nextFrame * HOP + HOP / 2 - WINDOW_SIZE / 2)
            .coerceAtMost(sampleCount)
        val discard = (keepFrom - sampleOffset).toInt()
        if (discard > 0) {
            samples = samples.copyOfRange(discard, samples.size)
            sampleOffset = keepFrom
        }
        return result
    }

    private fun sample(position: Long): Double {
        var index = position
        while (index < 0 || index >= sampleCount) {
            index = if (index < 0) -index - 1 else 2 * sampleCount - 1 - index
        }
        return samples[(index - sampleOffset).toInt()].toDouble()
    }

    private fun frame(start: Long): FloatArray {
        real.fill(0.0)
        imaginary.fill(0.0)
        var previous = sample(start)
        for (i in 0 until WINDOW_SIZE) {
            val current = sample(start + i)
            real[i] = (current - 0.97 * previous) * window[i]
            previous = current
        }
        fft()
        val power = DoubleArray(FFT_SIZE / 2 + 1) { real[it] * real[it] + imaginary[it] * imaginary[it] }
        return FloatArray(FEATURES) { band ->
            val filter = filters[band]
            var energy = 0.0
            for (bin in filter.indices) energy += filter[bin] * power[bin]
            ln(max(energy, 1.1920928955078125e-7)).toFloat()
        }
    }

    private fun fft() {
        var reversed = 0
        for (i in 1 until FFT_SIZE) {
            var bit = FFT_SIZE shr 1
            while (reversed and bit != 0) {
                reversed = reversed xor bit
                bit = bit shr 1
            }
            reversed = reversed xor bit
            if (i < reversed) {
                val swap = real[i]
                real[i] = real[reversed]
                real[reversed] = swap
            }
        }
        var size = 2
        while (size <= FFT_SIZE) {
            val half = size / 2
            for (start in 0 until FFT_SIZE step size) {
                for (j in 0 until half) {
                    val angle = j * FFT_SIZE / size
                    val upper = start + j
                    val lower = upper + half
                    val r = cosines[angle] * real[lower] - sines[angle] * imaginary[lower]
                    val im = cosines[angle] * imaginary[lower] + sines[angle] * real[lower]
                    real[lower] = real[upper] - r
                    imaginary[lower] = imaginary[upper] - im
                    real[upper] += r
                    imaginary[upper] += im
                }
            }
            size *= 2
        }
    }

    companion object {
        const val FEATURES = 128
        private const val HOP = 160
        private const val WINDOW_SIZE = 400
        private const val FFT_SIZE = 512
        private val window = DoubleArray(WINDOW_SIZE) { 0.5 - 0.5 * cos(2 * PI * it / WINDOW_SIZE) }
        private val cosines = DoubleArray(FFT_SIZE / 2) { cos(-2 * PI * it / FFT_SIZE) }
        private val sines = DoubleArray(FFT_SIZE / 2) { sin(-2 * PI * it / FFT_SIZE) }
        private val filters: Array<DoubleArray> = run {
            val logStep = ln(6.4) / 27
            val maxMel = 15 + ln(8.0) / logStep
            val edges = DoubleArray(FEATURES + 2) { index ->
                val mel = index * maxMel / (FEATURES + 1)
                if (mel < 15) mel * (200.0 / 3) else 1000 * exp(logStep * (mel - 15))
            }
            Array(FEATURES) { band ->
                DoubleArray(FFT_SIZE / 2 + 1) { bin ->
                    val frequency = bin * 16000.0 / FFT_SIZE
                    val rising = (frequency - edges[band]) / (edges[band + 1] - edges[band])
                    val falling = (edges[band + 2] - frequency) / (edges[band + 2] - edges[band + 1])
                    max(0.0, min(rising, falling)) * 2 / (edges[band + 2] - edges[band])
                }
            }
        }
    }
}

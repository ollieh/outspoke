package dev.brgr.outspoke.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dev.brgr.outspoke.audio.AudioChunk
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** English RNN-T with the streaming-trained encoder's own left/chunk/right window. */
class ParakeetUnifiedEngine(
    private val intraOpThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
) : StreamingSpeechEngine {
    init {
        require(intraOpThreads in 1..8) { "Expected 1 to 8 inference threads" }
    }
    private data class Models(
        val encoder: OrtSession,
        val decoder: OrtSession,
        val joiner: OrtSession,
        val vocabulary: Array<String>,
        val left: Int,
        val chunk: Int,
        val right: Int,
    ) {
        val total: Int get() = left + chunk + right
        val blank: Int get() = vocabulary.lastIndex
    }

    private val environment = OrtEnvironment.getEnvironment()
    @Volatile private var models: Models? = null
    override val isLoaded: Boolean get() = models != null
    override val currentLanguage: String get() = "en"

    override fun load(modelDir: File) {
        check(!isLoaded) { "Already loaded" }
        val opened = mutableListOf<OrtSession>()
        try {
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(intraOpThreads)
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                fun open(filename: String): OrtSession = environment.createSession(
                    File(modelDir, filename).absolutePath, options,
                ).also(opened::add)
                val encoder = open("encoder.int8.onnx")
                val decoder = open("decoder.int8.onnx")
                val joiner = open("joiner.int8.onnx")
                val metadata = encoder.metadata.customMetadata
                require(metadata["streaming_model_type"] == "nemo_parakeet_unified_streaming") {
                    "This encoder is not a Parakeet Unified streaming export"
                }
                require(metadata["normalize_type"] == "per_feature" && metadata["subsampling_factor"] == "8" &&
                    metadata["feat_dim"] == "128" && metadata["pred_rnn_layers"] == "2" && metadata["pred_hidden"] == "640") {
                    "Unsupported Parakeet Unified feature or decoder configuration"
                }
                val vocabulary = File(modelDir, "tokens.txt").readLines().mapIndexed { index, line ->
                    val separator = line.lastIndexOf(' ')
                    require(separator >= 0 && line.substring(separator + 1).toInt() == index) { "Invalid token vocabulary" }
                    line.substring(0, separator)
                }.toTypedArray()
                require(vocabulary.lastOrNull() == "<blk>" &&
                    vocabulary.size == metadata.getValue("vocab_size").toInt() + 1) { "Invalid blank token" }
                fun frames(name: String): Int = metadata.getValue(name).toInt().also {
                    require(it in 0..4096 && it % 8 == 0) { "Invalid streaming window" }
                }
                val left = frames("left_feature_frames")
                val chunk = frames("chunk_feature_frames")
                val right = frames("right_feature_frames")
                require(chunk > 0 && left + chunk + right <= 4096)
                require(decoder.inputNames.size == 4 && decoder.outputNames.size == 4)
                models = Models(encoder, decoder, joiner, vocabulary, left, chunk, right)
                RecordingTrace.log { "unified loaded threads=$intraOpThreads optimization=ALL_OPT provider=CPU left=$left chunk=$chunk right=$right" }
            }
        } catch (failure: Exception) {
            opened.asReversed().forEach { it.close() }
            throw failure
        }
    }

    override fun newStream(): SpeechStream = UnifiedStream(checkNotNull(models) { "Model is not loaded" })

    override fun transcribe(chunk: AudioChunk): TranscriptResult {
        require(chunk.sampleRate == 16_000)
        return newStream().use { stream ->
            stream.accept(chunk.samples)
            stream.finish()
        }
    }

    override fun close() {
        val loaded = models ?: return
        models = null
        loaded.joiner.close()
        loaded.decoder.close()
        loaded.encoder.close()
    }

    private inner class UnifiedStream(private val loaded: Models) : SpeechStream {
        private val frontend = NemoLogMel()
        private val frames = ArrayDeque<FloatArray>()
        private var firstFrame = 0
        private var processed = 0
        private var ready = 0
        private var ended = false
        private val tokens = mutableListOf<Int>()
        private var logProbability = 0.0
        private var state1 = FloatArray(1280)
        private var state2 = FloatArray(1280)
        private var prediction: FloatArray? = null
        private val decoderInputs = loaded.decoder.inputNames.toList()
        private val traceStream = RecordingTrace.nextSessionId()

        private fun checkActive() {
            check(!ended && models === loaded) { "Speech stream is no longer active" }
        }

        override fun accept(samples: ShortArray): TranscriptResult.Partial? {
            checkActive()
            val previousSize = tokens.size
            addFrames(frontend.accept(samples))
            return if (tokens.size != previousSize) TranscriptResult.Partial(text(), confidence()) else null
        }

        private fun addFrames(incoming: List<FloatArray>) {
            for (frame in incoming) {
                frames.addLast(frame)
                ready++
                if (processed + loaded.chunk + loaded.right <= ready) decodeWindow(loaded.chunk)
            }
        }

        override fun finish(): TranscriptResult.Final {
            checkActive()
            val start = System.nanoTime()
            RecordingTrace.log { "unified finish begin stream=$traceStream pendingFrames=${ready - processed}" }
            addFrames(frontend.finish())
            while (processed < ready) decodeWindow(minOf(loaded.chunk, ready - processed))
            ended = true
            frames.clear()
            RecordingTrace.log { "unified finish end stream=$traceStream durationMs=${(System.nanoTime() - start) / 1_000_000}" }
            return TranscriptResult.Final(text(), confidence = confidence())
        }

        override fun close() {
            ended = true
            frames.clear()
            tokens.clear()
            prediction = null
        }

        private fun text(): String = decodeParakeetTokens(tokens, loaded.vocabulary, loaded.blank)
        private fun confidence(): Float = if (tokens.isEmpty()) 1f else exp(logProbability / tokens.size).toFloat()

        private fun decodeWindow(validFrames: Int) {
            val windowStartTime = System.nanoTime()
            val values = FloatArray(NemoLogMel.FEATURES * loaded.total)
            val windowStart = processed - loaded.left
            for (time in 0 until loaded.total) {
                val absolute = windowStart + time
                if (absolute < firstFrame || absolute >= ready) continue
                val frame = frames[absolute - firstFrame]
                for (feature in frame.indices) values[feature * loaded.total + time] = frame[feature]
            }
            // Zero padding is part of normalization. Normalizing real speech first
            // produces a different input at startup and when Stop flushes the tail.
            for (feature in 0 until NemoLogMel.FEATURES) {
                val offset = feature * loaded.total
                var mean = 0.0
                for (t in 0 until loaded.total) mean += values[offset + t]
                mean /= loaded.total
                var variance = 0.0
                for (t in 0 until loaded.total) {
                    val centered = values[offset + t] - mean
                    variance += centered * centered
                }
                val denominator = sqrt(variance / loaded.total) + 1e-5
                for (t in 0 until loaded.total) values[offset + t] = ((values[offset + t] - mean) / denominator).toFloat()
            }
            tensor(values, 1, NemoLogMel.FEATURES.toLong(), loaded.total.toLong()).use { signal ->
                OnnxTensor.createTensor(environment, LongBuffer.wrap(longArrayOf(loaded.total.toLong())), longArrayOf(1)).use { length ->
                    val encoderStart = System.nanoTime()
                    loaded.encoder.run(mapOf("audio_signal" to signal, "length" to length)).use { output ->
                        RecordingTrace.log { "unified encoder stream=$traceStream processed=$processed centerFrames=$validFrames pendingFrames=${ready - processed} durationMs=${(System.nanoTime() - encoderStart) / 1_000_000}" }
                        val encoded = output[0] as OnnxTensor
                        val shape = encoded.info.shape
                        require(shape[0] == 1L && shape[1] == 1024L) { "Unexpected encoder output shape" }
                        val timeSteps = shape[2].toInt()
                        val centerStart = loaded.left / 8
                        val centerEnd = centerStart + ceil(validFrames / 8.0).toInt()
                        require(centerEnd <= timeSteps) { "Encoder omitted center frames" }
                        val data = floats(encoded)
                        if (prediction == null) predict(loaded.blank)
                        for (time in centerStart until centerEnd) {
                            val frame = FloatArray(1024) { data[it * timeSteps + time] }
                            decodeFrame(frame)
                        }
                    }
                }
            }
            processed += validFrames
            RecordingTrace.log { "unified window end stream=$traceStream processed=$processed pendingFrames=${ready - processed} durationMs=${(System.nanoTime() - windowStartTime) / 1_000_000}" }
            val retainFrom = maxOf(0, processed - loaded.left)
            while (firstFrame < retainFrom) {
                frames.removeFirst()
                firstFrame++
            }
        }

        private fun predict(token: Int) {
            OnnxTensor.createTensor(environment, IntBuffer.wrap(intArrayOf(token)), longArrayOf(1, 1)).use { target ->
                OnnxTensor.createTensor(environment, IntBuffer.wrap(intArrayOf(1)), longArrayOf(1)).use { length ->
                    tensor(state1, 2, 1, 640).use { first ->
                        tensor(state2, 2, 1, 640).use { second ->
                            loaded.decoder.run(mapOf(
                                decoderInputs[0] to target, decoderInputs[1] to length,
                                decoderInputs[2] to first, decoderInputs[3] to second,
                            )).use { output ->
                                prediction = floats(output[0] as OnnxTensor)
                                state1 = floats(output[2] as OnnxTensor)
                                state2 = floats(output[3] as OnnxTensor)
                            }
                        }
                    }
                }
            }
        }

        private fun decodeFrame(frame: FloatArray) {
            tensor(frame, 1, 1024, 1).use { encoderFrame ->
                repeat(10) {
                    val logits = tensor(checkNotNull(prediction), 1, 640, 1).use { decoderFrame ->
                        loaded.joiner.run(mapOf("encoder_outputs" to encoderFrame, "decoder_outputs" to decoderFrame)).use { output ->
                            floats(output[0] as OnnxTensor)
                        }
                    }
                    require(logits.size == loaded.vocabulary.size && logits.all { it.isFinite() }) { "Invalid recognition scores" }
                    val token = logits.indices.maxBy { logits[it] }
                    if (token == loaded.blank) return
                    val maximum = logits[token].toDouble()
                    logProbability -= ln(logits.sumOf { exp(it - maximum) })
                    tokens += token
                    // The predictor advances only on a nonblank symbol. Its cached
                    // output and LSTM state survive every encoder window unchanged.
                    predict(token)
                }
            }
        }
    }

    private fun tensor(values: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), shape)

    private fun floats(tensor: OnnxTensor): FloatArray =
        tensor.floatBuffer.let { buffer -> FloatArray(buffer.remaining()).also(buffer::get) }
}

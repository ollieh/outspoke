package dev.brgr.outspoke.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Debug
import android.util.Log
import dev.brgr.outspoke.audio.AudioChunk
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

private const val TAG = "ParakeetEngine"
private const val FALLBACK_BLANK_ID = 1024
/** Two seconds at the encoder's 12.5 frames/s. */
private const val PREDICTOR_RESET_BLANK_FRAMES = 25
private const val BLANK_CHUNK_RETRY_CONFIDENCE = 0.55f

/**
 * Output of [ParakeetEngine.greedyDecode]: the decoded text plus a per-token geometric-mean
 * confidence in [0.0, 1.0]. The confidence is exp(mean(log_softmax(argmax))) over all
 * non-blank emissions — a principled score that low-values hallucinations (flat / uncertain
 * token distributions on cold strides or noise) and high-values clean speech.
 */
private data class DecodeResult(val text: String, val confidence: Float)

/**
 * The TDT decoder's carry-over state between streaming chunks.
 *
 * The Parakeet TDT decoder is a temporal-difference transducer: each encoder frame is
 * decoded against the previous token and an LSTM state. When audio is processed in
 * chunks (the streaming path), this state is carried from the end of one chunk to the
 * start of the next so the decoder continues exactly where it left off — no re-emission,
 * no duplication. This is the same mechanism NeMo's `decoding_computer` uses with
 * `prev_batched_state`.
 *
 *  - [lstmState1] / [lstmState2]: the decoder LSTM hidden / cell, each `[2, 1, 640]`
 *    (1280 floats) as produced by `decoder_joint`'s `output_states_1/2`.
 *  - [prevToken]: the last emitted token ID (the predictor embedding input for the next
 *    frame). The blank token for a fresh start (see [ParakeetEngine.initialTdtState]).
 *  - [frameDelta]: how far the decoder's frame position overshot the range end when the
 *    previous chunk's decode loop exited. The TDT advance rule skips `duration` frames
 *    *without running the joint on them*; when a duration jump at the last frame of a
 *    chunk lands past the chunk boundary, the skipped frames must NOT be re-decoded in
 *    the next chunk (the one-shot decoder never visits them — re-running the joint there
 *    can emit phantom tokens and shifts the LSTM trajectory). The next chunk therefore
 *    starts `frameDelta` frames after its nominal start. A negative [frameDelta] means
 *    the loop terminated early (safety cap) and the next chunk must resume a few frames
 *    before its nominal start so no frame is skipped.
 */
data class TdtState(
    val lstmState1: FloatArray,
    val lstmState2: FloatArray,
    val prevToken: Int,
    val frameDelta: Int = 0,
    /** Encoder frames advanced without a token, carried across streaming chunks. */
    val blankFrames: Int = 0,
)

/**
 * A single token candidate at one [TokenEmission]: the token ID and its log-softmax
 * probability over the token portion `[0..blankId]`. The list of top candidates at an
 * emission is the acoustic evidence the greedy decode previously discarded: the
 * runner-up tokens are the model's own alternatives for that frame.
 */
data class EmissionToken(val token: Int, val logProb: Double)

/**
 * One non-blank token emission from the TDT decode loop, carrying the evidence needed
 * to build acoustic word alternatives:
 *  - [token]     the emitted (argmax) token ID
 *  - [frame]     the encoder frame index at which the token was emitted
 *  - [logProb]   log-softmax of [token] over `[0..blankId]`
 *  - [topTokens] the top tokens at this emission (including [token]) with their
 *                log-softmax probabilities — the acoustic runner-ups
 */
data class TokenEmission(
    val token: Int,
    val frame: Int,
    val logProb: Double,
    val topTokens: List<EmissionToken>,
)

/**
 * The TDT decoder's state just before the joint model call at [frame]: the LSTM
 * hidden / cell states and the previous token. A snapshot taken at every emitting
 * joint call so a local word beam can re-start the joint model from the exact state
 * the greedy decode had when it emitted a word's first token.
 */
data class FrameState(
    val frame: Int,
    val lstmState1: FloatArray,
    val lstmState2: FloatArray,
    val prevToken: Int,
)

/**
 * Result of decoding a frame range: the emitted tokens, the updated state, confidence,
 * per-emission acoustic evidence ([emissions] / [stateSnapshots]) for word-alternative
 * capture, and the utterance-level confidence accumulators.
 */
private data class DecodeRangeResult(
    val tokens: List<Int>,
    val state: TdtState,
    val confidence: Float,
    /** Sum of per-emission log-softmax probabilities (non-blank tokens only). */
    val logProbSum: Double,
    /** Number of non-blank emissions in the range. */
    val emissionCount: Int,
    /** Per-emission evidence: token, frame, log-prob, and top-K acoustic runner-ups. */
    val emissions: List<TokenEmission>,
    /** Decoder state snapshots at each emitting joint call (for local word beams). */
    val stateSnapshots: List<FrameState>,
)

private object Names {
    // nemo128.onnx  ← verified: expected [waveforms, waveforms_lens]
    const val PREP_IN_AUDIO = "waveforms"
    const val PREP_IN_LENGTH = "waveforms_lens"
    // Outputs accessed by index: [0] = features, [1] = feature lengths

    // encoder-model.int8.onnx  ← verified
    const val ENC_IN_SIGNAL = "audio_signal"   // FLOAT [-1, 128, -1]
    const val ENC_IN_LENGTH = "length"          // INT64 [-1]
    const val ENC_OUT_SIGNAL = "outputs"          // FLOAT [-1, 1024, -1]  (B, D, T)
    const val ENC_OUT_LEN = "encoded_lengths"  // INT64 [-1]

    // decoder_joint-model.int8.onnx  ← verified
    const val DEC_IN_ENC_OUT = "encoder_outputs"  // FLOAT [-1, 1024, -1]
    const val DEC_IN_TARGETS = "targets"           // INT32 [-1, -1]
    const val DEC_IN_TARGET_LEN = "target_length"     // INT32 [-1]
    const val DEC_IN_STATES_1 = "input_states_1"    // FLOAT [2, -1, 640]
    const val DEC_IN_STATES_2 = "input_states_2"    // FLOAT [2, -1, 640]
    // Outputs accessed by index:
    //   0 → "outputs"         FLOAT [-1,-1,-1, 8198]  (B, T_enc, T_tgt, vocab+dur)
    //   1 → "prednet_lengths" INT32 [-1]
    //   2 → "output_states_1" FLOAT [2,-1, 640]
    //   3 → "output_states_2" FLOAT [2,-1, 640]
}

/**
 * The stateful decode primitives the [InferenceRepository] chunked-TDT streaming path
 * requires. Implemented by [ParakeetEngine]; extracted as a capability interface so the
 * repository's streaming state machine (buffer management, chunk cadence, TDT state
 * carry-over, flush) can be driven by a deterministic fake in JVM integration tests
 * (see `FakeStreamingParakeet`) without ONNX inference.
 */
interface ChunkStreamingEngine {

    /** A fresh TDT decoder state for a new utterance (zeroed LSTM + blank prev token). */
    fun initialTdtState(): TdtState

    /** Preprocess + encode [samples] (float32 in [-1, 1], 16 kHz); caller closes [OnnxTensor]. */
    fun encodeBuffer(samples: FloatArray): Pair<OnnxTensor, Int>

    /** Decode encoder frames `[frameStart, frameEnd)` of [encoderOut] from the carried [state]. */
    fun decodeChunk(
        encoderOut: OnnxTensor,
        totalLength: Int,
        frameStart: Int,
        frameEnd: Int,
        state: TdtState,
    ): ChunkDecodeResult

    /** Detokenise [tokens] to a string (SentencePiece-aware). */
    fun detokenizeTokens(tokens: List<Int>): String

    /** `true` when [tokenId] is a SentencePiece word-initial token (starts a new word). */
    fun tokenStartsWord(tokenId: Int): Boolean

    /** Bounded local word beam over `[startFrame, endFrame]` from [initialState]. */
    fun localWordBeam(
        encoderOut: OnnxTensor,
        totalLength: Int,
        startFrame: Int,
        endFrame: Int,
        initialState: FrameState,
        beamWidth: Int = WordBeamTuning.BEAM_WIDTH,
        topK: Int = WordBeamTuning.TOP_K_TOKENS,
        maxSteps: Int = WordBeamTuning.MAX_BEAM_STEPS,
        maxAlternatives: Int = WordBeamTuning.MAX_ALTERNATIVES,
    ): List<WordAlternative>
}

/**
 * Tuning knobs for the word-alternative capture (top-K token swaps + local word
 * beam). Defaults for [ChunkStreamingEngine.localWordBeam]; shared by
 * [ParakeetEngine] and its test fakes.
 */
object WordBeamTuning {
    const val TOP_K_TOKENS = 3        // runner-up tokens captured per emission
    const val BEAM_WIDTH = 8          // live states in the local word beam
    const val MAX_BEAM_STEPS = 200    // joint-call cap per word beam
    const val MAX_ALTERNATIVES = 5    // word hypotheses returned per beam
}

/**
 * Result of a streaming [ChunkStreamingEngine.decodeChunk]: the chunk's token IDs, the
 * updated [TdtState], the chunk's confidence accumulators, and per-emission acoustic
 * evidence.
 *
 * [logProbSum] is the sum of per-emission log-softmax probabilities (non-blank tokens
 * only) and [emissionCount] the count of those emissions. The caller accumulates both
 * across chunks and computes the utterance confidence as
 * `exp(totalLogProbSum / totalEmissions)` — the geometric mean of every token's
 * probability. Summing log-probs (rather than averaging per-chunk confidences) is what
 * makes the combined score independent of how the audio is chunked.
 *
 * [emissions] carries, per non-blank emission, the token, its frame index, its
 * log-prob, and the top-K acoustic runner-up tokens — the evidence the word-alternative
 * capture (top-K token swaps + local word beam) needs. [stateSnapshots] carries the
 * decoder state at each emitting joint call so [ChunkStreamingEngine.localWordBeam] can
 * re-start from the exact state the greedy decode had at a word's first token.
 */
data class ChunkDecodeResult(
    val tokens: List<Int>,
    val state: TdtState,
    val logProbSum: Double,
    val emissionCount: Int,
    val emissions: List<TokenEmission>,
    val stateSnapshots: List<FrameState>,
)

/**
 * Wraps the three Parakeet-V3 ONNX sessions.
 *
 * Pipeline (all tensor names verified from device logcat):
 *  1. Normalise PCM  →  float32 in [-1, 1]
 *  2. nemo128.onnx   →  log-mel features  [1, 128, T′]
 *  3. encoder        →  encoded features  [1, 1024, T_enc]   ← (B, D, T) format!
 *  4. greedy TDT     →  token IDs via decoder_joint with LSTM state carry-over
 *  5. detokenise     →  string via vocab.txt
 */
class ParakeetEngine(
    private val intraOpThreads: Int = 2,
    private val optimizationLevel: OrtSession.SessionOptions.OptLevel = OrtSession.SessionOptions.OptLevel.BASIC_OPT,
) : SpeechEngine, ChunkStreamingEngine {

    private var env: OrtEnvironment? = null
    private var prepSession: OrtSession? = null
    private var encSession: OrtSession? = null
    private var decSession: OrtSession? = null

    private var vocabulary: Array<String> = emptyArray()
    private var blankId: Int = FALLBACK_BLANK_ID
    private var numDurations: Int = 0   // derived at load: outputDim - (blankId + 1)


    @Volatile
    override var isLoaded: Boolean = false
        private set

    /**
     * BCP-47 language tag to use for post-processing (filler removal, number normalisation).
     * `null` means auto-detect; the post-processing pipeline defaults to `"en"` in that case.
     *
     * **ONNX language conditioning note:**
     * The current Parakeet TDT 0.6B-v3 ONNX export (nemo128 + encoder + decoder_joint) does
     * NOT expose a language input tensor — the three verified input sets are:
     *   • nemo128:         [waveforms (FLOAT), waveforms_lens (INT64)]
     *   • encoder:         [audio_signal (FLOAT), length (INT64)]
     *   • decoder_joint:   [encoder_outputs, targets, target_length, input_states_1, input_states_2]
     * None of these carry a language token.  When a language-conditioned ONNX export becomes
     * available, the correct place to inject it is `encode()` — pass the language ID as an
     * additional INT64 tensor whose name matches the new export's input slot (e.g.
     * "language_id"), mapping each BCP-47 tag to the model's integer language index.
     * Until then, [forcedLanguage] is consumed only by the post-processing layer.
     */
    @Volatile
    private var forcedLanguage: String? = null

    /** Returns the active language tag for post-processing, defaulting to `"en"` for auto-detect. */
    override val currentLanguage: String get() = forcedLanguage ?: "en"

    /** Implements [SpeechEngine.setLanguage]; stores [tag] for post-processing use. */
    override fun setLanguage(tag: String) {
        forcedLanguage = if (tag == "auto") null else tag
    }


    /**
     * Creates all ONNX sessions and loads the vocabulary.
     * **Must be called on a background thread** - loading takes 1-3 s on first run.
     *
     * @throws IllegalStateException if called while already loaded.
     * @throws Exception (OrtException / IOException) if any model file is corrupt.
     */
    override fun load(modelDir: File) {
        val startTime = System.currentTimeMillis()
        val modelSizeMB = modelDir.walkTopDown().filter { it.isFile }.sumOf { it.length() } / (1024 * 1024)
        Log.i(TAG, "ParakeetEngine loading from ${modelDir.path}, size=${modelSizeMB}MB")
        if (modelSizeMB > 500) Log.w(TAG, "Parakeet model is very large (${modelSizeMB}MB) - may require high RAM")

        check(!isLoaded) { "Already loaded; call close() before reloading" }

        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(intraOpThreads)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(optimizationLevel)
        }

        env = OrtEnvironment.getEnvironment()
        val e = env!!

        val prepFile = File(modelDir, "nemo128.onnx")
        if (prepFile.exists()) {
            prepSession = e.createSession(prepFile.absolutePath, opts)
            logSession("nemo128 (preprocessor)", prepSession!!)
        } else {
            Log.w(TAG, "nemo128.onnx absent - will forward raw audio to encoder (shapes will mismatch)")
        }

        encSession = e.createSession(File(modelDir, "encoder-model.int8.onnx").absolutePath, opts)
        logSession("encoder", encSession!!)

        decSession = e.createSession(
            File(modelDir, "decoder_joint-model.int8.onnx").absolutePath, opts
        )
        logSession("decoder_joint", decSession!!)

        // vocab.txt format: "<token_text> <id>" (e.g. "▁like 2656").
        // Lines are in ID order so line N = token ID N - we just need the first field.
        vocabulary = File(modelDir, "vocab.txt").readLines()
            .map { line -> line.trim().split(Regex("\\s+")).firstOrNull().orEmpty() }
            .toTypedArray()
        Log.d(TAG, "Vocabulary: ${vocabulary.size} tokens")

        // Resolution order:
        //  1. Scan vocab.txt for a known blank label - most reliable for NeMo TDT exports
        //  2. config.json - may be stale or wrong in third-party ONNX conversions
        //  3. vocabulary.size - 1 - safe heuristic (blank is always the last NeMo token)
        blankId = scanVocabForBlankId()
            ?: parseBlankId(File(modelDir, "config.json"))
                    ?: run {
                val fallback = (vocabulary.size - 1).coerceAtLeast(0)
                Log.w(TAG, "blank_id not found in vocab or config.json - using vocabulary.size-1=$fallback")
                fallback
            }
        val blankLabel = vocabulary.getOrNull(blankId) ?: "<out-of-range>"
        Log.d(TAG, "Blank id: $blankId  ('$blankLabel')")

        // Derive numDurations from the decoder's first output dimension
        val decOutNames = decSession!!.outputNames.toList()
        val jointOutInfo = decSession!!.outputInfo[decOutNames[0]]
        val jointDim = (jointOutInfo?.info as? ai.onnxruntime.TensorInfo)?.shape?.last()?.toInt() ?: 0
        numDurations = if (jointDim > blankId + 1) jointDim - (blankId + 1) else 0
        Log.d(TAG, "Joint output dim: $jointDim  numDurations: $numDurations")

        opts.close()
        isLoaded = true
        Log.d(TAG, "ParakeetEngine ready (modelDir=${modelDir.path})")
        val elapsed = System.currentTimeMillis() - startTime
        Log.i(TAG, "ParakeetEngine loaded in ${elapsed}ms")
        logMemoryUsage()
    }


    /**
     * Transcribes [chunk] synchronously and returns a [TranscriptResult].
     *
     * - Short chunks (< ~200 ms) will typically return [TranscriptResult.Partial] with an
     *   empty string because the encoder does not have enough context. Accumulate chunks in
     *   [InferenceRepository] before calling this for production use.
     * - This is a **blocking** call - always dispatch to [kotlinx.coroutines.Dispatchers.Default].
     */
    override fun transcribe(chunk: AudioChunk): TranscriptResult {
        if (!isLoaded) return TranscriptResult.Failure(IllegalStateException("Engine not loaded"))
        return try {
            val e = env!!
            val enc = encSession!!
            val dec = decSession!!

            // 1. Normalise
            val samples = normalizePcm(chunk.samples)

            // 2. Preprocess → mel features
            val (feats, featLen) = preprocess(e, samples)

            // 3. Encode
            val (encOut, encLen) = encode(e, enc, feats, featLen)
            feats.close()

            // 4. Greedy TDT decode (full range, fresh state)
            val decode = decodeRange(e, dec, encOut, encLen, 0, encLen, initialTdtState())
            encOut.close()

            val text = detokenize(decode.tokens)
            if (text.isBlank()) TranscriptResult.Partial("") else TranscriptResult.Final(text, confidence = decode.confidence)
        } catch (ex: Exception) {
            Log.e(TAG, "transcribe() failed", ex)
            TranscriptResult.Failure(ex)
        }
    }

    companion object {
        /**
         * Tuning knobs for the word-alternative capture (top-K token swaps + local word
         * beam). Defaults for [localWordBeam]; aliases of [WordBeamTuning] so the
         * interface keeps a single source of truth. All are exposed as parameters on the
         * public functions so callers (the repository, tests) can override them without
         * structural change.
         */
        const val TOP_K_TOKENS = WordBeamTuning.TOP_K_TOKENS
        const val BEAM_WIDTH = WordBeamTuning.BEAM_WIDTH
        const val MAX_BEAM_STEPS = WordBeamTuning.MAX_BEAM_STEPS
        const val MAX_ALTERNATIVES = WordBeamTuning.MAX_ALTERNATIVES
    }

    /**
     * A fresh TDT decoder state: zeroed LSTM hidden/cell and the blank token as the initial
     * predictor target. The TDT predictor's first input must be the blank token (not SOS) -
     * this matches transcribe-rs, and feeding SOS (token 0) makes short low-margin words
     * collapse to empty output. Use this to start a new streaming session (the first chunk
     * of an utterance).
     */
    override fun initialTdtState(): TdtState = TdtState(FloatArray(2 * 640), FloatArray(2 * 640), blankId)

    /**
     * Preprocesses + encodes [samples] (float32 in [-1, 1], 16 kHz) and returns the encoder
     * output tensor plus its total frame count.
     *
     * **The caller MUST close the returned [OnnxTensor]** when done (it is a native
     * resource). This is the building block for the streaming path: the caller assembles a
     * `[left context | chunk | right context]` buffer, encodes it, then decodes only the
     * chunk's frame range via [decodeChunk] with a carried [TdtState].
     *
     * @throws IllegalStateException if the engine is not loaded.
     */
    override fun encodeBuffer(samples: FloatArray): Pair<OnnxTensor, Int> {
        check(isLoaded) { "Engine not loaded; call load() first" }
        val e = env!!
        val (feats, featLen) = preprocess(e, samples)
        val (encOut, encLen) = encode(e, encSession!!, feats, featLen)
        feats.close()
        return encOut to encLen
    }


    /**
     * Decodes encoder frames `[frameStart, frameEnd)` of [encoderOut] starting from the
     * carried [state], returning the chunk's token IDs, the updated [TdtState], and the
     * chunk's confidence accumulators (see [ChunkDecodeResult]).
     *
     * This is the streaming primitive: because the TDT decoder's LSTM state and previous
     * token are carried across chunks, each chunk emits exactly its own new tokens — no
     * re-emission of earlier content, no duplication. [totalLength] is the encoder output's
     * full frame count (used for frame extraction); [frameStart]/[frameEnd] select the chunk
     * sub-range (the left/right context frames are encoded but not decoded).
     *
     * @throws IllegalStateException if the engine is not loaded.
     */
    override fun decodeChunk(
        encoderOut: OnnxTensor,
        totalLength: Int,
        frameStart: Int,
        frameEnd: Int,
        state: TdtState,
    ): ChunkDecodeResult {
        check(isLoaded) { "Engine not loaded; call load() first" }
        var result = decodeRange(env!!, decSession!!, encoderOut, totalLength, frameStart, frameEnd, state)
        // A previous sentence's predictor can suppress every word in a chunk before the
        // blank-frame timeout resets it. Retry that audio now, while its opening
        // frames are still available, rather than losing it at the next stride.
        // A trailing punctuation token alone does not mean speech was recovered.
        if (detokenize(result.tokens).none { it.isLetterOrDigit() } && state.prevToken != blankId) {
            val retry = decodeRange(
                env!!, decSession!!, encoderOut, totalLength, frameStart, frameEnd, initialTdtState(),
            )
            if (retry.confidence >= BLANK_CHUNK_RETRY_CONFIDENCE &&
                detokenize(retry.tokens).any { it.isLetterOrDigit() }
            ) {
                RecordingTrace.log { "blank chunk recovered tokens=${retry.tokens.size} confidence=${retry.confidence}" }
                // The original predictor may finish the previous sentence before it
                // stalls. Preserve that endpoint when it precedes recovered speech.
                val keepEndpoint = detokenize(result.tokens).trim() in setOf(".", "!", "?") &&
                    result.emissions.isNotEmpty() && retry.emissions.isNotEmpty() &&
                    result.emissions.last().frame < retry.emissions.first().frame &&
                    detokenize(retry.tokens).trimStart().firstOrNull()?.isLetterOrDigit() == true
                result = if (keepEndpoint) retry.copy(
                    tokens = result.tokens + retry.tokens,
                    logProbSum = result.logProbSum + retry.logProbSum,
                    emissionCount = result.emissionCount + retry.emissionCount,
                    emissions = result.emissions + retry.emissions,
                    stateSnapshots = result.stateSnapshots + retry.stateSnapshots,
                ) else retry
            }
        }
        return ChunkDecodeResult(
            result.tokens, result.state, result.logProbSum, result.emissionCount,
            result.emissions, result.stateSnapshots,
        )
    }

    /**
     * Detokenises a list of token IDs to a display string (SentencePiece-aware). Exposed for
     * the streaming path, which accumulates tokens across chunks and detokenises the running
     * total for each partial.
     */
    override fun detokenizeTokens(tokens: List<Int>): String = detokenize(tokens)

    /**
     * Returns `true` when [tokenId] is a SentencePiece word-initial token (its vocabulary
     * entry starts with the `▁` word-boundary marker). Used by the repository to segment
     * per-chunk emissions into words for acoustic-alternative capture.
     */
    override fun tokenStartsWord(tokenId: Int): Boolean {
        if (tokenId < 0 || tokenId >= vocabulary.size || tokenId == blankId) return false
        return vocabulary[tokenId].startsWith("▁")
    }

    /**
     * Bounded local word beam: re-runs the joint model over `[startFrame, endFrame]`
     * starting from [initialState] (the decoder state the greedy decode had at the word's
     * first token) and returns the top [maxAlternatives] detokenised word hypotheses with
     * their length-normalised acoustic log-probs.
     *
     * This is the quality step of the word-alternative capture: a word is a *sequence* of
     * SentencePiece tokens, and a real alternative word is often a *different token
     * sequence* (e.g. a 2-token word vs the emitted 1-token word) that single-token
     * swapping cannot reach. The beam branches on the top [topK] tokens at each step
     * (plus the blank, which ends the word), carrying the LSTM state per beam.
     *
     * Bounded work: at most [beamWidth] live states, at most [maxSteps] joint calls total,
     * and frames advance monotonically, so a pathological word cannot blow up decode time.
     * The caller gates this on per-word confidence (only low-confidence words are beamed)
     * and caps the number of beams per chunk.
     *
     * @throws IllegalStateException if the engine is not loaded.
     */
    override fun localWordBeam(
        encoderOut: OnnxTensor,
        totalLength: Int,
        startFrame: Int,
        endFrame: Int,
        initialState: FrameState,
        beamWidth: Int,
        topK: Int,
        maxSteps: Int,
        maxAlternatives: Int,
    ): List<WordAlternative> {
        check(isLoaded) { "Engine not loaded; call load() first" }
        val e = env!!
        val session = decSession!!

        val encShape = encoderOut.info.shape
        val encDim = encShape[1].toInt()
        val encData = FloatArray(encoderOut.floatBuffer.remaining())
        encoderOut.floatBuffer.rewind()
        encoderOut.floatBuffer.get(encData)

        val stateShape = longArrayOf(2L, 1L, 640L)
        val decOutputNames = session.outputNames.toList()

        // One beam state: the decoder position plus the partial token sequence built so far.
        class BeamState(
            val frame: Int,
            val prevToken: Int,
            val lstm1: FloatArray,
            val lstm2: FloatArray,
            val tokens: IntArray,
            val logProb: Double,
        )

        val start = BeamState(startFrame, initialState.prevToken,
            initialState.lstmState1, initialState.lstmState2, IntArray(0), 0.0)
        var live = listOf(start)
        val finished = HashMap<String, Double>()   // detokenised word → best length-normalised log-prob
        var steps = 0

        // A joint call for one beam state at one frame: returns the top tokens + log-probs,
        // the predicted duration, and the updated LSTM states.
        class JointStep(
            val topTokens: List<EmissionToken>,
            val duration: Int,
            val lstm1: FloatArray,
            val lstm2: FloatArray,
        )

        fun jointCall(frame: Int, prevToken: Int, lstm1: FloatArray, lstm2: FloatArray): JointStep {
            val frameData = FloatArray(encDim) { d -> encData[d * totalLength + frame] }
            val frameTensor = OnnxTensor.createTensor(e, FloatBuffer.wrap(frameData), longArrayOf(1L, encDim.toLong(), 1L))
            val targetTensor = OnnxTensor.createTensor(e, IntBuffer.wrap(intArrayOf(prevToken)), longArrayOf(1L, 1L))
            val targetLenTensor = OnnxTensor.createTensor(e, IntBuffer.wrap(intArrayOf(1)), longArrayOf(1L))
            val statesTensor1 = OnnxTensor.createTensor(e, FloatBuffer.wrap(lstm1), stateShape)
            val statesTensor2 = OnnxTensor.createTensor(e, FloatBuffer.wrap(lstm2), stateShape)
            val inputs = mapOf(
                Names.DEC_IN_ENC_OUT to frameTensor,
                Names.DEC_IN_TARGETS to targetTensor,
                Names.DEC_IN_TARGET_LEN to targetLenTensor,
                Names.DEC_IN_STATES_1 to statesTensor1,
                Names.DEC_IN_STATES_2 to statesTensor2,
            )
            try {
                session.run(inputs).use { result ->
                    val logitsTensor = result.get(decOutputNames[0]).get() as OnnxTensor
                    val logits = FloatArray(logitsTensor.floatBuffer.remaining())
                    logitsTensor.floatBuffer.get(logits)

                    // Single pass: log-softmax denominator + top-K tokens by logit.
                    val maxLogit = (0..blankId).maxOf { logits[it] }
                    var expSum = 0.0
                    val topIdx = IntArray(topK)
                    val topVal = DoubleArray(topK) { Double.NEGATIVE_INFINITY }
                    for (k in 0..blankId) {
                        val v = logits[k].toDouble()
                        expSum += Math.exp(v - maxLogit)
                        // Insert into the descending top-K (topVal[0] = best):
                        // find the slot from the bottom, shifting smaller values down.
                        if (v > topVal[topK - 1]) {
                            var j = topK - 1
                            while (j > 0 && v > topVal[j - 1]) {
                                topVal[j] = topVal[j - 1]
                                topIdx[j] = topIdx[j - 1]
                                j--
                            }
                            topVal[j] = v
                            topIdx[j] = k
                        }
                    }
                    val logExpSum = Math.log(expSum)
                    val top = (0 until topK)
                        .map { i -> EmissionToken(topIdx[i], topVal[i] - maxLogit - logExpSum) }

                    // Duration: argmax over the last numDurations logits.
                    val nDur = numDurations
                    val duration = if (nDur > 0) {
                        val durBase = blankId + 1
                        (0 until nDur.coerceAtLeast(1)).maxByOrNull { logits[durBase + it] } ?: 0
                    } else 0

                    val s1 = result.get(decOutputNames[2]).get() as OnnxTensor
                    val lstm1Out = FloatArray(s1.floatBuffer.remaining()).also { s1.floatBuffer.get(it) }
                    val s2 = result.get(decOutputNames[3]).get() as OnnxTensor
                    val lstm2Out = FloatArray(s2.floatBuffer.remaining()).also { s2.floatBuffer.get(it) }

                    return JointStep(top, duration, lstm1Out, lstm2Out)
                }
            } finally {
                frameTensor.close()
                targetTensor.close()
                targetLenTensor.close()
                statesTensor1.close()
                statesTensor2.close()
            }
        }

        while (live.isNotEmpty() && steps < maxSteps) {
            val next = ArrayList<BeamState>(beamWidth * 2)
            for (bs in live) {
                if (bs.frame > endFrame) continue   // ran past the word's frame range
                if (++steps >= maxSteps) break
                val step = jointCall(bs.frame, bs.prevToken, bs.lstm1, bs.lstm2)

                // Blank branch: the word ends here.
                if (bs.tokens.isNotEmpty()) {
                    val word = detokenize(bs.tokens.toList())
                    if (word.isNotBlank()) {
                        val norm = bs.logProb / bs.tokens.size
                        if (norm > finished.getOrDefault(word, Double.NEGATIVE_INFINITY)) {
                            finished[word] = norm
                        }
                    }
                }

                // Non-blank branches: top-K tokens (skip blank; it is handled above).
                for (tok in step.topTokens) {
                    if (tok.token == blankId) continue
                    val newFrame = if (step.duration > 0) bs.frame + step.duration else bs.frame
                    if (newFrame > endFrame) continue
                    val newTokens = bs.tokens.copyOf(bs.tokens.size + 1).also { it[bs.tokens.size] = tok.token }
                    next.add(
                        BeamState(newFrame, tok.token, step.lstm1, step.lstm2, newTokens,
                            bs.logProb + tok.logProb)
                    )
                }
            }
            // Prune to the beam's highest-scoring live states.
            live = next.sortedByDescending { it.logProb }.take(beamWidth)
        }

        // Length-normalise (already done at finish) and take the top alternatives.
        return finished.entries
            .sortedByDescending { it.value }
            .take(maxAlternatives)
            .map { (word, lp) -> WordAlternative(word, lp.toFloat()) }
    }

    /** Releases all native ONNX Runtime resources. Safe to call more than once. */
    override fun close() {
        prepSession?.close(); prepSession = null
        encSession?.close(); encSession = null
        decSession?.close(); decSession = null
        env?.close(); env = null
        isLoaded = false
        Log.d(TAG, "ParakeetEngine closed")
        logMemoryUsage()
    }

    /**
     * Converts [pcm] PCM samples to float32 in [-1.0, 1.0].
     *
     * Intentionally allocates a fresh array per call so that concurrent partial-inference
     * coroutines each operate on independent data. The allocation cost (~64 KB per 1-s
     * chunk) is negligible compared to the ONNX inference itself.
     */
    private fun normalizePcm(pcm: ShortArray): FloatArray {
        val out = FloatArray(pcm.size)
        for (i in pcm.indices) out[i] = pcm[i] / 32_768f
        return out
    }

    /**
     * Runs nemo128.onnx to convert raw audio to log-mel features.
     * Input names verified: `waveforms` (FLOAT) + `waveforms_lens` (INT64).
     * Outputs accessed by index: [0] = feature tensor, [1] = lengths.
     * The feature length is read from the tensor's time dimension to avoid
     * INT32/INT64 ambiguity on the length output.
     */
    private fun preprocess(env: OrtEnvironment, samples: FloatArray): Pair<OnnxTensor, Long> {
        val audioLen = samples.size.toLong()
        val prep = prepSession ?: run {
            Log.w(TAG, "No preprocessor - forwarding raw audio (shapes will mismatch)")
            return OnnxTensor.createTensor(
                env, FloatBuffer.wrap(samples), longArrayOf(1L, audioLen)
            ) to audioLen
        }

        val prepOutputNames = prep.outputNames.toList()
        val audioTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(samples), longArrayOf(1L, audioLen)
        )
        val lenTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(audioLen)), longArrayOf(1L)
        )
        val inputs = mapOf(Names.PREP_IN_AUDIO to audioTensor, Names.PREP_IN_LENGTH to lenTensor)

        return prep.run(inputs).use { result ->
            audioTensor.close(); lenTensor.close()
            val featTensor = result.get(prepOutputNames[0]).get() as OnnxTensor
            // Read time dimension directly from shape - avoids INT32/INT64 ambiguity on length output
            val featLen = featTensor.info.shape[2]   // shape: [batch, 128, T′]
            cloneTensor(env, featTensor) to featLen
        }
    }

    private fun encode(
        env: OrtEnvironment,
        session: OrtSession,
        features: OnnxTensor,
        featLen: Long,
    ): Pair<OnnxTensor, Int> {
        val lenTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(longArrayOf(featLen)), longArrayOf(1L)
        )
        val inputs = mapOf(Names.ENC_IN_SIGNAL to features, Names.ENC_IN_LENGTH to lenTensor)

        return session.run(inputs).use { result ->
            lenTensor.close()
            val outTensor = result.get(Names.ENC_OUT_SIGNAL)
                .orElseThrow { RuntimeException("Encoder output '${Names.ENC_OUT_SIGNAL}' not found") }
                    as OnnxTensor
            val lenOut = result.get(Names.ENC_OUT_LEN)
                .orElseThrow { RuntimeException("Encoder output '${Names.ENC_OUT_LEN}' not found") }
                    as OnnxTensor
            val encLen = lenOut.longBuffer[0].toInt()
            cloneTensor(env, outTensor) to encLen
        }
    }

    /**
     * Greedy TDT decoder - all details verified from logcat on 2026-03-23:
     *
     * Encoder output layout: [batch, enc_dim=1024, enc_time]  ← (B, D, T) NOT (B, T, D)
     *
     * decoder_joint inputs:
     *   encoder_outputs  FLOAT [1, 1024, 1]    one frame at a time
     *   targets          INT32 [1, 1]           previous token
     *   target_length    INT32 [1]              always 1
     *   input_states_1   FLOAT [2, 1, 640]      LSTM hidden
     *   input_states_2   FLOAT [2, 1, 640]      LSTM cell
     *
     * decoder_joint outputs  (by index):
     *   0  outputs          FLOAT [1, 1, 1, 8198]  joint logits
     *   1  prednet_lengths  INT32 [1]              ignored
     *   2  output_states_1  FLOAT [2, 1, 640]      updated LSTM hidden
     *   3  output_states_2  FLOAT [2, 1, 640]      updated LSTM cell
     *
     * Logit layout: [0..blankId] = token logits (blank at blankId=8193),
     *               [blankId+1 .. blankId+numDurations] = TDT duration logits
     *
     * TDT advance rule:
     *   blank:     t += max(1, predictedDuration)
     *   non-blank: emit token; t += predictedDuration (0 = stay at same frame)
     */
    private fun decodeRange(
        env: OrtEnvironment,
        session: OrtSession,
        encoderOut: OnnxTensor,
        totalLength: Int,
        frameStart: Int,
        frameEnd: Int,
        state: TdtState,
    ): DecodeRangeResult {
        // Encoder layout: [1, D, T]
        val encShape = encoderOut.info.shape
        val encDim = encShape[1].toInt()   // D = 1024

        val encData = FloatArray(encoderOut.floatBuffer.remaining())
        encoderOut.floatBuffer.rewind()
        encoderOut.floatBuffer.get(encData)

        // LSTM state buffers: [2, 1, 640] = 1280 floats. Seeded from the carried
        // [state] (zero + SOS for a fresh start) so streaming chunks continue where the
        // previous chunk left off.
        val stateShape = longArrayOf(2L, 1L, 640L)
        var lstmState1 = state.lstmState1.copyOf()
        var lstmState2 = state.lstmState2.copyOf()

        val decOutputNames = session.outputNames.toList()

        val hypothesis = mutableListOf<Int>()
        val emissions = mutableListOf<TokenEmission>()
        val stateSnapshots = mutableListOf<FrameState>()
        // Per-token log-softmax accumulators for confidence scoring. Softmax is computed
        // over the token portion [0..blankId]; log(softmax[argmax]) is accumulated for every
        // non-blank emission. The geometric-mean probability exp(mean(logProbs)) is a
        // principled 0.0–1.0 confidence that low-values hallucinations (the model emitting
        // tokens with flat / uncertain distributions on cold strides or noise) while
        // high-valuing clean speech. Used by InferenceRepository to suppress low-confidence
        // first-stride outputs that would otherwise displace real content.
        var logProbSum = 0.0
        var nonBlankEmissions = 0
        // prevToken is carried in from [state]; for a fresh start it is the blank token (see
        // [initialTdtState]), which the joint's predictor embedding accepts as its first input.
        // Subsequent steps feed the last emitted (non-blank) token.
        var prevToken = state.prevToken
        var blankFrames = state.blankFrames
        var t = frameStart
        var maxIter = (frameEnd - frameStart) * 20 + 50   // per-range safety cap
        var tokensAtFrame = 0             // per-frame guard against duration=0 loops
        val maxTokensPerFrame = 30
        val maxHypothesis = 2000     // ~20-30 s of speech at typical token rate

        while (t < frameEnd && maxIter-- > 0 && hypothesis.size < maxHypothesis) {
            // After two seconds without a token, start a new predictor context.
            // Keeping the previous sentence's LSTM state through a long pause can
            // suppress every token of the next sentence. Preserve the frame position
            // and accumulated transcript; only the predictor state is reset.
            if (blankFrames >= PREDICTOR_RESET_BLANK_FRAMES && prevToken != blankId) {
                lstmState1 = FloatArray(2 * 640)
                lstmState2 = FloatArray(2 * 640)
                prevToken = blankId
                RecordingTrace.log { "decoder reset after blank frames=$blankFrames" }
            }
            // References to the pre-call decoder state (zero-copy: the LSTM update below
            // allocates fresh arrays, so these keep pointing at the pre-call state).
            // Kept so an emitting call can record a [FrameState] snapshot for the local
            // word beam — the exact state the greedy decode had at the word's first token.
            val preLstm1 = lstmState1
            val preLstm2 = lstmState2
            val preToken = prevToken
            // Extract one encoder frame: encoder[0, :, t] → frame shape [1, D, 1]
            val frameData = FloatArray(encDim) { d -> encData[d * totalLength + t] }

            val frameTensor = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(frameData), longArrayOf(1L, encDim.toLong(), 1L)
            )
            // targets and target_length are INT32 (verified from logcat)
            val targetTensor = OnnxTensor.createTensor(
                env, IntBuffer.wrap(intArrayOf(prevToken)), longArrayOf(1L, 1L)
            )
            val targetLenTensor = OnnxTensor.createTensor(
                env, IntBuffer.wrap(intArrayOf(1)), longArrayOf(1L)
            )
            val statesTensor1 = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(lstmState1), stateShape
            )
            val statesTensor2 = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(lstmState2), stateShape
            )

            val inputs = mapOf(
                Names.DEC_IN_ENC_OUT to frameTensor,
                Names.DEC_IN_TARGETS to targetTensor,
                Names.DEC_IN_TARGET_LEN to targetLenTensor,
                Names.DEC_IN_STATES_1 to statesTensor1,
                Names.DEC_IN_STATES_2 to statesTensor2,
            )

            try {
                session.run(inputs).use { result ->
                    // Joint logits: [1, 1, 1, 8198] → flat array of 8198 floats
                    val logitsTensor = result.get(decOutputNames[0]).get() as OnnxTensor
                    val logits = FloatArray(logitsTensor.floatBuffer.remaining())
                    logitsTensor.floatBuffer.get(logits)

                    // Token: argmax over [0..blankId] (inclusive)
                    val predictedToken = (0..blankId).maxByOrNull { logits[it] } ?: blankId

                    // Per-token confidence + acoustic evidence: one pass over the token
                    // portion [0..blankId] computes the argmax's log-softmax (confidence)
                    // and the top-[TOP_K_TOKENS] tokens with their log-softmax
                    // probabilities — the acoustic runner-ups the greedy decode discards.
                    // Computed for non-blank emissions only.
                    if (predictedToken != blankId) {
                        val maxLogit = (0..blankId).maxOf { logits[it] }
                        var expSum = 0.0
                        // Top-K selection by raw logit (monotonic with log-softmax):
                        // maintain a sorted top-K while sweeping — O(V) time, K allocs.
                        val topIdx = IntArray(TOP_K_TOKENS)
                        val topVal = DoubleArray(TOP_K_TOKENS) { Double.NEGATIVE_INFINITY }
                        for (k in 0..blankId) {
                            val v = logits[k].toDouble()
                            expSum += Math.exp(v - maxLogit)
                            // Insert into the descending top-K (topVal[0] = best):
                            // find the slot from the bottom, shifting smaller values down.
                            if (v > topVal[TOP_K_TOKENS - 1]) {
                                var j = TOP_K_TOKENS - 1
                                while (j > 0 && v > topVal[j - 1]) {
                                    topVal[j] = topVal[j - 1]
                                    topIdx[j] = topIdx[j - 1]
                                    j--
                                }
                                topVal[j] = v
                                topIdx[j] = k
                            }
                        }
                        val logExpSum = Math.log(expSum)
                        val topTokens = (0 until TOP_K_TOKENS)
                            .map { i -> EmissionToken(topIdx[i], topVal[i] - maxLogit - logExpSum) }
                        // The argmax is topIdx[0] (ties keep the first max, matching
                        // maxByOrNull); its log-softmax is the per-token confidence.
                        val logSoftmaxArgmax = topVal[0] - maxLogit - logExpSum
                        logProbSum += logSoftmaxArgmax
                        nonBlankEmissions++
                        emissions.add(TokenEmission(predictedToken, t, logSoftmaxArgmax, topTokens))
                        stateSnapshots.add(FrameState(t, preLstm1, preLstm2, preToken))
                    }

                    // Duration: argmax over the last numDurations logits.
                    val nDur = numDurations
                    val predictedDur = if (nDur > 0) {
                        val durBase = blankId + 1
                        (0 until nDur.coerceAtLeast(1)).maxByOrNull { logits[durBase + it] } ?: 0
                    } else 0

                    // Update LSTM states only on non-blank emissions. The TDT predictor state
                    // must FREEZE during blank runs (matching transcribe-rs): it advances only
                    // when a real token is emitted, and the encoder frame carries the temporal
                    // context across blanks. Updating on every step (including blank) lets a
                    // low-margin short word collapse into a degenerate period loop.
                    if (predictedToken != blankId) {
                        if (decOutputNames.size > 2) {
                            val s1 = result.get(decOutputNames[2]).get() as OnnxTensor
                            lstmState1 = FloatArray(s1.floatBuffer.remaining()).also { s1.floatBuffer.get(it) }
                        }
                        if (decOutputNames.size > 3) {
                            val s2 = result.get(decOutputNames[3]).get() as OnnxTensor
                            lstmState2 = FloatArray(s2.floatBuffer.remaining()).also { s2.floatBuffer.get(it) }
                        }
                    }

                    // TDT advance rule
                    if (predictedToken == blankId) {
                        val advance = maxOf(1, predictedDur)
                        t += advance
                        blankFrames = (blankFrames + advance).coerceAtMost(PREDICTOR_RESET_BLANK_FRAMES)
                        tokensAtFrame = 0
                    } else {
                        blankFrames = 0
                        hypothesis.add(predictedToken)
                        prevToken = predictedToken
                        tokensAtFrame++

                        if (predictedDur > 0) {
                            t += predictedDur
                            tokensAtFrame = 0
                        } else if (tokensAtFrame >= maxTokensPerFrame) {
                            // Safety: force advance when duration=0 emissions pile up on one frame
                            Log.w(TAG, "decodeRange: stuck at frame $t - forcing advance")
                            t++
                            tokensAtFrame = 0
                        }
                        // else: predictedDur == 0 and within per-frame cap → stay on frame
                    }
                }
            } finally {
                frameTensor.close()
                targetTensor.close()
                targetLenTensor.close()
                statesTensor1.close()
                statesTensor2.close()
            }
        }

        val confidence = if (nonBlankEmissions > 0) {
            // Geometric mean of per-token probabilities: exp(mean(logProbs)).
            Math.exp(logProbSum / nonBlankEmissions).toFloat().coerceIn(0f, 1f)
        } else 1.0f
        // Frame position at exit relative to the range end. Positive: a duration jump at
        // the boundary landed past frameEnd — the one-shot decoder never visits those
        // frames, so the next chunk must start that many frames later (see TdtState).
        // Negative: the loop terminated early (safety cap) before reaching frameEnd — the
        // next chunk must resume at the actual position or those frames are skipped.
        val frameDelta = t - frameEnd
        if (frameDelta < 0) {
            Log.w(TAG, "decodeRange: terminated $frameDelta frame(s) before frameEnd=$frameEnd (t=$t) - safety cap hit?")
        }
        return DecodeRangeResult(
            hypothesis,
            TdtState(lstmState1, lstmState2, prevToken, frameDelta, blankFrames),
            confidence,
            logProbSum,
            nonBlankEmissions,
            emissions,
            stateSnapshots,
        )
    }

    private fun detokenize(tokenIds: List<Int>): String =
        decodeParakeetTokens(tokenIds, vocabulary, blankId)

    /**
     * Copies [source]'s float data into a new [OnnxTensor] so that the parent
     * `OrtSession.Result` can be safely closed before the tensor is consumed.
     */
    private fun cloneTensor(env: OrtEnvironment, source: OnnxTensor): OnnxTensor {
        val shape = source.info.shape
        val data = FloatArray(source.floatBuffer.remaining())
        source.floatBuffer.rewind()
        source.floatBuffer.get(data)
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
    }

    /** Prints all input/output tensor names and shapes for a loaded session. */
    private fun logSession(label: String, session: OrtSession) {
        Log.d(TAG, "=== $label inputs ===")
        session.inputNames.forEach { n -> Log.d(TAG, "  [$n] ${session.inputInfo[n]}") }
        Log.d(TAG, "=== $label outputs ===")
        session.outputNames.forEach { n -> Log.d(TAG, "  [$n] ${session.outputInfo[n]}") }
    }

    /**
     * Scans [vocabulary] for a known blank-token label and returns its index.
     * NeMo TDT ONNX exports use `<blk>` by convention; guarding a few variants
     * makes this robust across different export tools.
     * Returns `null` if no recognisable blank entry is found.
     */
    private fun scanVocabForBlankId(): Int? {
        val blankLabels = setOf("<blk>", "<blank>", "[blank]", "<eps>")
        vocabulary.forEachIndexed { idx, token ->
            if (token in blankLabels) {
                Log.d(TAG, "Detected blank token '$token' at index $idx from vocabulary scan")
                return idx
            }
        }
        return null
    }

    /**
     * Attempts to read `blank_id` from `config.json`.
     * Checks (in order): top-level, `model_defaults`, `decoder`, `tokenizer` sections.
     * Returns `null` if the file is absent or no key is found - caller falls back to vocabulary.size-1.
     */
    private fun parseBlankId(configFile: File): Int? {
        if (!configFile.exists()) return null
        return runCatching {
            val json = JSONObject(configFile.readText())
            // Top-level key (common in ONNX-converted NeMo exports)
            if (json.has("blank_id")) return@runCatching json.getInt("blank_id")
            // NeMo model_defaults section
            json.optJSONObject("model_defaults")?.let {
                if (it.has("blank_id")) return@runCatching it.getInt("blank_id")
            }
            // NeMo decoder section
            json.optJSONObject("decoder")?.let {
                if (it.has("blank_id")) return@runCatching it.getInt("blank_id")
            }
            // NeMo tokenizer section
            json.optJSONObject("tokenizer")?.let {
                if (it.has("blank_id")) return@runCatching it.getInt("blank_id")
            }
            null
        }.getOrNull()
    }

    private fun logMemoryUsage() {
        val runtime = Runtime.getRuntime()
        val usedMemMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val maxMemMB = runtime.maxMemory() / (1024 * 1024)
        Log.i(TAG, "ParakeetEngine memory: used=${usedMemMB}MB, max=${maxMemMB}MB")
        val debugMem = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugMem)
        Log.i(
            TAG,
            "Debug memory: dalvik=${debugMem.dalvikPrivateDirty}KB, native=${debugMem.nativePrivateDirty}KB, totalPss=${debugMem.totalPss}KB"
        )
    }
}

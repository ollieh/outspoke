package dev.brgr.outspoke

import ai.onnxruntime.OrtSession.SessionOptions.OptLevel
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.inference.InferenceRepository
import dev.brgr.outspoke.inference.ParakeetEngine
import dev.brgr.outspoke.inference.TranscriptResult
import dev.brgr.outspoke.settings.model.ModelId
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Opt-in, real-device comparison for Parakeet V3: the live chunked path versus one
 * encoder pass over the whole recording, across thread counts. Uses the downloaded model.
 */
@RunWith(AndroidJUnit4::class)
class V3LatencyBenchmarkTest {
    @Test
    fun compareStreamingAndSinglePass() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("v3Benchmark") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.filesDir, "models/${ModelId.PARAKEET_V3.storageDirName}")
        assertTrue("Download Parakeet V3 in the debug app before benchmarking", File(directory, "encoder-model.int8.onnx").isFile)
        val hello = instrumentation.context.assets.open("audio/hello-world.wav").use { BenchmarkAudio.readPcm(it.readBytes()) }
        val counting = instrumentation.context.assets.open("audio/natural-counting.wav").use { BenchmarkAudio.readPcm(it.readBytes()) }
        val gap = ShortArray(16_000)
        val minute = counting + gap + counting + gap + counting
        val fixtures = linkedMapOf("hello-world" to hello, "counting-18s" to counting, "counting-57s" to minute)

        for ((threads, optimization) in listOf(2 to OptLevel.BASIC_OPT, 4 to OptLevel.ALL_OPT, 6 to OptLevel.ALL_OPT)) {
            val engine = ParakeetEngine(intraOpThreads = threads, optimizationLevel = optimization)
            try {
                val loadStart = System.nanoTime()
                engine.load(directory)
                val loadMs = (System.nanoTime() - loadStart) / 1_000_000.0
                Log.i(TAG, "threads=$threads opt=$optimization loadMs=$loadMs")
                singlePass(engine, hello) // Warm the native sessions.
                for ((name, pcm) in fixtures) {
                    val audioMs = pcm.size / 16.0
                    val (passMs, passText) = singlePass(engine, pcm)
                    Log.i(TAG, "threads=$threads opt=$optimization fixture=$name mode=single-pass audioMs=$audioMs ms=$passMs rtf=${passMs / audioMs} words=${passText.split(' ').count { it.isNotBlank() }}")
                    if (name == "counting-57s") continue // The live path is linear; 18 s is enough.
                    val (streamMs, streamText) = streamed(engine, pcm)
                    Log.i(TAG, "threads=$threads opt=$optimization fixture=$name mode=streaming audioMs=$audioMs ms=$streamMs rtf=${streamMs / audioMs} words=${streamText.split(' ').count { it.isNotBlank() }}")
                    if (name == "hello-world") {
                        assertTrue("Single pass must keep short speech", passText.lowercase().contains("hello"))
                    }
                }
            } finally {
                engine.close()
            }
        }
    }

    /** One preprocess + encoder call over all audio, then greedy TDT over every frame. */
    private fun singlePass(engine: ParakeetEngine, pcm: ShortArray): Pair<Double, String> {
        val start = System.nanoTime()
        val samples = FloatArray(pcm.size) { pcm[it] / 32768f }
        val (encoded, frames) = engine.encodeBuffer(samples)
        val text = try {
            val decoded = engine.decodeChunk(encoded, frames, 0, frames, engine.initialTdtState())
            engine.detokenizeTokens(decoded.tokens)
        } finally {
            encoded.close()
        }
        return (System.nanoTime() - start) / 1_000_000.0 to text
    }

    /** The production chunked path, fed 30 ms chunks as fast as it accepts them. */
    private fun streamed(engine: ParakeetEngine, pcm: ShortArray): Pair<Double, String> = runBlocking {
        val repository = InferenceRepository(engine)
        val audio = flow {
            for (offset in pcm.indices step 480) emit(AudioChunk(pcm.copyOfRange(offset, minOf(offset + 480, pcm.size))))
        }
        val start = System.nanoTime()
        val finals = StringBuilder()
        repository.transcribe(audio).collect { result ->
            if (result is TranscriptResult.Final) finals.append(result.text).append(' ')
        }
        (System.nanoTime() - start) / 1_000_000.0 to finals.toString().trim()
    }

    private companion object {
        const val TAG = "OutspokeBenchmark"
    }
}

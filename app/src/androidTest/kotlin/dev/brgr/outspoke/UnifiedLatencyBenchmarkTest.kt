package dev.brgr.outspoke

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.brgr.outspoke.inference.ParakeetUnifiedEngine
import dev.brgr.outspoke.settings.model.ModelId
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Opt-in, real-device comparison. Uses already downloaded models and test-only audio. */
@RunWith(AndroidJUnit4::class)
class UnifiedLatencyBenchmarkTest {
    @Test
    fun compareThreadCounts() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("unifiedBenchmark") == "true")
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val directory = File(instrumentation.targetContext.filesDir, "models/${ModelId.PARAKEET_UNIFIED_EN.storageDirName}")
        assertTrue("Download Unified in the debug app before benchmarking", File(directory, "encoder.int8.onnx").isFile)
        val fixtures = listOf("hello-world.wav", "natural-counting.wav").associateWith { name ->
            instrumentation.context.assets.open("audio/$name").use { BenchmarkAudio.readPcm(it.readBytes()) }
        }
        for (threads in listOf(1, 2, 4)) {
            val engine = ParakeetUnifiedEngine(intraOpThreads = threads)
            try {
                engine.load(directory)
                replay(engine, fixtures.getValue("hello-world.wav")) // Warm the native sessions.
                for ((name, pcm) in fixtures) {
                    val measurements = List(if (name == "hello-world.wav") 3 else 1) {
                        replay(engine, pcm).also { result ->
                            if (name == "hello-world.wav") {
                                val words = result.text.lowercase().replace(Regex("[^a-z ]"), "")
                                assertTrue("Short speech must survive finalization", words.contains("hello world"))
                            } else {
                                var position = 0
                                for (number in listOf("one|1", "two|2", "three|3", "four|4", "fifth|5th",
                                    "sixth|6th", "seventh|7th", "eighth|8th", "nine|9", "ten|10")) {
                                    val match = Regex("\\b($number)\\b", RegexOption.IGNORE_CASE).find(result.text, position)
                                    assertTrue("Numbered sentences must survive streaming", match != null)
                                    position = match!!.range.last + 1
                                }
                            }
                        }
                    }
                    val median = measurements.sortedBy { it.feedMs + it.finishMs }[measurements.size / 2]
                    val audioMs = pcm.size / 16.0
                    Log.i("OutspokeBenchmark", "threads=$threads fixture=$name audioMs=$audioMs feedMs=${median.feedMs} finishMs=${median.finishMs} rtf=${(median.feedMs + median.finishMs) / audioMs}")
                }
            } finally {
                engine.close()
            }
        }
    }

    private data class Measurement(val text: String, val feedMs: Double, val finishMs: Double)

    private fun replay(engine: ParakeetUnifiedEngine, pcm: ShortArray): Measurement = engine.newStream().use { stream ->
        val start = System.nanoTime()
        for (offset in pcm.indices step 480) stream.accept(pcm.copyOfRange(offset, minOf(offset + 480, pcm.size)))
        val finishStart = System.nanoTime()
        val final = stream.finish()
        Measurement(final.text, (finishStart - start) / 1_000_000.0, (System.nanoTime() - finishStart) / 1_000_000.0)
    }
}

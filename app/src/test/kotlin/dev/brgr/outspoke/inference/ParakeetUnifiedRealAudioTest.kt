package dev.brgr.outspoke.inference

import android.text.InputType
import android.view.inputmethod.EditorInfo
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.audio.SileroVadFilter
import kotlin.math.sqrt
import dev.brgr.outspoke.ime.FakeInputConnection
import dev.brgr.outspoke.ime.TextInjector
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.assertj.core.api.Assertions.assertThat
import org.junit.AfterClass
import org.junit.Assume
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/** Opt in with -Ptest.unified.model.dir pointing at the pinned streaming ONNX export. */
class ParakeetUnifiedRealAudioTest {
    companion object {
        private val engine = ParakeetUnifiedEngine()

        @BeforeClass @JvmStatic
        fun load() {
            val directory = System.getProperty("test.unified.model.dir")
            Assume.assumeTrue("Supply -Ptest.unified.model.dir", !directory.isNullOrBlank())
            engine.load(File(directory!!))
        }

        @AfterClass @JvmStatic
        fun close() = engine.close()
    }

    private fun samples(name: String): ShortArray = javaClass.classLoader!!
        .getResourceAsStream("audio/$name")!!.use { input ->
            val wav = WavReader().readPcm16(input)
            WavReader().resampleLinear(wav.samples, wav.sampleRate, 16_000)
        }

    private fun runPipeline(pcm: ShortArray, filterSilence: Boolean = false): Pair<String, List<TranscriptResult>> = runBlocking {
        val chunks = pcm.indices.step(480).map { start ->
            AudioChunk(pcm.copyOfRange(start, minOf(pcm.size, start + 480)))
        }
        val audio = if (!filterSilence) chunks else {
            val model = listOf(File("src/main/res/raw/silero_vad_v4.onnx"), File("app/src/main/res/raw/silero_vad_v4.onnx"))
                .first { it.isFile }.readBytes()
            val vad = SileroVadFilter(model)
            try {
                buildList {
                    chunks.forEach { chunk ->
                        val rms = sqrt(chunk.samples.sumOf { it.toDouble() * it } / chunk.samples.size) / 32767.0
                        addAll(vad.process(chunk, rms.toFloat()))
                    }
                    repeat(20) { if (vad.isSpeechActive) addAll(vad.process(AudioChunk(ShortArray(480)), 0f)) }
                }
            } finally {
                vad.flush()
                vad.close()
            }
        }
        val results = InferenceRepository(engine).transcribe(audio.asFlow()).toList()
        assertThat(results.filterIsInstance<TranscriptResult.Failure>()).isEmpty()
        val field = FakeInputConnection()
        val injector = TextInjector(field, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT })
        results.forEach { result ->
            when (result) {
                is TranscriptResult.Partial -> injector.setPartial(result.text)
                is TranscriptResult.Final -> injector.commitFinal(result.text)
                else -> Unit
            }
        }
        val finalText = results.filterIsInstance<TranscriptResult.Final>().joinToString(" ") { it.text }
        assertThat(field.fieldText.trim()).isEqualTo(finalText.trim())
        println("UNIFIED FINAL $finalText")
        finalText to results
    }

    @Test
    fun `streaming retains ten numbered sentences through editor insertion`() {
        val (text, results) = runPipeline(samples("natural-counting.wav"))
        assertThat(results.filterIsInstance<TranscriptResult.Partial>().size).isGreaterThan(5)
        val numbers = listOf("one|1", "two|2", "three|3", "four|4", "fifth|5th", "sixth|6th",
            "seventh|7th", "eighth|8th", "nine|9", "ten|10")
        var position = 0
        for (number in numbers) {
            val match = Regex("\\b($number)\\b", RegexOption.IGNORE_CASE).find(text, position)
            assertThat(match).describedAs("Missing $number after $position in $text").isNotNull()
            position = match!!.range.last + 1
        }
    }

    @Test
    fun `supplied human recording retains every numbered sentence with silence filtering`() {
        val recording = System.getProperty("test.counting.wav")
        Assume.assumeTrue("Supply the private recording via -Ptest.counting.wav", !recording.isNullOrBlank())
        val wav = File(recording!!).inputStream().use { WavReader().readPcm16(it) }
        val (text, _) = runPipeline(WavReader().resampleLinear(wav.samples, wav.sampleRate, 16_000), filterSilence = true)
        val expected = listOf("sentence number (one|1)", "sentence number (two|2)", "(third|3rd) sentence",
            "(fourth|4th)", "(fifth|5th) sentence", "sentence number (six|6)",
            "sentence number (seven|7)", "sentence number (eight|8)", "(ninth|9th) sentence", "(tenth|10th) sentence")
        for (phrase in expected) {
            assertThat(Regex("\\b($phrase)\\b", RegexOption.IGNORE_CASE).findAll(text).count())
                .describedAs("$phrase occurs once: $text").isEqualTo(1)
        }
        assertThat(text.lowercase()).contains("test audio", "that completes all the sentences")
    }

    @Test
    fun `stop flushes short speech and fresh streams do not inherit earlier text`() {
        for ((fixture, expected) in listOf("single-yes.wav" to "yes", "hello-world.wav" to "hello world")) {
            val (text, _) = runPipeline(samples(fixture))
            assertTranscriptionClose(expected, text)
        }
    }

    @Test
    fun `empty and silent streams do not inherit previous words`() {
        engine.newStream().use { assertThat(it.finish().text).isEmpty() }
        engine.newStream().use {
            it.accept(ShortArray(24_000))
            assertThat(it.finish().text).isEmpty()
        }
    }
}

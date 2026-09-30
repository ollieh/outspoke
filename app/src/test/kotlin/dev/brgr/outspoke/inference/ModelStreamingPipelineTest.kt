package dev.brgr.outspoke.inference

import dev.brgr.outspoke.audio.AudioChunk
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ModelStreamingPipelineTest {
    private class Stream(val text: String, val failOnFinish: Boolean = false) : SpeechStream {
        var closes = 0
        var finishes = 0
        override fun accept(samples: ShortArray) = TranscriptResult.Partial(text)
        override fun finish(): TranscriptResult.Final {
            finishes++
            check(!failOnFinish) { "final decode failed" }
            return TranscriptResult.Final(text)
        }
        override fun close() { closes++ }
    }

    private class Engine(val streams: List<Stream>) : StreamingSpeechEngine {
        var next = 0
        override val isLoaded = true
        override fun load(modelDir: File) = Unit
        override fun close() = Unit
        override fun transcribe(chunk: AudioChunk): TranscriptResult = error("Must use streaming")
        override fun newStream(): SpeechStream = streams[next++]
    }

    private val audio = AudioChunk(shortArrayOf(100))
    private val boundary = AudioChunk(ShortArray(0), isSilenceBoundary = true)

    @Test
    fun `silence commits and closes each utterance without ending the recording`() = runBlocking {
        val streams = listOf(Stream("first sentence"), Stream("next sentence"))
        val engine = Engine(streams)
        val results = InferenceRepository(engine).transcribe(flowOf(audio, boundary, boundary, audio),
            postprocessingEnabled = false).toList()
        val finals = results.filterIsInstance<TranscriptResult.Final>()
        assertEquals(listOf("first sentence", "next sentence"), finals.map { it.text })
        assertEquals(listOf(true, false), finals.map { it.isUtteranceBoundary })
        assertEquals(2, engine.next)
        streams.forEach { assertEquals(1, it.finishes); assertEquals(1, it.closes) }
    }

    @Test
    fun `stop after silence does not duplicate text or report no speech`() = runBlocking {
        val stream = Stream("already committed")
        val results = InferenceRepository(Engine(listOf(stream))).transcribe(flowOf(audio, boundary),
            postprocessingEnabled = false).toList()
        assertEquals(1, results.filterIsInstance<TranscriptResult.Final>().size)
        assertFalse(results.any { it is TranscriptResult.NoSpeech })
        assertEquals(1, stream.closes)
    }

    @Test
    fun `finish failure closes the stream and reaches the UI`() = runBlocking {
        val stream = Stream("partial text", failOnFinish = true)
        val results = InferenceRepository(Engine(listOf(stream))).transcribe(flowOf(audio)).toList()
        assertEquals("final decode failed", results.filterIsInstance<TranscriptResult.Failure>().single().cause.message)
        assertEquals(1, stream.closes)
    }

    @Test
    fun `cancellation closes without committing unfinished speech`() = runBlocking {
        withTimeout(5000) {
            val stream = Stream("unfinished")
            val received = CompletableDeferred<Unit>()
            val recording = launch {
                InferenceRepository(Engine(listOf(stream))).transcribe(flow {
                    emit(audio)
                    awaitCancellation()
                }).collect { received.complete(Unit) }
            }
            received.await()
            recording.cancelAndJoin()
            assertEquals(1, stream.closes)
            assertEquals(0, stream.finishes)
        }
    }

    @Test
    fun `custom words apply to model-owned partial and final text`() = runBlocking {
        val results = InferenceRepository(Engine(listOf(Stream("use out spoke today")))).transcribe(
            flowOf(audio), customWords = listOf("Outspoke"), postprocessingEnabled = false,
        ).toList()
        assertEquals("use Outspoke today", results.filterIsInstance<TranscriptResult.Partial>().single().text)
        assertEquals("use Outspoke today", results.filterIsInstance<TranscriptResult.Final>().single().text)
    }
}

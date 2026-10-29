package dev.brgr.outspoke.ui.keyboard

import android.Manifest
import android.media.AudioRecord
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.lifecycle.ViewModelStore
import dev.brgr.outspoke.audio.AudioCaptureManager
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.ime.DictationRescue
import dev.brgr.outspoke.ime.FakeInputConnection
import dev.brgr.outspoke.ime.TextInjector
import dev.brgr.outspoke.ime.WordSuggestionProvider
import dev.brgr.outspoke.inference.EngineState
import dev.brgr.outspoke.inference.InferenceRepository
import dev.brgr.outspoke.inference.SpeechStream
import dev.brgr.outspoke.inference.StreamingSpeechEngine
import dev.brgr.outspoke.inference.TranscriptResult
import dev.brgr.outspoke.inference.TranscriptSource
import dev.brgr.outspoke.remote.PcmRecording
import dev.brgr.outspoke.remote.ServerTranscriber
import dev.brgr.outspoke.remote.ServerTranscriptionSettings
import dev.brgr.outspoke.settings.preferences.AppPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowAudioRecord
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/** The keyboard's two server-race hooks: the raw audio tee and the final text choice. */
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], manifest = Config.NONE)
class KeyboardViewModelServerRaceTest {
    private val uploads = CopyOnWriteArrayList<PcmRecording>()
    private val rescue = object : DictationRescue {
        override fun save(text: String) = Unit
        override fun reportLost() = Unit
    }

    @Test
    fun `server text replaces the phone's and the upload holds all captured audio`() = runTest {
        withKeyboard(ScriptedEngine("Phone words."), server = { "Server words." }) { model, field, source ->
            repeat(10) { source.add(ShortArray(480) { 1000 }) }
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Server words.")
            assertThat(uploads.single().sampleCount).isEqualTo(4_800)
            assertThat(model.diagnostics.value.transcriptSource).isEqualTo(TranscriptSource.SERVER)
        }
    }

    @Test
    fun `server text is formatted with the user's preferences`() = runTest {
        withKeyboard(ScriptedEngine("Phone."), server = { "Send twelve copies to out spoke." }, configure = {
            it.addCustomWord("Outspoke")
        }, startWhen = { it.customWords.value.isNotEmpty() }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Send 12 copies to Outspoke.")
        }
    }

    @Test
    fun `a server error keeps the phone's text`() = runTest {
        withKeyboard(ScriptedEngine("Phone words."), server = { throw IOException("refused") }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Phone words.")
            assertThat(model.diagnostics.value.transcriptSource).isEqualTo(TranscriptSource.PHONE)
        }
    }

    @Test
    fun `the server rescues a failed final decode`() = runTest {
        withKeyboard(ScriptedEngine(null), server = { "Server words." }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Server words.")
        }
    }

    @Test
    fun `both failing keeps the error and the preview`() = runTest {
        withKeyboard(ScriptedEngine(null), server = { throw IOException("refused") }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { uploads.isNotEmpty() && model.uiState.value is KeyboardUiState.Error }
            // Let the race resolve after the server's failure.
            repeat(20) { runCurrent(); Thread.sleep(5) }

            assertThat((model.uiState.value as KeyboardUiState.Error).partial).isEqualTo("Phone preview")
            assertThat(field.commitCalls).isEmpty()
        }
    }

    @Test
    fun `no server URL means no upload and no badge`() = runTest {
        withKeyboard(ScriptedEngine("Phone words."), settings = ServerTranscriptionSettings(), server = { "Server." }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Phone words.")
            assertThat(uploads).isEmpty()
            assertThat(model.diagnostics.value.transcriptSource).isNull()
        }
    }

    @Test
    fun `a non-English recording stays on the phone`() = runTest {
        withKeyboard(ScriptedEngine("Telefon."), server = { "Server." }, configure = {
            it.setForcedLanguage("de")
        }, startWhen = { it.currentLanguage.value == "de" }) { model, field, source ->
            source.add(ShortArray(480) { 1000 })
            awaitCondition { source.isEmpty() }
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }

            assertThat(field.commitCalls).containsExactly("Telefon.")
            assertThat(uploads).isEmpty()
        }
    }

    /** Streams a fixed preview and finishes with [final], or fails when it is null. */
    private class ScriptedEngine(private val final: String?) : StreamingSpeechEngine {
        override val isLoaded = true
        override fun load(modelDir: File) = Unit
        override fun close() = Unit
        override fun transcribe(chunk: AudioChunk): TranscriptResult = error("Use the stream")
        override fun newStream(): SpeechStream = object : SpeechStream {
            override fun accept(samples: ShortArray) = TranscriptResult.Partial("Phone preview")
            override fun finish() = TranscriptResult.Final(checkNotNull(final) { "Final decoding failed" })
            override fun close() = Unit
        }
    }

    private fun TestScope.withKeyboard(
        engine: StreamingSpeechEngine,
        settings: ServerTranscriptionSettings = ServerTranscriptionSettings(baseUrl = "http://vps:8000"),
        server: suspend () -> String,
        configure: suspend (AppPreferences) -> Unit = {},
        startWhen: (KeyboardViewModel) -> Boolean = { true },
        test: (KeyboardViewModel, FakeInputConnection, LinkedBlockingQueue<ShortArray>) -> Unit,
    ) {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val context = RuntimeEnvironment.getApplication()
        shadowOf(context).grantPermissions(Manifest.permission.RECORD_AUDIO)
        val preferences = AppPreferences(context)
        runBlocking {
            preferences.setVadSensitivity(false)
            preferences.setTriggerMode("TAP_TOGGLE")
            preferences.setAutoStartRecording(false)
            preferences.setForcedLanguage(null)
            preferences.customWords.first().forEach { preferences.removeCustomWord(it) }
            configure(preferences)
        }
        val frames = LinkedBlockingQueue<ShortArray>()
        val recorders = ConcurrentHashMap<AudioRecord, Boolean>()
        ShadowAudioRecord.setSourceProvider { recorder ->
            recorders[recorder] = true
            object : ShadowAudioRecord.AudioRecordSource {
                override fun readInShortArray(audioData: ShortArray, offset: Int, size: Int, isBlocking: Boolean): Int {
                    val frame = (if (isBlocking) frames.poll(20, TimeUnit.MILLISECONDS) else frames.poll()) ?: return 0
                    frame.copyInto(audioData, offset)
                    return frame.size
                }
            }
        }
        val transcriber = ServerTranscriber(flowOf(settings), { _, recording, _ ->
            uploads.add(recording)
            server()
        })
        val suggestions = WordSuggestionProvider(context)
        val sessionScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
        val model = KeyboardViewModel(AudioCaptureManager(context), preferences, suggestions, rescue, sessionScope, transcriber)
        val store = ViewModelStore().apply { put("keyboard", model) }
        val field = FakeInputConnection()
        model.setTextInjector(TextInjector(field, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }))
        model.setInferenceRepository(InferenceRepository(engine))
        model.setEngineState(EngineState.Ready)
        try {
            // Recording snapshots its preferences when it starts.
            awaitCondition { !model.vadSensitivity.value && startWhen(model) }
            model.onRecordStart()
            awaitCondition { recorders.isNotEmpty() }
            test(model, field, frames)
        } finally {
            store.clear()
            runCurrent()
            sessionScope.cancel()
            runCurrent()
            suggestions.close()
            ShadowAudioRecord.clearSource()
            Dispatchers.resetMain()
            runBlocking { preferences.customWords.first().forEach { preferences.removeCustomWord(it) } }
        }
    }

    private fun TestScope.awaitCondition(condition: () -> Boolean) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (!condition() && System.nanoTime() < deadline) {
            runCurrent()
            Thread.sleep(5)
        }
        assertThat(condition()).isTrue()
    }
}

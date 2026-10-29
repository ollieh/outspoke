package dev.brgr.outspoke.ui.keyboard

import android.Manifest
import android.media.AudioRecord
import android.text.InputType
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnectionWrapper
import androidx.lifecycle.ViewModelStore
import dev.brgr.outspoke.audio.AudioCaptureManager
import dev.brgr.outspoke.ime.DictationRescue
import dev.brgr.outspoke.ime.FakeInputConnection
import dev.brgr.outspoke.ime.TextInjector
import dev.brgr.outspoke.ime.WordSuggestionProvider
import dev.brgr.outspoke.inference.EngineState
import dev.brgr.outspoke.inference.FakeStreamingParakeet
import dev.brgr.outspoke.inference.InferenceRepository
import dev.brgr.outspoke.inference.SpeechEngine
import dev.brgr.outspoke.inference.SpeechStream
import dev.brgr.outspoke.inference.StreamingSpeechEngine
import dev.brgr.outspoke.inference.TranscriptResult
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.settings.preferences.AppPreferences
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
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
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.CountDownLatch
import java.io.File

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], manifest = Config.NONE)
class KeyboardViewModelRecordingTest {
    private val rescue = RecordingRescue()
    private lateinit var store: ViewModelStore

    @Test
    fun `revisable preview leaves the field untouched and inserts once after final decoding`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            source.frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            assertThat(field.fieldText).isEmpty()
            assertThat(field.commitCalls).isEmpty()
            assertThat(field.composingCalls).isZero()
            source.frames.add(ShortArray(480) { 1000 })
            awaitCondition { (model.uiState.value as? KeyboardUiState.Processing)?.partial == "UI works" }
            model.onRecordStop()
            awaitCondition { engine.finishing.count == 0L }
            assertThat((model.uiState.value as KeyboardUiState.Transcribing).partial).isEqualTo("UI works")
            assertThat(field.fieldText).isEmpty()
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(field.fieldText).isEqualTo("UI works.")
            assertThat(field.commitCalls).containsExactly("UI works.")
            assertThat(field.composingCalls).isZero()
        }
    }

    // Previously the dictation was discarded; now it is rescued, and still never
    // inserted into the editor that replaced its own.
    @Test
    fun `switching editor during final decode copies the dictation instead of inserting it`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onRecordStop()
            awaitCondition { engine.finishing.count == 0L }
            val replacement = FakeInputConnection()
            model.setTextInjector(TextInjector(replacement, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }))
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Idle)
            engine.release.countDown()
            awaitCondition { rescue.saved.isNotEmpty() }
            assertThat(rescue.saved).containsExactly("UI works.")
            assertThat(field.fieldText).isEmpty()
            assertThat(replacement.fieldText).isEmpty()
        }
    }

    @Test
    fun `final decode failure preserves the preview without modifying existing editor text`() = runTest {
        val engine = PreviewEngine(failOnFinish = true)
        withKeyboard(engine) { model, field, sources ->
            field.commitText("Existing text", 1)
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onRecordStop()
            engine.release.countDown()
            awaitCondition { model.uiState.value is KeyboardUiState.Error }
            assertThat((model.uiState.value as KeyboardUiState.Error).partial).isEqualTo("You")
            assertThat(field.fieldText).isEqualTo("Existing text")
            assertThat(field.commitCalls).containsExactly("Existing text")
            assertThat(rescue.saved).containsExactly("You")
        }
    }

    // Previously the preview was discarded; now the recording stops, finishes in the
    // background and its text goes to the clipboard instead of the finished editor.
    @Test
    fun `input finish mid-recording finishes the transcription and copies it`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onInputFinished()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Idle)
            assertThat(model.isRecording.value).isFalse()
            awaitCondition { engine.finishing.count == 0L }
            engine.release.countDown()
            awaitCondition { rescue.saved.isNotEmpty() }
            assertThat(rescue.saved).containsExactly("UI works.")
            assertThat(field.commitCalls).isEmpty()
            assertThat(field.composingCalls).isZero()
        }
    }

    @Test
    fun `an editor rejecting the insertion keeps the finished transcript visible`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            val rejecting = object : InputConnectionWrapper(field, false) {
                override fun commitText(text: CharSequence?, newCursorPosition: Int) = false
            }
            model.setTextInjector(TextInjector(rejecting, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }))
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onRecordStop()
            engine.release.countDown()
            awaitCondition { model.uiState.value is KeyboardUiState.Error }
            val error = model.uiState.value as KeyboardUiState.Error
            assertThat(error.reason).isEqualTo(KeyboardUiState.ErrorReason.TextInsertionFailed)
            assertThat(error.partial).isEqualTo("UI works.")
            assertThat(field.commitCalls).isEmpty()
            assertThat(rescue.saved).containsExactly("UI works.")
        }
    }

    // Previously a cleared field cancelled the recording; the dictation now survives
    // and goes into the same, now empty, field.
    @Test
    fun `clearing the field while finishing still inserts the recording`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            field.commitText("Draft", 1)
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onRecordStop()
            awaitCondition { engine.finishing.count == 0L }
            model.deleteAll()
            model.onFieldCleared()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Transcribing("You"))
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(field.fieldText).isEqualTo("UI works.")
            assertThat(rescue.saved).isEmpty()
        }
    }

    @Test
    fun `clearing the field mid-recording keeps listening and inserts everything dictated`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            field.commitText("Draft", 1)
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            source.frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value == KeyboardUiState.Processing("You") }
            // Backspace to empty and the delete key both reach the IME as an empty field.
            model.deleteAll()
            model.onFieldCleared()
            assertThat(field.fieldText).isEmpty()
            assertThat(model.isRecording.value).isTrue()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Processing("You"))
            source.frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value == KeyboardUiState.Processing("UI works") }
            assertThat(sources).describedAs("capture must not restart").isEmpty()
            model.onRecordStop()
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(field.fieldText).isEqualTo("UI works.")
        }
    }

    @Test
    fun `send inserts the dictation and then performs the editor's send action`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, _, sources ->
            val chat = FakeInputConnection().apply {
                onEditorAction = { if (it == EditorInfo.IME_ACTION_SEND) clearField() }
            }
            model.setTextInjector(TextInjector(chat, EditorInfo().apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                imeOptions = EditorInfo.IME_ACTION_SEND
            }))
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onSend()
            assertThat(model.uiState.value).isInstanceOf(KeyboardUiState.Transcribing::class.java)
            assertThat(chat.submissions).isEmpty()
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(chat.submissions).containsExactly(
                FakeInputConnection.Submission(EditorInfo.IME_ACTION_SEND, null, "UI works."),
            )
            assertThat(chat.fieldText).isEmpty()
        }
    }

    @Test
    fun `enter while dictating finishes and sends, and inserts a newline when idle`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, _, sources ->
            // Telegram's message box: multi-line with Android's default Done; it sends on
            // IME_ACTION_SEND and then empties the field.
            val chat = FakeInputConnection().apply {
                onEditorAction = { if (it == EditorInfo.IME_ACTION_SEND) clearField() }
            }
            model.setTextInjector(TextInjector(chat, EditorInfo().apply {
                inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                imeOptions = EditorInfo.IME_ACTION_DONE or EditorInfo.IME_FLAG_NO_ENTER_ACTION
            }))
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.performEnterAction()
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(chat.submissions).containsExactly(
                FakeInputConnection.Submission(EditorInfo.IME_ACTION_SEND, null, "UI works."),
            )
            assertThat(chat.fieldText).isEmpty()
            model.performEnterAction()
            assertThat(chat.fieldText).isEqualTo("\n")
        }
    }

    @Test
    fun `switch key waits for the final words before switching`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            var textAtSwitch: String? = null
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onSwitchKeyboard { textAtSwitch = field.fieldText }
            assertThat(model.isRecording.value).isFalse()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Transcribing("You"))
            awaitCondition { engine.finishing.count == 0L }
            runCurrent()
            assertThat(textAtSwitch).isNull()
            engine.release.countDown()
            awaitCondition { textAtSwitch != null }
            assertThat(textAtSwitch).isEqualTo("UI works.")
            var idleSwitches = 0
            model.onSwitchKeyboard { idleSwitches++ }
            assertThat(idleSwitches).isEqualTo(1)
        }
    }

    @Test
    fun `done returns to the previous keyboard after inserting when enabled`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine, configure = { it.setReturnToPreviousKeyboard(true) }) { model, field, sources ->
            var textAtReturn: String? = null
            model.onReturnToPreviousKeyboard = { textAtReturn = field.fieldText }
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onDone()
            engine.release.countDown()
            awaitCondition { textAtReturn != null }
            assertThat(textAtReturn).isEqualTo("UI works.")
        }
    }

    @Test
    fun `discard drops the dictation without inserting or copying it`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            model.onDiscard()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Idle)
            assertThat(model.isRecording.value).isFalse()
            engine.release.countDown()
            awaitCondition { engine.closed.count == 0L }
            runCurrent()
            assertThat(field.commitCalls).isEmpty()
            assertThat(rescue.saved).isEmpty()
        }
    }

    @Test
    fun `a destroyed keyboard still finishes the dictation and copies it`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            store.clear()
            awaitCondition { engine.finishing.count == 0L }
            engine.release.countDown()
            awaitCondition { rescue.saved.isNotEmpty() }
            assertThat(rescue.saved).containsExactly("UI works.")
            assertThat(field.commitCalls).isEmpty()
        }
    }

    @Test
    fun `a destroyed keyboard reports speech it could not transcribe yet`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList()), EngineState.Loading) { model, _, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value == KeyboardUiState.Listening(modelLoading = true) }
            store.clear()
            runCurrent()
            assertThat(rescue.lost).isEqualTo(1)
        }
    }

    @Test
    fun `restarting input on the same field keeps the dictation for that field`() = runTest {
        val engine = PreviewEngine()
        withKeyboard(engine) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            sources.remove().frames.add(ShortArray(480) { 1000 })
            awaitCondition { model.uiState.value is KeyboardUiState.Processing }
            val restarted = FakeInputConnection()
            model.setTextInjector(
                TextInjector(restarted, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }),
                sameEditor = true,
            )
            assertThat(model.isRecording.value).isTrue()
            model.onRecordStop()
            engine.release.countDown()
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(restarted.fieldText).isEqualTo("UI works.")
            assertThat(field.fieldText).isEmpty()
        }
    }

    @Test
    fun `listening is shown only once the microphone delivers audio`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList())) { model, _, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            repeat(5) {
                Thread.sleep(20)
                runCurrent()
            }
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.MicStarting)
            source.frames.add(ShortArray(480))
            awaitCondition { model.uiState.value == KeyboardUiState.Listening(modelLoading = false) }
        }
    }

    private class PreviewEngine(private val failOnFinish: Boolean = false) : StreamingSpeechEngine {
        val finishing = CountDownLatch(1)
        val release = CountDownLatch(1)
        val closed = CountDownLatch(1)
        override val isLoaded = true
        override fun load(modelDir: File) = Unit
        override fun close() = Unit
        override fun transcribe(chunk: AudioChunk): TranscriptResult = error("Use the stream")
        override fun newStream(): SpeechStream = object : SpeechStream {
            private var accepts = 0
            override fun accept(samples: ShortArray) = TranscriptResult.Partial(
                if (++accepts == 1) "You" else "UI works",
            )
            override fun finish(): TranscriptResult.Final {
                finishing.countDown()
                check(release.await(5, TimeUnit.SECONDS)) { "Timed out waiting to finish" }
                check(!failOnFinish) { "Final decoding failed" }
                return TranscriptResult.Final("UI works.")
            }
            override fun close() { closed.countDown() }
        }
    }

    @Test
    fun `every tap toggle recording delivers its transcript`() = runTest {
        withKeyboard(FakeStreamingParakeet(
            listOf(0 to "Hello", 8000 to "from", 16000 to "this", 24000 to "recording.")
        )) { model, field, sources ->
            repeat(6) { recording ->
                model.onRecordStart()
                awaitCondition { sources.isNotEmpty() }
                val source = sources.remove()
                // Alternate clips finalized before the first partial with clips
                // long enough to show live text (2 s chunk + 4 s right context).
                val frameCount = if (recording % 2 == 0) 90 else 225
                repeat(frameCount) { source.frames.add(ShortArray(480) { 1000 }) }
                awaitCondition { source.frames.isEmpty() }
                if (frameCount == 225) {
                    awaitCondition { model.uiState.value is KeyboardUiState.Processing }
                }
                assertThat(model.amplitude.value).isGreaterThan(0f)
                model.onRecordStop()
                assertThat(model.uiState.value).isInstanceOf(KeyboardUiState.Transcribing::class.java)
                awaitCondition { model.uiState.value == KeyboardUiState.Idle }
                assertThat(field.fieldText.trim())
                    .describedAs("Recording ${recording + 1} must not disappear")
                    .isEqualTo(List(recording + 1) { "Hello from this recording." }.joinToString(" "))
            }
        }
    }

    @Test
    fun `blank recognition after moving waveform reports no speech instead of silently returning idle`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList())) { model, field, sources ->
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            repeat(100) { source.frames.add(ShortArray(480) { 1000 }) }
            awaitCondition { source.frames.isEmpty() && model.amplitude.value > 0f }
            model.onRecordStop()
            assertThat(model.uiState.value).isInstanceOf(KeyboardUiState.Transcribing::class.java)
            awaitCondition { model.uiState.value !is KeyboardUiState.Transcribing }
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.NoSpeech)
            assertThat(field.fieldText).isEmpty()
        }
    }

    @Test
    fun `empty audio reports no speech and retries get their own feedback timeout`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList())) { model, _, sources ->
            repeat(2) {
                model.onRecordStart()
                awaitCondition { sources.isNotEmpty() }
                sources.remove()
                model.onRecordStop()
                awaitCondition { model.uiState.value !is KeyboardUiState.Transcribing }
                assertThat(model.uiState.value).isEqualTo(KeyboardUiState.NoSpeech)
                advanceTimeBy(1500)
                runCurrent()
                assertThat(model.uiState.value).isEqualTo(KeyboardUiState.NoSpeech)
            }
            advanceTimeBy(500)
            runCurrent()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Idle)
        }
    }

    @Test
    fun `disabled keyboard automation neither records nor switches`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList())) { model, _, sources ->
            var returns = 0
            model.onReturnToPreviousKeyboard = { returns++ }
            model.onKeyboardShown()
            runCurrent()
            model.onKeyboardHidden()
            runCurrent()
            assertThat(model.isRecording.value).isFalse()
            assertThat(sources).isEmpty()
            assertThat(returns).isZero()
        }
    }

    @Test
    fun `auto recording starts once per opening and locks hold mode for tap to stop`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList()), configure = {
            it.setAutoStartRecording(true)
            it.setTriggerMode("HOLD")
        }) { model, _, sources ->
            repeat(2) {
                model.onKeyboardShown()
                awaitCondition { sources.isNotEmpty() }
                sources.remove()
                assertThat(model.isRecording.value).isTrue()
                assertThat(model.isContinuousMode.value).isTrue()
                model.onRecordStop()
                awaitCondition { model.uiState.value == KeyboardUiState.NoSpeech }
                model.onKeyboardShown()
                runCurrent()
                assertThat(model.isRecording.value).isFalse()
                assertThat(sources).isEmpty()
                model.onKeyboardHidden()
                runCurrent()
            }
        }
    }

    @Test
    fun `automatic and manual tap toggle recordings use the same button mode`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList()), configure = {
            it.setAutoStartRecording(true)
        }) { model, _, sources ->
            model.onKeyboardShown()
            awaitCondition { sources.isNotEmpty() }
            sources.remove()
            assertThat(model.isRecording.value).isTrue()
            assertThat(model.isContinuousMode.value).isFalse()
            model.onRecordStop()
            awaitCondition { model.uiState.value == KeyboardUiState.NoSpeech }
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            assertThat(model.isRecording.value).isTrue()
            assertThat(model.isContinuousMode.value).isFalse()
        }
    }

    @Test
    fun `recording while the model loads is transcribed once it is ready`() = runTest {
        withKeyboard(FakeStreamingParakeet(listOf(0 to "Hello", 8000 to "world.")), EngineState.Loading) { model, field, sources ->
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.EngineLoading(KeyboardUiState.LoadingReason.EngineStarting))
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            repeat(100) { source.frames.add(ShortArray(480) { 1000 }) }
            awaitCondition { source.frames.isEmpty() }
            awaitCondition { model.uiState.value == KeyboardUiState.Listening(modelLoading = true) }
            model.onRecordStop()
            assertThat(model.uiState.value).isEqualTo(KeyboardUiState.Transcribing("", modelLoading = true))
            assertThat(field.fieldText).isEmpty()
            model.setEngineState(EngineState.Ready)
            awaitCondition { model.uiState.value == KeyboardUiState.Idle }
            assertThat(field.fieldText).isEqualTo("Hello world.")
        }
    }

    // Automatic recording no longer waits for the model, only for the service binding.
    @Test
    fun `automatic recording starts while the model loads, but not before the engine connects`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList()), EngineState.Unloaded, configure = {
            it.setAutoStartRecording(true)
        }) { model, _, sources ->
            model.onKeyboardShown()
            runCurrent()
            model.onKeyboardHidden()
            model.setEngineState(EngineState.Loading)
            runCurrent()
            assertThat(model.isRecording.value).isFalse()
            assertThat(sources).isEmpty()
            model.onKeyboardShown()
            awaitCondition { sources.isNotEmpty() }
            assertThat(model.isRecording.value).isTrue()
        }
    }

    @Test
    fun `quick reopen waits for finalization then starts a fresh recording`() = runTest {
        withKeyboard(FakeStreamingParakeet(listOf(0 to "Hello", 8000 to "world.")), configure = {
            it.setAutoStartRecording(true)
            it.setReturnToPreviousKeyboard(true)
        }) { model, field, sources ->
            var returns = 0
            model.onReturnToPreviousKeyboard = { returns++ }
            model.onKeyboardShown()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            repeat(100) { source.frames.add(ShortArray(480) { 1000 }) }
            awaitCondition { source.frames.isEmpty() }
            model.onKeyboardHidden()
            model.onKeyboardShown()
            awaitCondition { sources.isNotEmpty() }
            assertThat(field.fieldText).isEqualTo("Hello world.")
            assertThat(model.isRecording.value).isTrue()
            assertThat(returns).isZero()
        }
    }

    @Test
    fun `keyboard returns only after hidden recording commits its final words`() = runTest {
        withKeyboard(FakeStreamingParakeet(listOf(0 to "Hello", 8000 to "world.")), configure = {
            it.setReturnToPreviousKeyboard(true)
        }) { model, field, sources ->
            var textAtReturn: String? = null
            model.onReturnToPreviousKeyboard = { textAtReturn = field.fieldText }
            model.onKeyboardShown()
            model.onRecordStart()
            awaitCondition { sources.isNotEmpty() }
            val source = sources.remove()
            repeat(100) { source.frames.add(ShortArray(480) { 1000 }) }
            awaitCondition { source.frames.isEmpty() }
            model.onKeyboardHidden()
            awaitCondition { textAtReturn != null }
            assertThat(textAtReturn).isEqualTo("Hello world.")
        }
    }

    @Test
    fun `reopening cancels pending return to previous keyboard`() = runTest {
        withKeyboard(FakeStreamingParakeet(emptyList()), configure = {
            it.setReturnToPreviousKeyboard(true)
        }) { model, _, _ ->
            var returns = 0
            model.onReturnToPreviousKeyboard = { returns++ }
            model.onKeyboardShown()
            model.onKeyboardHidden()
            model.onKeyboardShown()
            runCurrent()
            assertThat(returns).isZero()
            model.onKeyboardHidden()
            awaitCondition { returns == 1 }
        }
    }

    private fun TestScope.withKeyboard(
        engine: SpeechEngine,
        initialEngineState: EngineState = EngineState.Ready,
        configure: suspend (AppPreferences) -> Unit = {},
        test: (KeyboardViewModel, FakeInputConnection, LinkedBlockingQueue<MicrophoneSource>) -> Unit,
    ) {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val context = RuntimeEnvironment.getApplication()
        shadowOf(context).grantPermissions(Manifest.permission.RECORD_AUDIO)
        val preferences = AppPreferences(context)
        runBlocking {
            preferences.setVadSensitivity(false)
            preferences.setTriggerMode("TAP_TOGGLE")
            preferences.setAutoStartRecording(false)
            preferences.setReturnToPreviousKeyboard(false)
            configure(preferences)
        }
        val sources = LinkedBlockingQueue<MicrophoneSource>()
        val sourceByRecorder = ConcurrentHashMap<AudioRecord, MicrophoneSource>()
        ShadowAudioRecord.setSourceProvider { recorder ->
            sourceByRecorder.computeIfAbsent(recorder) {
                MicrophoneSource().also { sources.add(it) }
            }
        }
        val suggestions = WordSuggestionProvider(context)
        val sessionScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
        val model = KeyboardViewModel(AudioCaptureManager(context), preferences, suggestions, rescue, sessionScope)
        store = ViewModelStore().apply { put("keyboard", model) }
        val field = FakeInputConnection()
        model.setTextInjector(TextInjector(field, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }))
        model.setInferenceRepository(InferenceRepository(engine))
        model.setEngineState(initialEngineState)
        try {
            awaitCondition { !model.vadSensitivity.value }
            test(model, field, sources)
        } finally {
            store.clear()
            runCurrent()
            sessionScope.cancel()
            runCurrent()
            suggestions.close()
            ShadowAudioRecord.clearSource()
            Dispatchers.resetMain()
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

    private class RecordingRescue : DictationRescue {
        val saved = mutableListOf<String>()
        var lost = 0
        override fun save(text: String) { saved += text }
        override fun reportLost() { lost++ }
    }

    private class MicrophoneSource : ShadowAudioRecord.AudioRecordSource {
        val frames = LinkedBlockingQueue<ShortArray>()

        override fun readInShortArray(audioData: ShortArray, offset: Int, size: Int, isBlocking: Boolean): Int {
            val frame = if (isBlocking) frames.poll(20, TimeUnit.MILLISECONDS) else frames.poll()
            if (frame == null) return 0
            frame.copyInto(audioData, offset)
            return frame.size
        }
    }
}

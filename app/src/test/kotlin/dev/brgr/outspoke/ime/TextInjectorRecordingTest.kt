package dev.brgr.outspoke.ime

import android.text.InputType
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import kotlinx.coroutines.test.runTest
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

// Robolectric provides real KeyEvent objects for the Enter fallback.
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], manifest = Config.NONE)
class TextInjectorRecordingTest {
    private fun injector(field: FakeInputConnection) = TextInjector(field,
        EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT })

    @Test
    fun `the recording and both separators use one commit`() {
        val field = FakeInputConnection()
        field.commitText("Before", 1)
        assertThat(injector(field).insertTranscript("Hello world.", addTrailingSpace = true)).isTrue()
        assertThat(field.fieldText).isEqualTo("Before Hello world. ")
        assertThat(field.commitCalls).containsExactly("Before", " Hello world. ")
        assertThat(field.composingCalls).isZero()
    }

    @Test
    fun `a recording replaces the selection without adding a separator inside it`() {
        val field = FakeInputConnection()
        field.commitText("Before old after", 1)
        field.setSelection(7, 10)
        assertThat(injector(field).insertTranscript("new", addTrailingSpace = false)).isTrue()
        assertThat(field.fieldText).isEqualTo("Before new after")
        assertThat(field.commitCalls).containsExactly("Before old after", "new")
    }

    @Test
    fun `blank recordings do not modify the editor`() {
        val field = FakeInputConnection()
        assertThat(injector(field).insertTranscript(" ", addTrailingSpace = true)).isFalse()
        assertThat(field.commitCalls).isEmpty()
    }

    @Test
    fun `send performs a declared send action even in a multi-line field`() = runTest {
        val field = FakeInputConnection()
        val chat = TextInjector(field, EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            imeOptions = EditorInfo.IME_ACTION_SEND or EditorInfo.IME_FLAG_NO_EXTRACT_UI
        })
        assertThat(chat.enterAction).isEqualTo(EnterAction.NEWLINE)
        assertThat(chat.send()).isEqualTo(SendOutcome.EDITOR_ACTION)
        assertThat(field.submissions.map { it.editorAction }).containsExactly(EditorInfo.IME_ACTION_SEND)
        assertThat(field.commitCalls).isEmpty()
    }

    @Test
    fun `send asks a chat box for its send action instead of the Done Android filled in`() = runTest {
        // Telegram's message box: multi-line with Android's default Done, sends on IME_ACTION_SEND.
        val field = chatBox("Hello").apply {
            onEditorAction = { if (it == EditorInfo.IME_ACTION_SEND) clearField() }
        }
        assertThat(injector(field, EditorInfo.IME_ACTION_DONE or EditorInfo.IME_FLAG_NO_ENTER_ACTION).send())
            .isEqualTo(SendOutcome.EDITOR_ACTION)
        assertThat(field.submissions.map { it.editorAction }).containsExactly(EditorInfo.IME_ACTION_SEND)
        assertThat(field.fieldText).isEmpty()
    }

    @Test
    fun `send presses Enter when the app ignores the send action but sends on Enter`() = runTest {
        // e.g. WhatsApp with "Enter is send" turned on.
        val field = chatBox("Hello").apply {
            onKeyEvent = { if (it.keyCode == KeyEvent.KEYCODE_ENTER && it.action == KeyEvent.ACTION_DOWN) clearField() }
        }
        assertThat(injector(field, EditorInfo.IME_ACTION_DONE or EditorInfo.IME_FLAG_NO_ENTER_ACTION).send())
            .isEqualTo(SendOutcome.ENTER_KEY)
        assertThat(field.submissions.map { it.editorAction ?: it.keyEvent?.keyCode }).containsExactly(
            EditorInfo.IME_ACTION_SEND, KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_ENTER,
        )
        assertThat(field.submissions.mapNotNull { it.keyEvent }).allMatch { it.flags and KeyEvent.FLAG_SOFT_KEYBOARD != 0 }
        assertThat(field.fieldText).isEmpty()
    }

    @Test
    fun `send leaves the message unchanged when Enter only adds a line break`() = runTest {
        val field = chatBox("Hello").apply {
            onKeyEvent = { if (it.keyCode == KeyEvent.KEYCODE_ENTER && it.action == KeyEvent.ACTION_DOWN) typeNewline() }
        }
        assertThat(injector(field, EditorInfo.IME_ACTION_DONE or EditorInfo.IME_FLAG_NO_ENTER_ACTION).send())
            .isEqualTo(SendOutcome.NOT_SENT)
        assertThat(field.fieldText).isEqualTo("Hello")
    }

    @Test
    fun `send trusts a declared search action without pressing Enter`() = runTest {
        // A search box keeps its query, so an empty field is not the signal here.
        val field = chatBox("weather")
        val search = TextInjector(field, EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT
            imeOptions = EditorInfo.IME_ACTION_SEARCH
        })
        assertThat(search.send()).isEqualTo(SendOutcome.EDITOR_ACTION)
        assertThat(field.submissions.map { it.editorAction }).containsExactly(EditorInfo.IME_ACTION_SEARCH)
    }

    private fun chatBox(text: String) = FakeInputConnection().apply { commitText(text, 1) }

    private fun injector(field: FakeInputConnection, imeOptions: Int) = TextInjector(field, EditorInfo().apply {
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        this.imeOptions = imeOptions
    })
}

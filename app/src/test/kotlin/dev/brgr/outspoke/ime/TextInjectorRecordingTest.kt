package dev.brgr.outspoke.ime

import android.text.InputType
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
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
    fun `send performs the declared action even in a multi-line field`() {
        val field = FakeInputConnection()
        val chat = TextInjector(field, EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            imeOptions = EditorInfo.IME_ACTION_SEND or EditorInfo.IME_FLAG_NO_EXTRACT_UI
        })
        assertThat(chat.enterAction).isEqualTo(EnterAction.NEWLINE)
        assertThat(chat.send()).isTrue()
        assertThat(field.submissions.map { it.editorAction }).containsExactly(EditorInfo.IME_ACTION_SEND)
        assertThat(field.commitCalls).isEmpty()
    }

    @Test
    fun `send presses Enter in an editor without a declared action`() {
        val field = FakeInputConnection()
        val chat = TextInjector(field, EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            imeOptions = EditorInfo.IME_FLAG_NO_EXTRACT_UI
        })
        assertThat(chat.send()).isTrue()
        val keys = field.submissions.map { requireNotNull(it.keyEvent) }
        assertThat(keys.map { it.keyCode to it.action })
            .containsExactly(KeyEvent.KEYCODE_ENTER to KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER to KeyEvent.ACTION_UP)
        assertThat(keys).allMatch { it.flags and KeyEvent.FLAG_SOFT_KEYBOARD != 0 }
        assertThat(field.commitCalls).isEmpty()
    }
}

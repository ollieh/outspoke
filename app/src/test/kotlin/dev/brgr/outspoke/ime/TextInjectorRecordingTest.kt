package dev.brgr.outspoke.ime

import android.text.InputType
import android.view.inputmethod.EditorInfo
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

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
}

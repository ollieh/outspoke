package dev.brgr.outspoke.ime

import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputConnectionWrapper
import dev.brgr.outspoke.inference.RecordingTrace

/** Debug-only editor diagnostics: lengths and acceptance, never text content. */
internal class TracingInputConnection(target: InputConnection) : InputConnectionWrapper(target, false) {
    override fun commitText(text: CharSequence?, newCursorPosition: Int): Boolean =
        super.commitText(text, newCursorPosition).also { accepted ->
            RecordingTrace.log { "editor commit chars=${text?.length ?: 0} accepted=$accepted" }
        }

    override fun setComposingText(text: CharSequence?, newCursorPosition: Int): Boolean =
        super.setComposingText(text, newCursorPosition).also { accepted ->
            RecordingTrace.log { "editor compose chars=${text?.length ?: 0} accepted=$accepted" }
        }

    override fun finishComposingText(): Boolean = super.finishComposingText().also { accepted ->
        RecordingTrace.log { "editor finish composing accepted=$accepted" }
    }
}

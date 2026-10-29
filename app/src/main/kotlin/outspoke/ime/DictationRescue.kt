package dev.brgr.outspoke.ime

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import dev.brgr.outspoke.R
import dev.brgr.outspoke.inference.RecordingTrace

/**
 * Keeps dictated speech the keyboard could not insert into its editor and tells the user.
 * Called after the editor is gone, possibly after the keyboard itself was destroyed.
 */
interface DictationRescue {
    /** Puts [text] where the user can retrieve it. */
    fun save(text: String)

    /** Speech was captured but could not be transcribed at all. */
    fun reportLost()
}

/** Copies rescued dictation to the clipboard. Uses only the application context. */
class ClipboardDictationRescue(context: Context) : DictationRescue {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun save(text: String) {
        mainHandler.post {
            val clipboard = appContext.getSystemService(ClipboardManager::class.java)
            clipboard?.setPrimaryClip(ClipData.newPlainText(appContext.getString(R.string.clip_label_dictation), text))
            RecordingTrace.log { "dictation rescued to clipboard chars=${text.length} clipboard=${clipboard != null}" }
            Toast.makeText(appContext, R.string.toast_dictation_copied, Toast.LENGTH_LONG).show()
        }
    }

    override fun reportLost() {
        mainHandler.post {
            RecordingTrace.log { "dictation lost reported" }
            Toast.makeText(appContext, R.string.toast_dictation_lost, Toast.LENGTH_LONG).show()
        }
    }
}

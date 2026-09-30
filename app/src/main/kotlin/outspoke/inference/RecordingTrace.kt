package dev.brgr.outspoke.inference

import android.util.Log
import dev.brgr.outspoke.BuildConfig
import java.util.concurrent.atomic.AtomicLong

/** Debug-build progress markers. Never pass audio, transcript text, or editor content. */
internal object RecordingTrace {
    private val sessionIds = AtomicLong()
    fun nextSessionId(): Long = sessionIds.incrementAndGet()

    inline fun log(message: () -> String) {
        if (BuildConfig.DEBUG) Log.d("OutspokeTrace", "atMs=${System.nanoTime() / 1_000_000} ${message()}")
    }
}

package dev.brgr.outspoke.inference

import dev.brgr.outspoke.settings.preferences.AppPreferences
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35], manifest = Config.NONE)
class CustomWordPreferencesTest {
    @Test
    fun `custom terms persist across preference instances and can be removed`() {
        runBlocking {
            val context = RuntimeEnvironment.getApplication()
            val preferences = AppPreferences(context)
            preferences.customWords.first().forEach { preferences.removeCustomWord(it) }
            try {
                preferences.addCustomWord("  MacBook   Pro ")
                preferences.addCustomWord("macbook pro")
                preferences.addCustomWord("!!!")
                val reopened = AppPreferences(context)
                assertThat(reopened.customWords.first()).containsExactly("MacBook Pro")
                reopened.removeCustomWord("MacBook Pro")
                assertThat(preferences.customWords.first()).isEmpty()
            } finally {
                preferences.customWords.first().forEach { preferences.removeCustomWord(it) }
            }
        }
    }
}

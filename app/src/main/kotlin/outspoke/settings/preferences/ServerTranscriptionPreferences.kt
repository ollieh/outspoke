package dev.brgr.outspoke.settings.preferences

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import dev.brgr.outspoke.remote.DEFAULT_SERVER_GRACE_MS
import dev.brgr.outspoke.remote.MAX_SERVER_GRACE_MS
import dev.brgr.outspoke.remote.ServerTranscriptionSettings
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map

/** Server transcription settings. The API key lives in [ServerApiKeyStore], not here. */
class ServerTranscriptionPreferences(private val context: Context) {
    private val keyBaseUrl = stringPreferencesKey("server_base_url")
    private val keyModel = stringPreferencesKey("server_model")
    private val keyGraceMs = longPreferencesKey("server_grace_ms")

    val settings: Flow<ServerTranscriptionSettings> = context.dataStore.data.map { prefs ->
        ServerTranscriptionSettings(
            baseUrl = prefs[keyBaseUrl].orEmpty(),
            model = prefs[keyModel].orEmpty(),
            graceMs = prefs[keyGraceMs] ?: DEFAULT_SERVER_GRACE_MS,
        )
    }.distinctUntilChanged()

    suspend fun save(settings: ServerTranscriptionSettings) {
        context.dataStore.edit { prefs ->
            prefs[keyBaseUrl] = settings.baseUrl.trim()
            prefs[keyModel] = settings.model.trim()
            prefs[keyGraceMs] = settings.graceMs.coerceIn(0, MAX_SERVER_GRACE_MS)
        }
    }
}

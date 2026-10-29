package dev.brgr.outspoke.settings.preferences

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.brgr.outspoke.remote.DEFAULT_SERVER_GRACE_MS
import dev.brgr.outspoke.remote.MAX_SERVER_GRACE_MS
import dev.brgr.outspoke.remote.ServerEndpoint
import dev.brgr.outspoke.remote.ServerTranscriptionClient
import dev.brgr.outspoke.remote.ServerTranscriptionException
import dev.brgr.outspoke.remote.ServerTranscriptionSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Edits server transcription settings; nothing is stored until [save]. */
class ServerTranscriptionViewModel(application: Application) : AndroidViewModel(application) {

    data class Form(
        val baseUrl: String = "",
        val apiKey: String = "",
        val model: String = "",
        val graceMs: String = DEFAULT_SERVER_GRACE_MS.toString(),
    ) {
        val urlValid: Boolean get() = baseUrl.isBlank() || ServerEndpoint.parse(baseUrl) != null
        val grace: Long? get() = graceMs.trim().toLongOrNull()?.takeIf { it in 0..MAX_SERVER_GRACE_MS }
        val valid: Boolean get() = urlValid && grace != null
        val cleartext: Boolean get() = ServerEndpoint.parse(baseUrl)?.v1?.isHttps == false
    }

    sealed interface TestState {
        data object Idle : TestState
        data object Running : TestState
        /** [filledModel] is set when the empty model field was filled from [models]. */
        data class Connected(val models: List<String>, val filledModel: String?) : TestState
        data class Failed(val error: ServerTranscriptionException) : TestState
    }

    private val preferences = ServerTranscriptionPreferences(application)
    private val keys = ServerApiKeyStore(application)
    private val client = ServerTranscriptionClient()

    private val _form = MutableStateFlow(Form())
    val form: StateFlow<Form> = _form.asStateFlow()

    /** Null until the stored values have loaded. */
    private val _saved = MutableStateFlow<Form?>(null)
    val saved: StateFlow<Form?> = _saved.asStateFlow()

    private val _testState = MutableStateFlow<TestState>(TestState.Idle)
    val testState: StateFlow<TestState> = _testState.asStateFlow()

    /** False when the Keystore refused to encrypt the API key on the last save. */
    private val _keyStored = MutableStateFlow(true)
    val keyStored: StateFlow<Boolean> = _keyStored.asStateFlow()

    private var testJob: Job? = null

    init {
        viewModelScope.launch {
            val stored = preferences.settings.first()
            val apiKey = withContext(Dispatchers.IO) { keys.get() }
            val loaded = Form(stored.baseUrl, apiKey, stored.model, stored.graceMs.toString())
            _form.value = loaded
            _saved.value = loaded
        }
    }

    fun update(transform: (Form) -> Form) {
        _form.update(transform)
        testJob?.cancel()
        _testState.value = TestState.Idle
    }

    fun save() {
        val form = _form.value
        val grace = form.grace ?: return
        if (!form.urlValid) return
        val normalized = form.copy(
            baseUrl = form.baseUrl.trim(), apiKey = form.apiKey.trim(),
            model = form.model.trim(), graceMs = grace.toString(),
        )
        _form.value = normalized
        viewModelScope.launch {
            preferences.save(ServerTranscriptionSettings(normalized.baseUrl, normalized.model, grace))
            _keyStored.value = withContext(Dispatchers.IO) { keys.set(normalized.apiKey) }
            _saved.value = normalized
        }
    }

    /** Checks the entered (unsaved) URL and key; fills an empty model with the first listed one. */
    fun testConnection() {
        val form = _form.value
        val endpoint = ServerEndpoint.parse(form.baseUrl) ?: run {
            _testState.value = TestState.Failed(ServerTranscriptionException(ServerTranscriptionException.Kind.INVALID_URL))
            return
        }
        testJob?.cancel()
        testJob = viewModelScope.launch {
            _testState.value = TestState.Running
            _testState.value = try {
                val models = client.listModels(endpoint, form.apiKey)
                val fill = models.first().takeIf { _form.value.model.isBlank() }
                if (fill != null) _form.update { it.copy(model = fill) }
                TestState.Connected(models, fill)
            } catch (e: CancellationException) {
                throw e
            } catch (e: ServerTranscriptionException) {
                TestState.Failed(e)
            }
        }
    }
}

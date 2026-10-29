package dev.brgr.outspoke.ui.keyboard

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import dev.brgr.outspoke.audio.AudioCaptureManager
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.ime.DictationRescue
import dev.brgr.outspoke.ime.EnterAction
import dev.brgr.outspoke.ime.TextInjector
import dev.brgr.outspoke.ime.WordSuggestionProvider
import dev.brgr.outspoke.inference.EngineState
import dev.brgr.outspoke.inference.InferenceRepository
import dev.brgr.outspoke.inference.PipelineDiagnostics
import dev.brgr.outspoke.inference.RecordingTrace
import dev.brgr.outspoke.inference.TranscriptResult
import dev.brgr.outspoke.remote.ServerRaceSession
import dev.brgr.outspoke.remote.ServerTranscriber
import dev.brgr.outspoke.remote.uploadWhenComplete
import dev.brgr.outspoke.settings.preferences.AppPreferences
import dev.brgr.outspoke.ui.keyboard.components.WHISPER_LANGUAGE_OPTIONS
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "KeyboardViewModel"

/**
 * Bridges the IME lifecycle, audio capture, and inference pipeline into a stream of
 * [KeyboardUiState] values consumed by [KeyboardScreen].
 *
 * Dictated speech is never dropped silently: only [onDiscard] throws it away. Text that
 * cannot reach the editor it was dictated for goes to [dictationRescue].
 */
class KeyboardViewModel(
    private val audioCaptureManager: AudioCaptureManager,
    private val appPreferences: AppPreferences,
    private val wordSuggestionProvider: WordSuggestionProvider,
    private val dictationRescue: DictationRescue,
    /** Runs dictations. Outlives [viewModelScope] so a dictation can finish after the IME is gone. */
    private val sessionScope: CoroutineScope,
    /** Optional server race; inert unless a server URL is configured. */
    private val serverTranscriber: ServerTranscriber? = null,
) : ViewModel() {

    private val _uiState = MutableStateFlow<KeyboardUiState>(
        KeyboardUiState.EngineLoading(KeyboardUiState.LoadingReason.EngineStarting)
    )
    val uiState: StateFlow<KeyboardUiState> = _uiState.asStateFlow()

    /** Normalised RMS amplitude [0.0, 1.0] - updated by [AudioCaptureManager] per chunk. */
    val amplitude: StateFlow<Float> = audioCaptureManager.amplitude

    val trailingSpace = appPreferences.trailingSpace
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    fun insertSpace() { textInjector?.insertSpace() }

    /**
     * `"HOLD"` (default) or `"TAP_TOGGLE"`.
     * Collected eagerly so the TalkButton always has the latest value without
     * needing a suspend context.
     */
    val triggerMode: StateFlow<String> = appPreferences.triggerMode
        .stateIn(viewModelScope, SharingStarted.Eagerly, "HOLD")

    /**
     * What the delete (trash) button does: `"DELETE_ALL"` (default) or
     * `"DELETE_LAST_SENTENCE"`. Collected eagerly so [deleteAll] can read a
     * synchronous snapshot without a suspend context.
     */
    val deleteButtonMode: StateFlow<String> = appPreferences.deleteButtonMode
        .stateIn(viewModelScope, SharingStarted.Eagerly, "DELETE_ALL")

    /**
     * `true` when the user opted into raw (unprocessed) microphone capture,
     * which bypasses the platform's echo cancellation - needed when transcribing
     * audio played from the device's own speaker.
     * Collected eagerly so [onRecordStart] can read a synchronous snapshot.
     */
    val rawMicCapture: StateFlow<Boolean> = appPreferences.rawMicCapture
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /**
     * Whether VAD (voice activity detection) is enabled. Collected eagerly so the value is
     * always available as a snapshot when recording starts - no suspend context required.
     */
    val vadSensitivity: StateFlow<Boolean> = appPreferences.vadSensitivity
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)

    /**
     * `true` when the currently selected model is a Whisper variant.
     * Used to conditionally show the language selector in the keyboard UI.
     */
    val isWhisperEngine: StateFlow<Boolean> = appPreferences.selectedModelId
        .map { it.name.startsWith("WHISPER") }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /**
     * Currently selected Whisper language tag: `"auto"`, `"en"`, `"de"`, or `"nl"`.
     * Collected eagerly so the selector always reflects the saved value on first draw.
     */
    val whisperLanguage: StateFlow<String> = appPreferences.whisperLanguage
        .stateIn(viewModelScope, SharingStarted.Eagerly, "auto")

    /**
     * The post-processing language used by the cleaning pipeline (filler removal, number
     * normalisation, spurious-period handling). Mirrors [SpeechEngine.currentLanguage] for
     * the active engine: the user's forced language, or `"en"` when set to `"auto"` / unset.
     *
     * Collected eagerly so [dev.brgr.outspoke.ime.TextInjector]'s display-cleaning lambda can
     * read a synchronous snapshot when injecting text — this keeps the display path's
     * language consistent with the engine's (e.g. so German noun capitalisation is
     * preserved and German fillers are removed in the displayed text, not just in the
     * stable-chunk tracking path).
     */
    val currentLanguage: StateFlow<String> = appPreferences.forcedLanguage
        .map { it ?: "en" }
        .stateIn(viewModelScope, SharingStarted.Eagerly, "en")

    /**
     * `true` when the transcript post-processing pipeline (filler removal, stutter collapse,
     * repetition deduplication, capitalisation) is active.  Defaults to `true`.
     * Collected eagerly so the live value is always available when recording starts.
     */
    val postprocessingEnabled: StateFlow<Boolean> = appPreferences.postprocessingEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)

    val customWords = appPreferences.customWords
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    private val serverSettings = serverTranscriber?.settings
        ?.stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** Streaming results have already been cleaned and custom spellings must stay intact. */
    val transcriptAlreadyCleaned: Boolean
        get() = _repository.value?.producesCleanTranscripts == true

    /**
     * Pipeline diagnostic counters for the current or most recent recording session.
     *
     * Updated live during recording and kept visible after the session ends so the user
     * can glance at the keyboard status bar to see if anything unusual happened (trims,
     * alignment recoveries, blanks discarded) without needing logcat.
     *
     * Reset at the start of each new recording in [onRecordStart].
     */
    private val _diagnostics = MutableStateFlow(PipelineDiagnostics())
    val diagnostics: StateFlow<PipelineDiagnostics> = _diagnostics.asStateFlow()

    /**
     * Whether the pipeline diagnostics badge is visible on the keyboard.
     * Collected eagerly so the UI always reflects the saved preference on first draw.
     */
    val showPipelineDiagnostics: StateFlow<Boolean> = appPreferences.showPipelineDiagnostics
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /**
     * `true` when the word-suggestion bar feature is enabled in preferences.
     * When `false` the bar is never shown and no dictionary work is performed.
     */
    val suggestionBarFeatureEnabled: StateFlow<Boolean> = appPreferences.suggestionBarEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /**
     * The word currently under the text cursor, or `null` when the cursor is not inside
     * a word.  Updated by [updateWordAtCursor] on every cursor movement and after each
     * committed final transcript.  Consumed by the suggestion bar.
     */
    private val _wordAtCursor = MutableStateFlow<WordAtCursor?>(null)
    val wordAtCursor: StateFlow<WordAtCursor?> = _wordAtCursor.asStateFlow()

    /**
     * Spelling suggestions for the word currently under the cursor, delivered
     * asynchronously by Android's [WordSuggestionProvider] (SpellCheckerSession).
     *
     * Empty when the cursor is between words, the suggestion bar is dismissed, or the
     * system spell-checker returns no alternatives (word is correctly spelled).
     *
     * Reset to empty at the start of each new recording session via [onRecordStart].
     */
    private val _wordSuggestions = MutableStateFlow<List<String>>(emptyList())
    val wordSuggestions: StateFlow<List<String>> = _wordSuggestions.asStateFlow()

    /**
     * `true` after the user has tapped the dismiss (×) button in the suggestion bar.
     * Suppresses further suggestions until the cursor moves to a new word.
     * In-memory only — intentionally not persisted to DataStore.
     */
    private val _suggestionBarDismissed = MutableStateFlow(false)
    val suggestionBarDismissed: StateFlow<Boolean> = _suggestionBarDismissed.asStateFlow()

    init {
        // Wire up the spell-checker callback.
        // dismissed state is in-memory only — never read from DataStore.
        wordSuggestionProvider.onSuggestions = { suggestions ->
            // Ignore suggestions that arrive after the bar was dismissed, or while
            // recording / transcription is active — showing them mid-session confuses
            // users who see suggestions flash and disappear as new words come in.
            if (!_suggestionBarDismissed.value && !isActiveSession()) {
                _wordSuggestions.value = suggestions
            }
        }

        // Propagate the user's active-language selection to the provider whenever it changes.
        viewModelScope.launch {
            appPreferences.suggestionBarLanguages.collect { tags ->
                wordSuggestionProvider.setActiveLanguages(tags)
            }
        }
    }

    /**
     * Reads the word at the current cursor position from [textInjector] and publishes it
     * to [wordAtCursor].  When a word is found and the bar is not dismissed, also fires an
     * asynchronous spell-checker query whose result updates [wordSuggestions].
     *
     * Safe to call from the main thread — [TextInjector.wordAtCursor] performs only
     * lightweight [android.view.inputmethod.InputConnection] calls.
     *
     * Called on every [onUpdateSelection] and after each [TranscriptResult.Final] commit.
     */
    fun updateWordAtCursor() {
        // Do nothing when the feature is disabled — avoids even reading the cursor word.
        if (!suggestionBarFeatureEnabled.value) return
        // Do nothing during an active recording or transcription session — suggestions
        // would flash briefly and disappear as each new word commits, which is confusing.
        if (isActiveSession()) return
        val wac = textInjector?.wordAtCursor()
        _wordAtCursor.value = wac
        if (wac != null && !_suggestionBarDismissed.value) {
            Log.d(TAG, "updateWordAtCursor (${wac.word.length} chars) — querying corrector")
            wordSuggestionProvider.getSuggestions(wac.word, wac.sentenceContext)
        } else {
            Log.d(TAG, "updateWordAtCursor → no cursor word (dismissed=${_suggestionBarDismissed.value})")
            _wordSuggestions.value = emptyList()
        }
    }

    /**
     * Returns true when the keyboard is actively recording or processing a transcription.
     * The suggestion bar must not be shown in these states.
     */
    private fun isActiveSession(): Boolean = _uiState.value.isDictating

    /**
     * Clears word suggestions so the bar vanishes, without setting a persistent dismissed
     * state. Behaves identically to clicking at the end of the text field — the bar
     * reappears the next time the user taps on a word.
     */
    fun dismissSuggestionBar() {
        _wordSuggestions.value = emptyList()
    }

    /**
     * Replaces the word currently under the text cursor with [word].
     *
     * Delegates to [TextInjector.replaceCursorWord] and then refreshes [wordAtCursor] so
     * the suggestion bar reflects the new cursor state immediately.  Safe to call on the
     * main thread.
     *
     * Called by the suggestion bar when the user taps a chip.
     */
    fun replaceWordAtCursor(word: String) {
        textInjector?.replaceCursorWord(word)
        updateWordAtCursor()
    }

    /**
     * When `true` (default), number-word sequences in the transcript are converted to
     * digit form by [dev.brgr.outspoke.inference.NumberNormaliser] as part of the
     * post-processing pipeline.
     */
    val formatNumbersAsDigits: StateFlow<Boolean> = appPreferences.formatNumbersAsDigits
        .stateIn(viewModelScope, SharingStarted.Eagerly, true)

    /**
     * Persists [tag] to preferences and immediately forwards it to the loaded engine.
     * Safe to call at any time - the engine's [setLanguage] is thread-safe.
     */
    fun setWhisperLanguage(tag: String) {
        _repository.value?.setLanguage(tag)
        viewModelScope.launch { appPreferences.setWhisperLanguage(tag) }
    }

    /**
     * Whether the first-run keyboard tutorial should currently be visible.
     * Resolves to `true` on the very first keyboard opening and `false` permanently
     * after [dismissTutorial] is called.
     */
    val showTutorial: StateFlow<Boolean> = appPreferences.keyboardTutorialShown
        .map { shown -> !shown }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /** Persists the tutorial as seen so it is never shown again. */
    fun dismissTutorial() {
        viewModelScope.launch { appPreferences.setKeyboardTutorialShown(true) }
    }

    private val _engineState = MutableStateFlow<EngineState>(EngineState.Unloaded)

    /** Called by [OutspokeInputMethodService] whenever [InferenceService.engineState] changes. */
    fun setEngineState(state: EngineState) {
        _engineState.value = state
        // A dictation waits for the model itself and owns the status line until it ends.
        if (session != null) return
        showEngineState(state)
    }

    private fun showEngineState(state: EngineState) {
        when (state) {
            EngineState.Unloaded -> _uiState.value = KeyboardUiState.EngineLoading(
                KeyboardUiState.LoadingReason.ModelNotDownloaded
            )

            EngineState.Loading -> _uiState.value = KeyboardUiState.EngineLoading(
                KeyboardUiState.LoadingReason.EngineStarting
            )

            EngineState.Ready -> {
                // Clear any engine-driven blocking state so the user can start recording.
                if (_uiState.value is KeyboardUiState.EngineLoading ||
                    (_uiState.value is KeyboardUiState.Error &&
                            (_uiState.value as KeyboardUiState.Error).reason == KeyboardUiState.ErrorReason.EngineLoadFailed)
                ) {
                    _uiState.value = KeyboardUiState.Idle
                }
            }

            is EngineState.Error -> _uiState.value = KeyboardUiState.Error(
                reason = KeyboardUiState.ErrorReason.EngineLoadFailed,
                detail = state.message,
            )
        }
    }

    private val _repository = MutableStateFlow<InferenceRepository?>(null)

    /** Set by [OutspokeInputMethodService] when the service binding is established. */
    fun setInferenceRepository(repo: InferenceRepository?) {
        _repository.value = repo
        // Apply the persisted language tag immediately.
        repo?.setLanguage(whisperLanguage.value)
        // Constrain auto-detection to only the languages shown in the selector.
        // "Auto" then means "detect from EN/DE/ES" instead of "any of ~100 languages",
        // which eliminates false picks like ES (50262) beating EN (50259) by a tiny margin.
        repo?.setLanguageConstraints(
            WHISPER_LANGUAGE_OPTIONS.filter { (tag, _) -> tag != "auto" }.map { (tag, _) -> tag }
        )
    }

    private var textInjector: TextInjector? = null

    private val _enterAction = MutableStateFlow(EnterAction.DONE)

    /**
     * The context-aware action the Enter key should perform for the currently focused editor.
     * Updated each time [setTextInjector] is called (i.e. on every [onStartInput]).
     */
    val enterAction: StateFlow<EnterAction> = _enterAction.asStateFlow()

    /**
     * Attaches the editor of a new input session. A dictation in progress follows the
     * editor only when [sameEditor] says the app restarted input on the same field;
     * any other change detaches it so its text can never land in a different field.
     */
    fun setTextInjector(injector: TextInjector?, sameEditor: Boolean = false) {
        RecordingTrace.log { "editor attached=${injector != null} sameEditor=$sameEditor dictation=${session?.id}" }
        val s = session
        if (s != null && injector !== s.injector) {
            if (sameEditor && injector != null) {
                s.injector = injector
            } else {
                detach(s, if (injector == null) "editor closed" else "editor changed")
            }
        }
        textInjector = injector
        _enterAction.value = injector?.enterAction ?: EnterAction.DONE
    }

    private val _isContinuousMode = MutableStateFlow(false)

    /**
     * `true` while the keyboard is in locked continuous-recording mode (engaged by
     * dragging the talk button upward).  Resets to `false` whenever recording stops.
     */
    val isContinuousMode: StateFlow<Boolean> = _isContinuousMode.asStateFlow()

    /**
     * Called by the UI when the user drags the talk button past the lock threshold.
     * Recording continues uninterrupted - only the visual state changes.
     */
    fun onContinuousModeEnabled() {
        _isContinuousMode.value = true
    }

    /**
     * Called when the user taps "Retry" after a transient error (e.g. low-confidence failure).
     *
     * Clears the error and returns to [KeyboardUiState.Idle] so the user starts a fresh
     * recording with a normal press (HOLD) or a drag-up-to-lock — exactly like the initial
     * recording.
     *
     * We intentionally do NOT auto-start recording and do NOT engage continuous mode. The
     * previous behaviour forced continuous mode and immediately started capturing, which
     * made the button jump straight into locked recording with no finger on it — confusing
     * and not what "retry" implies. "Retry" means "let me try again", i.e. dismiss the
     * error and give me back the idle button; the user then chooses how to record.
     */
    fun onRetry() {
        // A failed dictation may still be draining its flow. Its preview was already
        // rescued when it failed, so it cannot bleed into the next one.
        session?.let { onDiscard() }
        _isRecording.value = false
        _isContinuousMode.value = false
        _uiState.value = KeyboardUiState.Idle
    }

    /** Delete the character immediately before the cursor. */
    fun deleteChar() {
        textInjector?.deleteChar()
    }

    /** Delete backward to the previous word boundary. */
    fun deleteWord() {
        textInjector?.deleteWord()
    }

    /**
     * Performs the configured delete action: clears the whole editor field
     * ([deleteButtonMode] == `"DELETE_ALL"`, the default) or removes only the
     * last sentence before the cursor (`"DELETE_LAST_SENTENCE"`).
     */
    fun deleteAll() {
        if (deleteButtonMode.value == "DELETE_LAST_SENTENCE") {
            textInjector?.deleteLastSentence()
        } else {
            textInjector?.deleteAll()
        }
    }

    /** Insert a newline at the current cursor position, replacing any active selection. */
    fun newline() {
        textInjector?.sendNewline()
    }

    /**
     * Perform the context-aware Enter action for the currently focused editor.
     * Inserts a newline for multi-line fields; otherwise triggers the editor's IME action
     * (search, send, go, done, next) via [InputConnection.performEditorAction].
     * During a dictation Enter means [onSend]: finish, insert, then send.
     */
    fun performEnterAction() {
        if (session != null) onSend() else textInjector?.performEnterAction()
    }

    /**
     * One dictation, from the record request until its text is inserted, rescued or
     * discarded. Owned by [sessionScope], so it can finish after the keyboard is gone.
     */
    private class Session(val id: Long, var injector: TextInjector?) {
        val transcript = RecordingTranscript()

        /** Captured audio, buffered until the speech model is ready to consume it. */
        val audio = Channel<AudioChunk>(Channel.UNLIMITED)

        /** Set on the capture thread when the microphone first delivers samples. */
        val audioStarted = AtomicBoolean(false)
        var job: Job? = null

        /** Uploads the recording to the user's server and races it against the phone. */
        var serverRace: ServerRaceSession? = null
        var recording = true
        var micLive = false
        var modelReady = false
        var failed = false

        /** The input session ended; the text goes to [DictationRescue], never to a later editor. */
        var orphaned = false
        var send = false
        var returnWhenDone = false
        var switchKeyboard: (() -> Unit)? = null
    }

    /** The dictation shown by the keyboard. Detached dictations finish in [sessions] only. */
    private var session: Session? = null
    private val sessions = mutableListOf<Session>()
    private var noSpeechJob: Job? = null
    private var keyboardVisible = false
    private var autoStartJob: Job? = null
    private var returnKeyboardJob: Job? = null

    /** The IME checks that it is still selected before switching keyboards. */
    var onReturnToPreviousKeyboard: (() -> Unit)? = null

    fun onKeyboardShown() {
        RecordingTrace.log { "keyboard shown alreadyVisible=$keyboardVisible" }
        if (keyboardVisible) return
        keyboardVisible = true
        returnKeyboardJob?.cancel()
        autoStartJob = viewModelScope.launch {
            if (!appPreferences.autoStartRecording.first()) return@launch
            // Recording starts while the model loads; its audio is buffered until it is ready.
            _engineState.first { it == EngineState.Loading || it == EngineState.Ready }
            // A quick reopen can arrive while the previous recording is finalizing.
            session?.job?.join()
            val holdToRecord = appPreferences.triggerMode.first() == "HOLD"
            autoStartJob = null
            if (!keyboardVisible || session != null) return@launch
            onRecordStart()
            // Automatic capture has no held gesture. Allow tap-to-stop in HOLD mode too.
            if (_isRecording.value && holdToRecord) onContinuousModeEnabled()
        }
    }

    fun onKeyboardHidden() {
        RecordingTrace.log { "keyboard hidden wasVisible=$keyboardVisible" }
        if (!keyboardVisible) return
        keyboardVisible = false
        autoStartJob?.cancel()
        val finishing = session
        onRecordStop()
        returnKeyboardJob = viewModelScope.launch {
            if (!appPreferences.returnToPreviousKeyboard.first()) return@launch
            finishing?.job?.join()
            if (!keyboardVisible) onReturnToPreviousKeyboard?.invoke()
        }
    }
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    /**
     * Set by [dev.brgr.outspoke.ime.OutspokeInputMethodService] to a lambda that requests
     * the inference service to reload its (still-present) model and re-establish the
     * repository binding.
     *
     * Invoked from [onRecordStart] when the engine reports [EngineState.Ready] but no
     * [InferenceRepository] is currently bound — a binder desync. The recording still
     * starts and waits for the repository, so the callback lets the IME rebind / reload.
     */
    var onMissingRepo: (() -> Unit)? = null

    /**
     * Start microphone capture immediately and transcribe the audio once the speech model
     * is ready. Ignored while another dictation is in progress, and when there is no
     * model to load ([EngineState.Unloaded] or [EngineState.Error]).
     */
    fun onRecordStart() {
        RecordingTrace.log { "record requested engine=${_engineState.value.javaClass.simpleName} repo=${_repository.value != null} active=${session?.id} injector=${textInjector != null}" }
        autoStartJob?.cancel()
        if (session != null) {
            Log.w(TAG, "onRecordStart() ignored - a dictation is already in progress")
            return
        }
        val engine = _engineState.value
        if (engine == EngineState.Unloaded || engine is EngineState.Error) {
            Log.w(TAG, "onRecordStart() ignored - no speech model to load ($engine)")
            return
        }
        if (engine == EngineState.Ready && _repository.value == null) {
            Log.w(TAG, "Engine Ready but no InferenceRepository bound (desync); requesting reload")
            onMissingRepo?.invoke()
        }

        noSpeechJob?.cancel()
        val s = Session(RecordingTrace.nextSessionId(), textInjector)
        session = s
        sessions += s
        _isRecording.value = true
        _diagnostics.value = PipelineDiagnostics()
        _wordSuggestions.value = emptyList()
        // Dismissed flag is no longer set persistently — bar reappears on next word tap.
        publish(s)
        s.job = sessionScope.launch { runSession(s) }
    }

    private class EngineUnavailable(message: String) : Exception(message)

    private suspend fun runSession(s: Session) {
        var results = 0
        var inserted = false
        val addTrailingSpace = trailingSpace.value
        RecordingTrace.log { "session start id=${s.id}" }
        // The race's upload is cancelled when this dictation's coroutine completes.
        s.serverRace = serverTranscriber?.startSession(
            owner = CoroutineScope(currentCoroutineContext()), settings = serverSettings?.value,
            recordingId = s.id, language = currentLanguage.value, customWords = customWords.value,
            formatNumbersAsDigits = postprocessingEnabled.value && formatNumbersAsDigits.value,
        )
        try {
            coroutineScope {
                launch { captureAudio(s) }
                val repo = awaitRepository(s)
                s.modelReady = true
                publish(s)
                RecordingTrace.log { "session model ready id=${s.id} recording=${s.recording}" }
                // Results update the preview; completion performs the editor insertion.
                repo.transcribe(
                    audio = s.audio.consumeAsFlow(),
                    postprocessingEnabled = postprocessingEnabled.value,
                    formatNumbersAsDigits = formatNumbersAsDigits.value,
                    customWords = customWords.value,
                    recordingId = s.id,
                ).collect { result ->
                    results++
                    RecordingTrace.log {
                        val chars = when (result) {
                            is TranscriptResult.Partial -> result.text.length
                            is TranscriptResult.Final -> result.text.length
                            else -> 0
                        }
                        "result session=${s.id} count=$results type=${result.javaClass.simpleName} chars=$chars current=${session === s} orphaned=${s.orphaned}"
                    }
                    onResult(s, result)
                }
            }
            inserted = complete(s, addTrailingSpace)
        } catch (e: CancellationException) {
            throw e
        } catch (e: EngineUnavailable) {
            Log.e(TAG, "Speech model unavailable for dictation", e)
            fail(s, KeyboardUiState.ErrorReason.EngineLoadFailed, e.message)
        } catch (e: SecurityException) {
            Log.e(TAG, "Microphone permission denied", e)
            fail(s, KeyboardUiState.ErrorReason.MicPermissionDenied)
        } catch (e: IllegalStateException) {
            Log.e(TAG, "AudioRecord failed to initialise", e)
            fail(s, KeyboardUiState.ErrorReason.MicInitFailed, e.message)
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected audio capture error", e)
            fail(s, KeyboardUiState.ErrorReason.AudioCaptureFailed, e.message)
        } finally {
            RecordingTrace.log { "session end id=${s.id} results=$results inserted=$inserted failed=${s.failed} orphaned=${s.orphaned} current=${session === s}" }
            release(s)
        }
        // Not reached when discarded. The keyboard is already free for a new dictation.
        leave(s, inserted)
    }

    /** Feeds the microphone into [Session.audio] until Stop, or until the microphone fails. */
    private suspend fun captureAudio(s: Session) {
        // A Stop that arrives before this runs has no capture flag to set yet.
        if (!s.recording) {
            s.audio.close()
            return
        }
        try {
            audioCaptureManager.startCapture(
                vadEnabled = vadSensitivity.value,
                rawSource = rawMicCapture.value,
                recordingId = s.id,
                onAudioStarted = {
                    s.audioStarted.set(true)
                    sessionScope.launch {
                        s.micLive = true
                        publish(s)
                    }
                },
                pcmTap = s.serverRace?.pcm,
            ).uploadWhenComplete(s.serverRace).collect { s.audio.send(it) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // Without any audio there is nothing to finish: report the failure. A later
            // failure ends the recording but keeps transcribing what was captured.
            if (!s.audioStarted.get()) throw e
            Log.e(TAG, "Audio capture failed mid-recording; finishing with the captured audio", e)
        } finally {
            s.audio.close()
        }
        stop(s)
    }

    private suspend fun awaitRepository(s: Session): InferenceRepository {
        if (_engineState.value != EngineState.Ready || _repository.value == null) {
            RecordingTrace.log { "session waiting for model id=${s.id} engine=${_engineState.value.javaClass.simpleName}" }
        }
        val (repo, engine) = combine(_repository, _engineState) { repo, engine -> repo to engine }
            .first { (repo, engine) ->
                (repo != null && engine == EngineState.Ready) ||
                        engine is EngineState.Error || engine == EngineState.Unloaded
            }
        if (repo != null && engine == EngineState.Ready) return repo
        throw EngineUnavailable((engine as? EngineState.Error)?.message ?: "Speech model is not available")
    }

    private fun onResult(s: Session, result: TranscriptResult) {
        if (s.failed) return
        when (result) {
            is TranscriptResult.Partial -> {
                s.transcript.partial(result.text)
                publish(s)
            }

            is TranscriptResult.Final -> {
                s.transcript.final(result.text)
                publish(s)
            }

            is TranscriptResult.Failure -> {
                Log.e(TAG, "Transcription failure", result.cause)
                fail(s, KeyboardUiState.ErrorReason.TranscriptionFailed, result.cause.message)
            }

            is TranscriptResult.WindowTrimmed -> {
                s.transcript.windowTrimmed()
                if (session === s) {
                    _diagnostics.value = _diagnostics.value.copy(
                        windowTrims = _diagnostics.value.windowTrims + 1
                    )
                }
            }

            // The model saw audio but couldn't resolve a word (weak
            // high-frequency fricatives). Surface a brief "didn't catch that"
            // cue, then return to the listening/idle state so the user can
            // retry without tapping to clear it.
            is TranscriptResult.NoSpeech -> {
                if (s.transcript.hasFinalText) return
                Log.d(TAG, "NoSpeech — surfacing 'didn't catch that'")
                if (s.recording) publish(s) else if (session === s) showNoSpeech()
            }
        }
    }

    /** Shows [s] in the status line while it is the keyboard's dictation. */
    private fun publish(s: Session) {
        if (session !== s || s.failed) return
        val text = s.transcript.text
        _uiState.value = when {
            !s.recording -> KeyboardUiState.Transcribing(text, modelLoading = !s.modelReady)
            text.isNotEmpty() -> KeyboardUiState.Processing(text)
            !s.micLive -> KeyboardUiState.MicStarting
            else -> KeyboardUiState.Listening(modelLoading = !s.modelReady)
        }
    }

    /**
     * Ends a dictation whose transcription completed: insert it, and send if requested.
     * Returns whether the text reached its editor.
     */
    private suspend fun complete(s: Session, addTrailingSpace: Boolean): Boolean {
        val localText = s.transcript.finalText.takeIf { s.transcript.hasFinalText && !s.failed }
        // VAD can filter an entire recording, and a long decode can return
        // blank without emitting NoSpeech. Both still need visible feedback.
        if (localText == null && !s.failed && session === s) showNoSpeech()
        val race = s.serverRace
        // The server's text may replace the phone's, or rescue a recording the phone
        // failed or heard nothing in.
        val text = (if (race == null) localText else race.choose(localText).let { decision ->
            if (session === s) _diagnostics.value = _diagnostics.value.copy(transcriptSource = decision.source)
            decision.text
        }) ?: return false
        s.failed = false
        val foreground = session === s
        val editor = s.injector.takeUnless { s.orphaned }
        val insertionStart = System.nanoTime()
        val inserted = editor?.insertTranscript(text, addTrailingSpace) == true
        RecordingTrace.log {
            "insertion session=${s.id} durationMs=${(System.nanoTime() - insertionStart) / 1_000_000} accepted=$inserted orphaned=${s.orphaned}"
        }
        if (editor == null || !inserted) {
            dictationRescue.save(text)
            if (foreground) {
                _isContinuousMode.value = false
                _uiState.value = KeyboardUiState.Error(
                    reason = KeyboardUiState.ErrorReason.TextInsertionFailed,
                    partial = text,
                )
            }
            return false
        }
        if (s.send) editor.send()
        if (foreground) {
            Log.d(TAG, "Transcription flow completed - resetting to Idle")
            _isContinuousMode.value = false
            _uiState.value = KeyboardUiState.Idle
        }
        return true
    }

    /**
     * Runs the keyboard change the user asked for when finishing [s]: the switch key
     * always switches (any unsaved text was rescued); Done and Send return to the
     * previous keyboard after a successful insertion when that preference is on.
     */
    private suspend fun leave(s: Session, finished: Boolean) {
        if (s.orphaned) return
        val switch = s.switchKeyboard
        when {
            switch != null -> switch()
            finished && s.returnWhenDone && appPreferences.returnToPreviousKeyboard.first() ->
                onReturnToPreviousKeyboard?.invoke()
        }
    }

    private fun fail(s: Session, reason: KeyboardUiState.ErrorReason, detail: String? = null) {
        s.failed = true
        stop(s)
        val preview = s.transcript.text
        if (preview.isNotBlank()) {
            dictationRescue.save(preview)
        } else if (session !== s && s.audioStarted.get()) {
            dictationRescue.reportLost()
        }
        if (session === s) {
            _isContinuousMode.value = false
            _uiState.value = KeyboardUiState.Error(reason = reason, detail = detail, partial = preview)
        }
    }

    /** Ends capture for [s]; its buffered audio is still transcribed. */
    private fun stop(s: Session) {
        if (!s.recording) return
        s.recording = false
        audioCaptureManager.stopCapture(s.id)
        RecordingTrace.log { "session stop id=${s.id} current=${session === s}" }
        if (session === s) {
            _isRecording.value = false
            _isContinuousMode.value = false
            publish(s)
        }
    }

    /**
     * The editor of [s] is gone. The dictation keeps finishing in the background and its
     * text is rescued instead of inserted; the keyboard is free for a new dictation.
     */
    private fun detach(s: Session, reason: String) {
        RecordingTrace.log { "session detached id=${s.id} reason=$reason recording=${s.recording} modelReady=${s.modelReady}" }
        stop(s)
        s.orphaned = true
        s.injector = null
        if (session === s) {
            session = null
            _isRecording.value = false
            _isContinuousMode.value = false
            _uiState.value = KeyboardUiState.Idle
            showEngineState(_engineState.value)
        }
    }

    private fun release(s: Session) {
        audioCaptureManager.stopCapture(s.id)
        s.audio.cancel()
        sessions -= s
        if (session !== s) return
        session = null
        _isRecording.value = false
        _isContinuousMode.value = false
        if (_uiState.value.isDictating) _uiState.value = KeyboardUiState.Idle
        if (_uiState.value == KeyboardUiState.Idle) showEngineState(_engineState.value)
        updateWordAtCursor()
    }

    private fun showNoSpeech() {
        noSpeechJob?.cancel()
        _uiState.value = KeyboardUiState.NoSpeech
        noSpeechJob = viewModelScope.launch {
            delay(2000)
            if (_uiState.value is KeyboardUiState.NoSpeech) {
                _uiState.value = KeyboardUiState.Idle
            }
        }
    }

    /**
     * Stop the microphone. The repository finishes collecting the buffered audio and runs
     * its definitive final pass; successful completion inserts the finalized text. The
     * dictation is intentionally not cancelled: slow engines like Whisper can take 10-20 s
     * to produce the Final result after audio ends.
     */
    fun onRecordStop() {
        RecordingTrace.log { "record stop active=${session?.id} recording=${_isRecording.value}" }
        autoStartJob?.cancel()
        // Always reset continuous mode when recording stops from any code path.
        _isContinuousMode.value = false
        val s = session
        if (s == null) {
            _isRecording.value = false
            return
        }
        stop(s)
    }

    /** Done: finish and insert, then return to the previous keyboard if the user enabled that. */
    fun onDone() {
        val s = session ?: return
        RecordingTrace.log { "done requested id=${s.id}" }
        s.returnWhenDone = true
        stop(s)
    }

    /** Send: like [onDone], and submits the editor (e.g. sends the message) after inserting. */
    fun onSend() {
        val s = session ?: return
        RecordingTrace.log { "send requested id=${s.id}" }
        s.send = true
        s.returnWhenDone = true
        stop(s)
    }

    /** Switch key: a dictation in progress is finished and inserted before [switch] runs. */
    fun onSwitchKeyboard(switch: () -> Unit) {
        val s = session
        if (s == null) {
            switch()
            return
        }
        RecordingTrace.log { "switch requested id=${s.id}" }
        s.switchKeyboard = switch
        stop(s)
    }

    /** Explicit user action: drops the dictation without inserting or copying it. */
    fun onDiscard() {
        val s = session ?: return
        RecordingTrace.log { "session discarded id=${s.id}" }
        stop(s)
        s.job?.cancel()
        release(s)
    }

    /**
     * Called when the focused text field is cleared externally - by backspace or the
     * delete key, or because the app cleared it after "Send". A dictation in progress
     * keeps recording and inserts into the same field when it completes.
     */
    fun onFieldCleared() {
        RecordingTrace.log { "field cleared dictation=${session?.id} recording=${_isRecording.value}" }
        _wordSuggestions.value = emptyList()
        _wordAtCursor.value = null
    }

    /**
     * The input session ended (field or app change, IME switched away). A dictation in
     * progress stops recording, finishes transcribing and is rescued instead of inserted.
     */
    fun onInputFinished() {
        RecordingTrace.log { "input finished dictation=${session?.id} recording=${_isRecording.value}" }
        autoStartJob?.cancel()
        returnKeyboardJob?.cancel()
        session?.let { detach(it, "input finished") }
        val ui = _uiState.value
        if (ui is KeyboardUiState.NoSpeech ||
            (ui is KeyboardUiState.Error && ui.reason != KeyboardUiState.ErrorReason.EngineLoadFailed)
        ) {
            _uiState.value = KeyboardUiState.Idle
        }
    }

    override fun onCleared() {
        super.onCleared()
        for (s in sessions.toList()) {
            if (!s.orphaned) detach(s, "keyboard destroyed")
            // The repository binding dies with the keyboard: audio still waiting for the
            // model can never be transcribed. Ready dictations keep their repository.
            if (!s.modelReady) {
                s.job?.cancel()
                if (s.audioStarted.get()) dictationRescue.reportLost()
            }
        }
        _isRecording.value = false
        _isContinuousMode.value = false
        textInjector = null
        _repository.value = null
    }

    class Factory(
        private val audioCaptureManager: AudioCaptureManager,
        private val appPreferences: AppPreferences,
        private val wordSuggestionProvider: WordSuggestionProvider,
        private val dictationRescue: DictationRescue,
        private val serverTranscriber: ServerTranscriber? = null,
        /** Outlives the keyboard so an unfinished dictation completes after the IME is destroyed. */
        private val sessionScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate),
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            KeyboardViewModel(
                audioCaptureManager, appPreferences, wordSuggestionProvider, dictationRescue, sessionScope,
                serverTranscriber,
            ) as T
    }
}

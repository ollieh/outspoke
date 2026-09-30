package dev.brgr.outspoke.ui.keyboard

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import dev.brgr.outspoke.audio.AudioCaptureManager
import dev.brgr.outspoke.ime.EnterAction
import dev.brgr.outspoke.ime.TextInjector
import dev.brgr.outspoke.ime.WordSuggestionProvider
import dev.brgr.outspoke.inference.EngineState
import dev.brgr.outspoke.inference.InferenceRepository
import dev.brgr.outspoke.inference.PipelineDiagnostics
import dev.brgr.outspoke.inference.RecordingTrace
import dev.brgr.outspoke.inference.TranscriptResult
import dev.brgr.outspoke.settings.preferences.AppPreferences
import dev.brgr.outspoke.ui.keyboard.components.WHISPER_LANGUAGE_OPTIONS
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

private const val TAG = "KeyboardViewModel"

/**
 * Bridges the IME lifecycle, audio capture, and inference pipeline into a stream of
 * [KeyboardUiState] values consumed by [KeyboardScreen].
 */
class KeyboardViewModel(
    private val audioCaptureManager: AudioCaptureManager,
    private val appPreferences: AppPreferences,
    private val wordSuggestionProvider: WordSuggestionProvider,
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

    /** Streaming results have already been cleaned and custom spellings must stay intact. */
    val transcriptAlreadyCleaned: Boolean
        get() = inferenceRepository?.producesCleanTranscripts == true

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
    private fun isActiveSession(): Boolean {
        val s = _uiState.value
        return s is KeyboardUiState.Listening ||
                s is KeyboardUiState.Processing ||
                s is KeyboardUiState.Transcribing
    }

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
        inferenceRepository?.setLanguage(tag)
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
        if (captureJob?.isActive == true) return
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

    private var inferenceRepository: InferenceRepository? = null

    /** Set by [OutspokeInputMethodService] when the service binding is established. */
    fun setInferenceRepository(repo: InferenceRepository?) {
        inferenceRepository = repo
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

    fun setTextInjector(injector: TextInjector?) {
        RecordingTrace.log { "editor attached=${injector != null} recording=${_isRecording.value}" }
        if (textInjector !== injector && captureJob?.isActive == true) onInputFinished()
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
        // Cancel any lingering capture / inference from the failed session so it cannot
        // bleed into the next one, then return to Idle.
        captureJob?.cancel()
        captureJob = null
        recordingTranscript = null
        audioCaptureManager.stopCapture()
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
     */
    fun performEnterAction() {
        textInjector?.performEnterAction()
    }

    private var captureJob: Job? = null
    private var recordingTranscript: RecordingTranscript? = null
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
            _engineState.first { it == EngineState.Ready }
            // A quick reopen can arrive while the previous recording is finalizing.
            captureJob?.join()
            val holdToRecord = appPreferences.triggerMode.first() == "HOLD"
            autoStartJob = null
            if (!keyboardVisible || captureJob?.isActive == true) return@launch
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
        val finishingCapture = captureJob
        onRecordStop()
        returnKeyboardJob = viewModelScope.launch {
            if (!appPreferences.returnToPreviousKeyboard.first()) return@launch
            finishingCapture?.join()
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
     * [InferenceRepository] is currently bound — a binder desync that would otherwise let
     * the mic open with no transcription. The callback lets the IME react (rebind / reload)
     * instead of the VM silently doing nothing.
     */
    var onMissingRepo: (() -> Unit)? = null

    /**
     * Start microphone capture and pipe audio through the inference engine.
     * Ignored if the engine is not yet [EngineState.Ready] or if no
     * [InferenceRepository] is currently bound (a binder desync).
     */
    fun onRecordStart() {
        RecordingTrace.log { "record requested ready=${_engineState.value == EngineState.Ready} previousActive=${captureJob?.isActive} injector=${textInjector != null}" }
        autoStartJob?.cancel()
        if (_engineState.value !is EngineState.Ready) {
            Log.w(TAG, "onRecordStart() ignored - engine not ready (${_engineState.value})")
            return
        }

        // EngineState.Ready without a bound InferenceRepository means the UI shows
        // a ready talk button but transcription would silently no-op (the old code fell
        // into a "capture audio without transcription" branch). Surface the desync as a
        // loading state and ask the IME to reload/rebind instead of opening the mic.
        if (inferenceRepository == null) {
            Log.w(
                TAG,
                "onRecordStart() ignored - engine Ready but no InferenceRepository bound (desync); requesting reload"
            )
            _uiState.value = KeyboardUiState.EngineLoading(
                KeyboardUiState.LoadingReason.EngineStarting
            )
            onMissingRepo?.invoke()
            return
        }

        captureJob?.cancel()
        noSpeechJob?.cancel()
        _isRecording.value = true
        _uiState.value = KeyboardUiState.Listening
        _diagnostics.value = PipelineDiagnostics()
        _wordSuggestions.value = emptyList()
        // Dismissed flag is no longer set persistently — bar reappears on next word tap.

        val transcript = RecordingTranscript()
        recordingTranscript = transcript
        val sessionInjector = textInjector
        captureJob = viewModelScope.launch {
            // Capture this coroutine's Job reference so the collect lambda can detect
            // whether it belongs to the currently active session.  If onRecordStart() is
            // called again (e.g. from onFieldCleared after a "Send"), captureJob is
            // replaced with the new job; any collect callbacks still queued from the OLD
            // job see captureJob != myJob and return immediately - preventing stale
            // partial results from bleeding into the fresh TextInjector state.
            val myJob = coroutineContext[Job]
            val traceSession = RecordingTrace.nextSessionId()
            var results = 0
            RecordingTrace.log { "session start id=$traceSession" }
            var inserted = false
            var failed = false
            val addTrailingSpace = trailingSpace.value

            try {
                if (!_isRecording.value) {
                    _uiState.value = KeyboardUiState.Idle
                    return@launch
                }
                // Repo was verified non-null above; re-read defensively in case the binder
                // detached between the guard and this launch. If it did become null, abort
                // cleanly rather than capturing audio with nowhere to send it.
                val repo = inferenceRepository
                if (repo == null) {
                    Log.w(TAG, "InferenceRepository detached before capture started - aborting session")
                    _isContinuousMode.value = false
                    _uiState.value = KeyboardUiState.EngineLoading(
                        KeyboardUiState.LoadingReason.EngineStarting
                    )
                    onMissingRepo?.invoke()
                    return@launch
                }

                // Pipe audio through the inference engine on Dispatchers.Default.
                // Results update the preview; completion performs the editor insertion.
                repo.transcribe(
                    audio = audioCaptureManager.startCapture(
                        vadEnabled = vadSensitivity.value,
                        rawSource = rawMicCapture.value,
                        recordingId = traceSession,
                    ),
                    postprocessingEnabled = postprocessingEnabled.value,
                    formatNumbersAsDigits = formatNumbersAsDigits.value,
                    customWords = customWords.value,
                    recordingId = traceSession,
                ).collect { result ->
                    results++
                    RecordingTrace.log {
                        val chars = when (result) {
                            is TranscriptResult.Partial -> result.text.length
                            is TranscriptResult.Final -> result.text.length
                            else -> 0
                        }
                        "result session=$traceSession count=$results type=${result.javaClass.simpleName} chars=$chars current=${captureJob == myJob} injector=${textInjector != null}"
                    }
                    // Stale-session guard: if this job has been superseded (e.g. the field
                    // was cleared and a new session started), discard this result entirely
                    // so no old-session text is injected into the fresh TextInjector.
                    if (captureJob != myJob || failed) return@collect

                    when (result) {
                        is TranscriptResult.Partial -> {
                            transcript.partial(result.text)
                            _uiState.value = if (_isRecording.value) KeyboardUiState.Processing(transcript.text)
                                else KeyboardUiState.Transcribing(transcript.text)
                        }

                        is TranscriptResult.Final -> {
                            transcript.final(result.text)
                            _uiState.value = if (_isRecording.value) KeyboardUiState.Processing(transcript.text)
                                else KeyboardUiState.Transcribing(transcript.text)
                        }

                        is TranscriptResult.Failure -> {
                            failed = true
                            _isRecording.value = false
                            audioCaptureManager.stopCapture()
                            Log.e(TAG, "Transcription failure", result.cause)
                            _isContinuousMode.value = false
                            _uiState.value = KeyboardUiState.Error(
                                reason = KeyboardUiState.ErrorReason.TranscriptionFailed,
                                detail = result.cause.message,
                                partial = transcript.text,
                            )
                        }

                        is TranscriptResult.WindowTrimmed -> {
                            transcript.windowTrimmed()
                            _diagnostics.value = _diagnostics.value.copy(
                                windowTrims = _diagnostics.value.windowTrims + 1
                            )
                        }

                        // The model saw audio but couldn't resolve a word (weak
                        // high-frequency fricatives). Surface a brief "didn't catch that"
                        // cue, then return to the listening/idle state so the user can
                        // retry without tapping to clear it.
                        is TranscriptResult.NoSpeech -> {
                            if (transcript.hasFinalText) return@collect
                            Log.d(TAG, "NoSpeech — surfacing 'didn't catch that'")
                            if (_isRecording.value) {
                                _uiState.value = if (transcript.text.isEmpty()) KeyboardUiState.Listening
                                    else KeyboardUiState.Processing(transcript.text)
                                return@collect
                            }
                            showNoSpeech()
                        }
                    }
                }

                if (captureJob != myJob || sessionInjector !== textInjector) return@launch
                if (transcript.hasFinalText && !failed) {
                    val insertionStart = System.nanoTime()
                    inserted = sessionInjector?.insertTranscript(transcript.finalText, addTrailingSpace) == true
                    RecordingTrace.log {
                        "insertion session=$traceSession durationMs=${(System.nanoTime() - insertionStart) / 1_000_000} accepted=$inserted"
                    }
                    if (!inserted) {
                        failed = true
                        _uiState.value = KeyboardUiState.Error(
                            reason = KeyboardUiState.ErrorReason.TextInsertionFailed,
                            partial = transcript.text,
                        )
                    }
                }
                // VAD can filter an entire recording, and a long decode can return
                // blank without emitting NoSpeech. Both still need visible feedback.
                if (!transcript.hasFinalText && !failed) showNoSpeech()
                // Completion, including an empty final utterance, releases the recording UI.
                if (_uiState.value is KeyboardUiState.Transcribing ||
                    _uiState.value == KeyboardUiState.Listening ||
                    _uiState.value is KeyboardUiState.Processing
                ) {
                    Log.d(TAG, "Transcription flow completed - resetting to Idle")
                    _isContinuousMode.value = false
                    _uiState.value = KeyboardUiState.Idle
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: SecurityException) {
                Log.e(TAG, "Microphone permission denied", e)
                _isContinuousMode.value = false
                _uiState.value = KeyboardUiState.Error(
                    reason = KeyboardUiState.ErrorReason.MicPermissionDenied,
                    partial = transcript.text,
                )
            } catch (e: IllegalStateException) {
                Log.e(TAG, "AudioRecord failed to initialise", e)
                _isContinuousMode.value = false
                _uiState.value = KeyboardUiState.Error(
                    reason = KeyboardUiState.ErrorReason.MicInitFailed,
                    detail = e.message,
                    partial = transcript.text,
                )
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected audio capture error", e)
                _isContinuousMode.value = false
                _uiState.value = KeyboardUiState.Error(
                    reason = KeyboardUiState.ErrorReason.AudioCaptureFailed,
                    detail = e.message,
                    partial = transcript.text,
                )
            } finally {
                RecordingTrace.log { "session end id=$traceSession results=$results inserted=$inserted failed=$failed current=${captureJob == myJob}" }
                if (captureJob == myJob) {
                    audioCaptureManager.stopCapture()
                    _isRecording.value = false
                    _isContinuousMode.value = false
                    captureJob = null
                    recordingTranscript = null
                    setEngineState(_engineState.value)
                    updateWordAtCursor()
                }
            }
        }
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

    fun onRecordStop() {
        RecordingTrace.log { "record stop active=${captureJob?.isActive} recording=${_isRecording.value}" }
        autoStartJob?.cancel()
        _isRecording.value = false
        // Always reset continuous mode when recording stops from any code path.
        _isContinuousMode.value = false

        // Stop the microphone - this terminates the upstream audio flow, which causes
        // InferenceRepository to finish collecting audio and run its definitive final
        // inference pass.  We intentionally do NOT cancel captureJob here: slow engines
        // like Whisper can take 10-20 s to produce the Final result after audio ends,
        // and cancelling the job would discard the recording. Successful flow
        // completion inserts the finalized text and releases the job.
        audioCaptureManager.stopCapture()

        // Switch to Transcribing so the UI clearly signals "mic off, engine still working".
        // The collect loop in onRecordStart() will transition back to Idle (and commit
        // the text) once the recording flow completes.
        if (_uiState.value is KeyboardUiState.Listening ||
            _uiState.value is KeyboardUiState.Processing
        ) {
            _uiState.value = KeyboardUiState.Transcribing(recordingTranscript?.text.orEmpty())
        }
    }

    /**
     * Called when the focused text field is cleared externally - typically because the user
     * pressed "Send" in a messaging app and the app cleared the [EditText] content.
     *
     * Discards pending text. Active dictation resumes as a fresh recording;
     * a stopped recording is cancelled so it cannot fill the cleared field later.
     */
    fun onFieldCleared() {
        RecordingTrace.log { "field cleared recording=${_isRecording.value}" }
        val resumeRecording = _isRecording.value
        val continuous = _isContinuousMode.value
        onInputFinished()
        _wordSuggestions.value = emptyList()
        _wordAtCursor.value = null
        if (resumeRecording) {
            onRecordStart()
            _isContinuousMode.value = continuous
        }
    }

    /** Discards unfinished speech when its input session is no longer valid. */
    fun onInputFinished() {
        RecordingTrace.log { "input finished active=${captureJob?.isActive} recording=${_isRecording.value}" }
        autoStartJob?.cancel()
        returnKeyboardJob?.cancel()
        audioCaptureManager.stopCapture()
        captureJob?.cancel()
        captureJob = null
        recordingTranscript = null
        _isRecording.value = false
        _isContinuousMode.value = false
        _uiState.value = KeyboardUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        captureJob?.cancel()
        audioCaptureManager.stopCapture()
        _isRecording.value = false
        _isContinuousMode.value = false
        textInjector = null
        inferenceRepository = null
    }

    class Factory(
        private val audioCaptureManager: AudioCaptureManager,
        private val appPreferences: AppPreferences,
        private val wordSuggestionProvider: WordSuggestionProvider,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            KeyboardViewModel(audioCaptureManager, appPreferences, wordSuggestionProvider) as T
    }
}

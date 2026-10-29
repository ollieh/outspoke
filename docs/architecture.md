# Outspoke - Architecture Reference

> **Scope**: Authoritative technical reference for the Outspoke Android IME codebase.
> Intended for developers adding features, new engines, or debugging the inference pipeline.
> For user-facing information see README.md; for potential future improvements see analysis.md.

---

## Table of Contents

1. High-Level Overview
2. Module and Package Map
3. Layer-by-Layer Architecture
4. Data Flow End to End
5. Key Interfaces and Sealed Classes
6. Sliding-Window Inference Details
7. Post-Processing Pipeline
8. Text Injection and Alignment
9. Model Registry and Download
10. Engine Implementations
11. Service and Lifecycle Management
12. State Machines
13. Build Configuration and ABI Splits
14. Testing Strategy
15. Extension Guide: Adding a New Engine
16. Key Conventions and Invariants

---

## 1. High-Level Overview

Outspoke is a privacy-first Android Input Method Editor (IME). All speech recognition runs
on-device via ONNX Runtime; no audio ever leaves the device after the one-time model download.

    Active App (Text Field)
         | InputConnection API
    OutspokeInputMethodService    <- Android IME entry-point (LifecycleOwner + Compose UI)
         | KeyboardViewModel      <- UI state + capture lifecycle
         | binds to
    InferenceService              <- Foreground LifecycleService
         | InferenceRepository    <- Sliding-window buffer (<=30 s)
         |   SpeechEngine         <- Interface; swap models here (currently ParakeetEngine)
         | Flow<AudioChunk>
    AudioCaptureManager           <- 16 kHz / 16-bit / mono PCM
    SileroVadFilter               <- Neural VAD (Silero v4, ONNX) - primary
    RMSVadFilter                  <- Energy-threshold VAD - fallback

**Core design principles:**

- SpeechEngine is the *only* seam for adding a new ASR model. Nothing in the IME or service layer changes.
- InferenceService keeps the engine alive across keyboard hide/show cycles. Unbinding does **not** stop the service.
- Constructor injection only throughout; no field injection.
- No external SDKs that phone home (no analytics, no crash reporters).

---

## 2. Module and Package Map

Single Gradle module (app). All Kotlin source lives under app/src/main/kotlin/ (package root dev.brgr.outspoke).

| Package | Key files | Responsibility |
|---|---|---|
| audio | AudioCaptureManager, MicCalibrationManager, SileroVadFilter, RMSVadFilter, VadFilter, AudioChunk, PermissionHelper | Mic capture, PCM chunking, Voice Activity Detection, optional mic calibration |
| inference | SpeechEngine, ParakeetEngine, ChunkStreamingEngine, ParakeetUnifiedEngine, StreamingSpeechEngine, NemoLogMel, WhisperEngine, VoxtralEngine, SpeechEngineFactory, InferenceRepository, InferenceService, TranscriptResult, EngineState, PipelineDiagnostics, NumberNormaliser, GrammarCorrector, AcousticCandidates | ASR pipeline, sliding window, post-processing, foreground service, acoustic word-alternative capture |
| ime | OutspokeInputMethodService, TextInjector, TranscriptAligner, EnterAction, WordSuggestionProvider, DictationRescue | Keyboard service, recording insertion, clipboard rescue, legacy alignment, word-correction facade |
| ime/correction | WordCorrector, CandidateGenerator, ArpaLanguageModel, SuggestionFileDownloader, SuggestionFileManager, SuggestionLanguage, DoubleMetaphone, KolnerPhonetik, EditDistance | On-device word correction + downloadable language pack management |
| settings/model | ModelId, ModelRegistry, ModelDownloadManager, ModelStorageManager, ModelState, ModelViewModel, DownloadService | Model enumeration, download, SHA-256 verification, on-disk paths |
| settings/preferences | AppPreferences, PreferencesViewModel | DataStore-backed user preferences |
| settings/screens | HomeScreen, ModelScreen, PreferencesScreen, MicCalibrationScreen | Settings Compose UI |
| ui/keyboard | KeyboardViewModel, KeyboardUiState, RecordingTranscript, KeyboardScreen, ImeComposeView, WordAtCursor | IME Compose hosting, UI state |
| ui/keyboard/components | TalkButton, SuggestionBar, WaveformBar, StatusIndicator, KeyboardActionButton, KeyboardTutorialOverlay, LanguageSelector, RecordingControls | Keyboard UI sub-components |
| ui/theme | OutspokeKeyboardTheme | Compose theming |

---

## 3. Layer-by-Layer Architecture

### 3.1 Audio Capture and VAD

**AudioCaptureManager**

- Opens AudioRecord with source DEFAULT (the vendor-recommended source), 16 kHz mono PCM-16. No application-side gain: DEFAULT arrives at a usable level.
- Emits 480-sample (30 ms) chunks as a cold Flow<AudioChunk>.
- Uses a 2 s AudioRecord buffer. A dedicated reader drains AudioRecord into an unbounded queue; RMS, VAD and the downstream send run on a separate coroutine, so they never delay a read. Recording starts before the VAD model is created.
- `onAudioStarted` fires when the microphone first delivers samples; the keyboard shows "Listening" only after that. `stopCapture(recordingId)` stops only that recording, never a newer one.
- When the user has run mic calibration, applies the selected microphone via AudioRecord.setPreferredDevice (no-op when unset or the device is gone).
- On stop: drains the hardware buffer then waits HANGOVER_DRAIN_SAFETY_FRAMES = 20 frames (600 ms) to flush VAD hangover before the flow completes.

**VadFilter interface**

    interface VadFilter {
        fun process(chunk: AudioChunk, rms: Float): AudioChunk?  // null = silence, drop
        fun flush(): List<AudioChunk>                             // drain tail on stop
        fun isSpeechActive(): Boolean
        fun close()
    }

**SileroVadFilter** (primary)

- Runtime neural VAD failures switch to energy detection for the rest of the recording, retaining onset and hangover state. Input tensors are closed even when inference throws.
- Silero VAD v4 ONNX. RNN state tensors h/c shape [2,1,64] carried across chunks.
- Speech probability threshold: 0.3.
- Three layers: onset gate (2 frames/60 ms), pre-roll buffer (15 frames/450 ms lead-in), hangover.
- Falls back to RMSVadFilter if the ONNX model fails to load.

**RMSVadFilter** (fallback)

- Energy threshold with the same 3-layer onset/pre-roll/hangover structure.

**MicCalibrationManager / MicScorer** (optional, user-triggered)

- MicCalibrationScreen (settings) lists the input microphones, records a short reference clip on each, and ranks them with MicScorer (HF energy fraction + percentile SNR).
- The winning mic id is persisted (AppPreferences.preferredMicId) and applied to production capture via AudioRecord.setPreferredDevice; falls back to the system default when the device is no longer present.
- Strict opt-in: nothing runs unless the user opens the calibration screen.

---

### 3.2 Inference Pipeline

**SpeechEngine interface** - the single extension point for new models:

    interface SpeechEngine {
        val isLoaded: Boolean
        fun load(modelDir: File)
        fun transcribe(chunk: AudioChunk): TranscriptResult
        fun setLanguage(tag: String)
        fun setLanguageFilter(tags: List<String>)
        fun close()
    }

**SpeechEngineFactory** routes ModelId to a concrete engine implementation.

Parakeet carries predictor state across streaming chunks. After 25 encoder frames
(two seconds) without a token, it resets the predictor's LSTM state and previous
token. The blank-frame count travels in `TdtState` so pauses spanning chunk
boundaries also reset it. This prevents the previous sentence's state from
suppressing speech after a pause, including when VAD is disabled. Accumulated
tokens and the decoder's frame position are preserved.

If a chunk emits no letters or digits with a carried predictor, Parakeet also retries that
same frame range once with fresh predictor state, reusing the encoder output.
The retry replaces the blank result only when it contains a letter or digit and
its geometric-mean token confidence is at least 0.55. This recovers speech that
would otherwise be skipped before the blank-frame reset takes effect.
Punctuation-only chunks qualify for recovery: a trailing period can be emitted
while all the following speech is suppressed.
Detokenisation preserves literal digits and standalone SentencePiece word
boundaries from the vocabulary.

Each two-second chunk is encoded with two seconds of preceding audio and four
seconds of lookahead. The first live partial can therefore appear after six
seconds plus inference time; stopping flushes the remaining audio without waiting
for more lookahead.

If a chunk still contains no words, the repository re-encodes up to four seconds
of preceding audio and decodes that overlap through the current chunk with fresh
predictor state. `findStreamingRecoveryStart` anchors the result to a suffix of
the existing transcript, checking at most 48 previous tokens and requiring at
least two matching tokens containing a word. Only the new continuation is
appended, with a confidence gate of 0.55 calculated over its tokens alone.
Unmatched, punctuation-only and already-emitted results are discarded. The
normal encoder window is at most eight seconds; recovery uses at most ten.
This recovers speech whose onset fell in an earlier chunk that suppressed it.

**InferenceRepository**

Number formatting validates phrase boundaries before combining number words, preserving countdowns and punctuation-separated lists. Stutter and phrase cleanup retain repeated numeric tokens, including number words, so dictated codes are not shortened.

- Receives Flow<AudioChunk> from AudioCaptureManager.
- Maintains a rolling audio window (see Section 6).
- Emits Flow<TranscriptResult> consumed by KeyboardViewModel -> TextInjector.
- Applies an 8-step post-processing pipeline to every raw transcript (see Section 7).

**InferenceService**

- LifecycleService running as an Android foreground service.
- Notification channel: outspoke_inference, notification ID 1001.
- Owns the SpeechEngine instance; reloads on selectedModelId preference change.
- Uses a Mutex to protect engine swaps. The repository serializes recording sessions and engine teardown, including the final decode; memory-pressure unload skips an active session.
- Unloads the model only on `TRIM_MEMORY_RUNNING_CRITICAL` or `onLowMemory`. Background trim levels are routine after a keyboard switch and keep the model loaded.
- Observes distinct selected model IDs so unrelated preference changes do not reload the model.
- Exposes StateFlow<EngineState> to bound clients.
- FileObserver on models/ directory detects newly downloaded models.
- Stays alive independently of IME bind/unbind; shuts down only when explicitly stopped. If foreground promotion fails it stays started as a background service rather than calling `stopSelf`.
- Debug `OutspokeTrace` lines record service create/start/destroy, trim levels, and engine load/unload with a reason.

---

### 3.3 IME and Text Injection

**OutspokeInputMethodService**

- Extends InputMethodService; implements LifecycleOwner and SavedStateRegistryOwner to host Compose.
- Hosts the keyboard UI via ImeComposeView.
- Binds to InferenceService for the IME lifetime and starts it when shown so it can survive IME recreation.
- Applies FLAG_KEEP_SCREEN_ON while capture is active and clears it when capture stops.
- Forwards InputConnection changes to TextInjector and KeyboardViewModel.
- Input preferences `auto_start_recording` and `return_to_previous_keyboard` are booleans, both false by default. Auto recording starts once per keyboard opening as soon as the service reports Loading or Ready, with tap-to-stop enabled even in Hold mode. Hiding cancels a pending start and stops capture. Returning to the previous keyboard happens after Done or Send, or after hiding; it waits for the final transcript, is cancelled on reopening, and checks that Outspoke is still the selected IME.
- `onStartInput(restarting = true)` keeps a dictation attached to the same field. Any other new or finished input session detaches it.

**TextInjector**

- Manages all writes to the active InputConnection.
- `insertTranscript` writes a completed recording in one `commitText` call, including separators and the optional trailing space.
- `send` performs the editor's declared IME action, even in multi-line fields; without one it sends an Enter key down/up pair.
- Recording previews and audio-window overlap live in `RecordingTranscript`, with no editor writes while recording or finishing.
- Older composing-span methods remain for legacy callers and alignment tests, delegating overlap discovery to TranscriptAligner.

**TranscriptAligner** (pure / stateless)

- normalizeWord(), splitToWords(), wordsMatch().
- findNewContent(committed, partial) performs a 3-layer overlap search (see Section 8).
- Tolerates Parakeet attention drift and post-trim leading garbage tokens.

---

### 3.4 Settings and Model Lifecycle

**ModelId enum:**

| Value | storageDirName | Status |
|---|---|---|
| PARAKEET_V3 | parakeet-v3 | Active |
| VOXTRAL_MINI | voxtral-mini-4b | Disabled (resource limits) |
| WHISPER_SMALL | whisper-small-int8 | Disabled (resource limits) |

> Warning: storageDirName must never change after release - it is the on-disk key for existing installations.

**ModelRegistry** maps each ModelId to a ModelInfo: display name, DownloadSource, RemoteFile list with URLs and SHA-256 hashes, size estimate. ModelRegistry.all contains only models confirmed to run within acceptable resource limits on real devices.

**ModelStorageManager** stores all files in <filesDir>/models/<storageDirName>/. Checks requiredFiles list for readiness; uses INSTALLED_MARKER file for ZIP-installed models.

**ModelDownloadManager** downloads via OkHttp with resume support, verifies SHA-256 after each file, emits ModelState.Downloading(progress).

**AppPreferences** (DataStore, store name outspoke_prefs): trigger_mode (String, default HOLD), trailing_space (Boolean, default false; applied once after successful recording completion), delete_button_mode (String, DELETE_ALL | DELETE_LAST_SENTENCE, default DELETE_ALL), vad_sensitivity (Float, default 0.0), selected_model_id (String), whisper_language (String, default "auto"), postprocessing_enabled (Boolean, default true), show_pipeline_diagnostics (Boolean, default false), keyboard_tutorial_shown (Boolean, default false), forced_language (String?, default null), format_numbers_as_digits (Boolean, default true), suggestion_bar_enabled (Boolean, default false), suggestion_bar_languages (String, comma-separated BCP-47 tags, default ""), suggestion_bar_dismissed (Boolean, default false), raw_mic_capture (Boolean, default false — true captures from AudioSource.UNPROCESSED to bypass AEC, needed for the speakerphone use case), preferredMicId (Int, default 0).

---

### 3.5 UI Layer

**Custom vocabulary:** `custom_words` is a DataStore string set, empty by default. Speech preferences support adding and removing up to 100 terms, each at most 50 characters and three words. `CustomWordCorrector` normalizes duplicate entries while preserving the saved spelling. The repository snapshots the vocabulary when a recording begins and corrects full partial/final transcripts before editor alignment. The streaming display path passes these results through unchanged, avoiding a second cleaner that could alter saved casing or number-containing terms.

Matching considers up to three recognized words without crossing punctuation or newlines. It uses normalized edit distance below 0.18, with a phonetic bonus only for ASCII words within two edits. Short words and numeric strings require exact normalized matches; equally good alternatives stay unchanged. This is text correction, not acoustic decoder biasing, and cannot restore words the model omitted. It requires no download and works independently of the suggestion bar.

**KeyboardViewModel** bridges IME lifecycle, AudioCaptureManager, and InferenceRepository results into KeyboardUiState. Owns captureJob - the coroutine driving audio capture for a recording session.

**KeyboardUiState** (sealed class):

| State | Meaning |
|---|---|
| Idle | Keyboard visible, not recording |
| Listening | Mic open, VAD active |
| Processing(partial) | Partial transcript available |
| Transcribing(partial) | Final inference running; recording preview remains visible |
| EngineLoading(reason) | Engine is being loaded |
| Error(reason, detail) | Unrecoverable error |

---

## 4. Data Flow End to End

    Microphone (PCM-16, 16 kHz, 480-sample chunks)
        |
    AudioCaptureManager  [peak-envelope normalisation]
        |  AudioChunk (float32)
    SileroVadFilter / RMSVadFilter  [silence suppression]
        |  AudioChunk (speech only)
    InferenceRepository  [rolling window, fires every 1 s once >=2 s buffered]
        |
    SpeechEngine.transcribe()  [ParakeetEngine: nemo128 -> encoder -> decoder/joint]
        |  raw transcript string
    cleanTranscript()  [8-step post-processing]
        |  TranscriptResult (Partial | Final | WindowTrimmed | Failure)
    KeyboardViewModel
        |-- KeyboardUiState --> KeyboardScreen (Compose)
        +-- TextInjector
                |  TranscriptAligner.findNewContent()
            InputConnection (setComposingText / commitText)
                |
            Active App Text Field

---

## 5. Key Interfaces and Sealed Classes

### TranscriptResult (sealed class)

| Variant | Fields | Recording action |
|---|---|---|
| Partial(text) | text: String | Replace the active utterance in the keyboard preview |
| Final(text, isUtteranceBoundary) | isUtteranceBoundary: Boolean | Accumulate a finalized utterance; boundary finals continue capture; insert the recording once after successful flow completion |
| WindowTrimmed(stableWords) | stableWords: List<String> | Reconcile the retained audio-window overlap in recording memory |
| Failure(exception) | exception: Throwable | Log; surface to user via KeyboardUiState.Error |

### EngineState (sealed class)

Unloaded | Loading | Ready | Error(message: String)

### ModelState (sealed class)

NotDownloaded | Downloading(progress: Float) | Ready | Corrupted

### PipelineDiagnostics

Counters: windowTrims, alignmentRecoveries, blanksDiscarded.
summary() returns e.g. "2T . 1R . 3B" or "OK" when all zero.

---

## 6. Sliding-Window Inference Details

This section describes the fallback path for engines without `ChunkStreamingEngine` or `StreamingSpeechEngine`.
Parakeet uses the chunked path described above. Key constants in InferenceRepository:

| Constant | Value | Purpose |
|---|---|---|
| MIN_SAMPLES | 2 s | Minimum window before first partial fires |
| STRIDE_SAMPLES | 1 s | New inference every second of new audio |
| MAX_WINDOW_SAMPLES | 30 s | Hard ceiling |
| STABLE_STRIDES | 3 | Consecutive strides that must agree before audio trim |
| TRIGGER_WINDOW_SAMPLES | 6 s | Trim logic activates above this size |
| MIN_CONTEXT_SAMPLES | 4 s | Tail context kept after a stable-chunk trim |
| FORCE_TRIM_WINDOW_SAMPLES | 12 s | Aggressive trim when strides diverge |
| SILENCE_TRIM_STRIDES | 2 | Consecutive blank strides before proactive trim |
| SHORT_UTTERANCE_THRESHOLD_SAMPLES | 2.5 s | Below this the final flush uses the short-utterance path |
| MIN_PADDING_SAMPLES | 1.25 s | Tail zero-pad floor for short-utterance decodes |
| SHORT_UTT_LEAD_SILENCE_SAMPLES | 600 ms | Lead silence prepended to short-utterance decodes |
| CONFIDENCE_THRESHOLD | 0.55 | Legacy (non-Parakeet) final-confidence gate; the Parakeet short-utterance path uses a plausibility floor (≥2 word chars) instead |

**Trim triggers** (all emit TranscriptResult.WindowTrimmed):

1. Stable-prefix trim: last 3 partials share a common leading-word prefix -> trim audio corresponding to stable words, retain MIN_CONTEXT_SAMPLES tail.
2. Silence trim: 2 consecutive blank strides -> proactive trim regardless of prefix agreement.
3. Force trim: window > FORCE_TRIM_WINDOW_SAMPLES with no stable prefix -> unconditional trim.

**Mid-session final**: when isSilenceBoundary is set on an AudioChunk, the repository emits Final(isUtteranceBoundary = true) without stopping capture (useful for long continuous dictation).

**Recording-stop final**: one final inference runs over the remaining window when the audio flow completes. **Short utterances** (< SHORT_UTTERANCE_THRESHOLD_SAMPLES) get a decode-context retry: the TDT decoder is extremely context-sensitive on short clips (real-model probes: 600 ms of leading silence flips blank decodes into correct words; trailing digital silence can flip a correct decode into blank). The primary attempt re-decodes the buffer from frame 0 with SHORT_UTT_LEAD_SILENCE_SAMPLES of silence prepended (and a tail zero-pad to MIN_PADDING_SAMPLES when still short). A decode is accepted when it contains a plausible word (≥2 letters/digits); if the lead attempt yields no word, one retry runs the original context (no lead), and the first attempt with a word wins. There is **no confidence gate** on the Parakeet short-utterance path: a correct-but-low-confidence decode is always emitted (the geometric-mean confidence of a single short word is fragile — one low-probability token, often the trailing period, can drag it below any fixed threshold — and other Parakeet tools emit whatever the model decodes rather than gating on confidence). A no-word decode in both contexts stays silent (no output, no warning) rather than emitting a hallucination.

---

## 7. Post-Processing Pipeline

Applied to every raw transcript string before wrapping in a TranscriptResult:

| Step | Operation |
|---|---|
| 1 | Filler-word removal (um, uh, hmm, etc.) |
| 2 | Stutter collapse (>=3x consecutive word repeats -> single instance) |
| 3 | Phrase-loop deduplication (repeated phrase sequences) |
| 3.5 | Spurious-period removal (`filterSpuriousPeriods`): strips mid-utterance periods produced by Parakeet TDT on prosodic pauses. A period is removed when it is **not** the final token AND either (a) the preceding word is a conjunction/preposition/article/determiner in `NON_SENTENCE_CLOSING_WORDS`, or (b) the sentence segment before the period is fewer than 5 words. Must run before step 8 (capitalisation) to prevent false sentence-start capitalisation. |
| 4 | Leading-dot strip (LEADING_DOTS_RE) |
| 5 | Leading-punctuation strip (LEADING_PUNCT_RE: ^[,;]+) |
| 6 | Multi-dot normalisation (two or more consecutive dots -> single dot) |
| 7 | Missing sentence-space repair (punctuation followed by capital letter without space) |
| 8 | Sentence-boundary capitalisation |

---

## 8. Text Injection and Alignment

### Recording preview and insertion

`KeyboardViewModel` owns one `RecordingTranscript` per capture job. Partials replace the active utterance in the keyboard preview; finals accumulate completed utterances without writing to the editor. A silence boundary resets the active utterance and continues capture. Legacy `WindowTrimmed` events preserve the dropped prefix in memory and reconcile later hypotheses through overlap matching.

Successful flow completion inserts finalized utterances once through `TextInjector.insertTranscript`. A single `commitText` includes the recording, a preceding separator when needed, and the optional trailing space. Selected text is replaced directly. Suggestions refresh after insertion; automatic keyboard return waits for completion.

Recording starts the microphone at once, even while the model loads. Audio is buffered in an unbounded channel until the repository is Ready, then transcribed. The status line shows `MicStarting` until the microphone delivers samples, then `Listening(modelLoading)`, `Processing(partial)` and `Transcribing(partial, modelLoading)`.

Dictated speech is dropped only by the explicit Discard control. Clearing the field (backspace, the delete key, or the app) keeps recording and inserts into the same field. A finished or changed input session, or a destroyed keyboard, detaches the dictation: capture stops, transcription finishes in a scope that outlives the view model, and the text goes to the clipboard with a toast (`ClipboardDictationRescue`). It is never inserted into another field. A rejected insertion and a failed decode with a preview are rescued the same way. Audio that was still waiting for the model when the keyboard was destroyed cannot be transcribed; the user is told.

While dictating, the keyboard shows Discard beside the waveform, and the Enter key turns into Send. The talk button finishes and inserts, leaving the keyboard up for edits. Send inserts, then calls `TextInjector.send`, then returns to the previous keyboard when `return_to_previous_keyboard` is on. The switch-keyboard key finishes and inserts first, showing "Finishing…", then switches.

`Transcribing(partial)` preserves the preview after Stop. Errors retain the preview and leave the editor unchanged; a rejected editor insertion has its own error reason. The older composing-span methods remain for legacy callers and alignment tests; the recording UI no longer freezes words or updates composing text.

---

## 9. Model Registry and Download

### Storage layout

    <filesDir>/models/
      parakeet-v3/
        nemo128.onnx
        encoder-model.int8.onnx
        decoder_joint-model.int8.onnx
        config.json
        vocab.txt
      voxtral-mini-4b/    (placeholder - disabled)
      whisper-small-int8/ (placeholder - disabled)

    <filesDir>/suggestion_files/
      en/
        dict_en.txt
        lm_en.arpa
      de/
        dict_de.txt
        lm_de.arpa
      (one subdirectory per downloaded language)

### Download flow

1. ModelDownloadManager fetches each RemoteFile via OkHttp.
2. Supports HTTP range requests for resume.
3. SHA-256 verified after each file; on mismatch the file is deleted and download fails with ModelState.Corrupted.
4. ZIP archives are extracted in place; INSTALLED_MARKER is written on success.
5. ModelStorageManager.isReady(modelId) checks all requiredFiles exist (and the marker for ZIP installs).

---

## 10. Engine Implementations

### ParakeetUnifiedEngine (experimental debug option)

English-only Parakeet Unified EN 0.6B uses the pinned sherpa-onnx INT8 streaming
export, about 663 MB across encoder, predictor, joiner and tokens. Every downloaded
file has a SHA-256 digest. The existing multilingual V3 remains the default.
Unified appears in debug builds pending Android device validation.

`StreamingSpeechEngine.newStream()` creates an utterance session. The session owns
the trained attention window and RNN-T state; `InferenceRepository` feeds PCM,
cleans accumulated hypotheses, and closes the session after Stop, a silence
boundary, failure or cancellation. Silence boundaries start fresh decoder state
for the next utterance without stopping the recording. Custom words and text
insertion use the recording preview and completion path. Suggestions fall back to the dictionary because
this engine does not collect acoustic alternatives.

`NemoLogMel` extracts unnormalized 128-bin Slaney log-mel frames from 16 kHz PCM.
The engine assembles 560 left-context, 56 center and 56 right-context frames,
pads missing feature frames with zero, then normalizes each feature over the whole
window. It decodes only the center frames and carries predictor state across
windows. Stop flushes the remaining center frames with right padding. The
1.12-second profile describes audio context, not a measured phone response time.
Sessions use `ALL_OPT` and up to four intra-op CPU threads. `ParakeetUnifiedEngine(intraOpThreads = N)` supports explicit 1–8 thread counts for benchmarks. `just benchmark-unified` runs the opt-in instrumented comparison on an arm64 device with Unified downloaded. Test audio lives only in the test APK.

Debug `OutspokeTrace` records monotonic timestamps, recording/utterance IDs, queued audio duration and age, encoder/window times, finalization duration, and editor insertion duration/acceptance. It excludes transcript and audio content. Use these to distinguish microphone shutdown, queued work and the final tail; thread tuning still needs real-device and sustained thermal measurements.

See [model research](parakeet-streaming-models.md) before changing export,
preprocessing, window dimensions or decoder behavior. Real-model tests opt in via
`-Ptest.unified.model.dir=/path/to/export`; private counting audio is supplied with
`-Ptest.counting.wav=/path/to/recording.wav` and stays outside the repository.

### ParakeetEngine (active)

NVIDIA Parakeet-TDT 0.6B v3, INT8 quantized, ~700 MB on disk.

3-ONNX pipeline:

| Session | File | Input -> Output |
|---|---|---|
| Preprocessor | nemo128.onnx | Raw PCM float32 -> 128-dim log-mel spectrogram |
| Encoder | encoder-model.int8.onnx | Spectrogram -> [B, 1024, T_enc] encoder features |
| Decoder/Joint | decoder_joint-model.int8.onnx | Encoder features + LSTM state -> [B, T_enc, T_tgt, 8198] logits |

- Tokenizer: SentencePiece-like vocabulary from vocab.json (1024 tokens + blank).
- Decoding: greedy TDT (Token-and-Duration Transducer). LSTM state is carried across strides.
- Logit tensor: first 1025 entries = token probabilities (vocab 1024 + blank); remainder = duration predictions. Only the greedy argmax is currently consumed.

### WhisperEngine (disabled - resource limits)

Targets whisper-large-v3-turbo INT8. Files: encoder_model_int8.onnx, decoder_model_merged_int8.onnx, tokenizer.json. 128 mel bins, 30 s window, 3000 frames. Merged decoder with use_cache_branch flag (autoregressive decoding with KV-cache).

### VoxtralEngine (disabled - resource limits)

Voxtral-Mini-4B-Realtime ONNX (~4 GB RAM requirement). Same Whisper-compatible log-mel preprocessing. Mistral decoder architecture with KV-cache.

---

## 11. Service and Lifecycle Management

### Binding lifecycle

    OutspokeInputMethodService.onCreate()
        -> bindService(InferenceService, BIND_AUTO_CREATE)

    OutspokeInputMethodService.onWindowShown()
        -> startService(InferenceService)   (foreground with type microphone when allowed)

    Switch to another keyboard
        -> OutspokeInputMethodService.onDestroy() -> unbindService()
           InferenceService stays started; the loaded model survives

Unbinding alone does NOT stop InferenceService. It unloads the model only under critical memory pressure.

### Engine reload on model change

    AppPreferences.selectedModelId (DataStore Flow)
        -> InferenceService observer
               -> acquire Mutex
               -> close old SpeechEngine
               -> SpeechEngineFactory.create(newModelId)
               -> engine.load(modelDir)
               -> emit EngineState.Ready
               -> release Mutex

### FileObserver integration

InferenceService watches <filesDir>/models/ for CLOSE_WRITE / MOVED_TO events. When a new model finishes downloading, the observer triggers a reload check without requiring an app restart.

---

## 12. State Machines

### Engine state

    Unloaded --load()--> Loading --success--> Ready
                             |                  |
                             +--failure--> Error(message)
                                                |
                                           retry load()--> Loading

### Model download state

    NotDownloaded --start--> Downloading(0..1) --complete--> Ready
                                  |                   |
                             cancel/error        SHA-256 fail
                                  v                   v
                            NotDownloaded        Corrupted --retry--> Downloading

### Keyboard UI state

    Idle --tap mic--> Listening --audio buffered--> Processing(partial)
     ^                    |                               |
     |              release mic                    recording stop
     |                    v                               v
     +--------------- Transcribing <---------- (final inference)
                           |
                      inject text
                           v
                         Idle

---

## 13. Build Configuration and ABI Splits

- compileSdk 36, minSdk 30 (Android 11+), targetSdk 36
- versionCode 11, versionName 0.3.1
- Kotlin 2.3.20, AGP 8.13.2, JVM target 11
- buildFeatures: compose = true, buildConfig = true
- Release: isMinifyEnabled = true, isShrinkResources = true, ProGuard enabled

**ABI splits:**

| ABI | Code offset | versionCode formula |
|---|---|---|
| armeabi-v7a | x1 | defaultVersionCode * 10 + 1 |
| arm64-v8a | x2 | defaultVersionCode * 10 + 2 |
| universal | x0 | defaultVersionCode * 10 + 0 |

dependenciesInfo { includeInApk = false; includeInBundle = false } - required for IzzyOnDroid / F-Droid reproducibility.

Repositories: Google, MavenCentral, Gradle Plugin Portal only.

---

## 13.5 Word-Correction Subsystem

Introduced in v0.2.0. Opt-in; disabled by default (`suggestion_bar_enabled = false`).

### Overview

After each dictation commit, a `SuggestionBar` chip row animates into view above the keyboard. Tapping a word in the transcription (tracked as `WordAtCursor` in `KeyboardUiState`) triggers a candidate query. Up to 5 ranked candidates are shown; tapping one replaces the word in the text field.

**Candidates are acoustic-first.** At decode time the Parakeet TDT decoder already computes the full 8198-way acoustic distribution at every step; the engine captures the top-K runner-up tokens per emission and, for low-confidence words, runs a bounded local beam over the word's frame range. The repository segments emissions into words and writes the per-word alternatives (word + length-normalised acoustic log-prob) to a bounded `AcousticCandidateCache` (§13.6). When the user queries a word, those alternatives are rescored with the ARPA language model and the surrounding context — the standard ASR n-best re-ranking combination — and the result is what the bar shows. Because the TDT decoder is lexicon-unconstrained (its beam can emit acoustically plausible strings that are not words), the acoustic alternatives are filtered to genuine dictionary words (case-insensitive, deduplicated case-insensitively) before rescoring; the dictionary (phonetic + edit distance) is a low-prior fallback for words with no usable acoustic evidence (manually typed, decoded before capture was active, or all acoustic alternatives filtered out as non-words).

### Language Packs

Each supported language requires two files, stored in `<filesDir>/suggestion_files/<tag>/`:

| File | Approximate size | Purpose |
|---|---|---|
| `dict_<tag>.txt` | ~2 MB | Frequency-sorted word list (`word\tlog10_freq` per line) |
| `lm_<tag>.arpa` | ~6 MB | ARPA language model (bigram packs; the parser also supports trigrams) for context re-ranking |

Files are downloaded on demand from `https://github.com/minburg/outspoke-data/releases/download/v2` — **the only external URL used at runtime besides the Hugging Face model download**. The URL is pinned to a specific release tag in `SuggestionFileManager.BASE_URL`; bump the tag there (and in the `outspoke-data` repository) whenever file format or content changes.

Supported languages (`SuggestionLanguage` enum):

| Tag | Language |
|---|---|
| `bg` | Bulgarian |
| `hr` | Croatian |
| `cs` | Czech |
| `da` | Danish |
| `nl` | Dutch |
| `en` | English |
| `et` | Estonian |
| `fi` | Finnish |
| `fr` | French |
| `de` | German |
| `el` | Greek |
| `hu` | Hungarian |
| `it` | Italian |
| `lv` | Latvian |
| `lt` | Lithuanian |
| `mt` | Maltese |
| `pl` | Polish |
| `pt` | Portuguese |
| `ro` | Romanian |
| `ru` | Russian |
| `sk` | Slovak |
| `sl` | Slovenian |
| `es` | Spanish |
| `sv` | Swedish |
| `uk` | Ukrainian |

### Correction Pipeline

    WordSuggestionProvider.getSuggestions(word, sentenceContext)
        |
        acoustic = acousticLookup(word)          ← InferenceRepository.getAcousticAlternatives(word)
                                                      (bounded acoustic cache, populated at decode time)
        |
        for each active language tag:
            WordCorrector.correct(word, leftContext, acoustic)
                |
                if acoustic is non-empty:
                    candidates = acoustic alternatives (top-K token swaps + local beam)
                                filtered to genuine dictionary words (case-insensitive)
                                + deduplicated case-insensitively (highest score wins)
                    if nothing survives the filter → dictionary fallback below
                else:
                    CandidateGenerator.getCandidates(word)      ← dictionary fallback
                        1. Phonetic index lookup (Kölner Phonetik for "de", Double Metaphone otherwise)
                        2. Damerau-Levenshtein sweep (distance ≤ 2, ±3 char length filter)
                        → up to 50 candidates
                |
                score = acousticLogProb + λ · ln(10) · lmLog10(candidate | leftContext)
                    acousticLogProb: length-normalised natural-log acoustic probability
                                      (dictionary fallback uses the fixed prior -2.0)
                    lmLog10: raw ARPA log10 with standard backoff (trigram → bigram → unigram)
                    λ = 0.5 — the acoustic term dominates; the LM breaks ties / fits context
                |
                → top 5 candidates (query word excluded)
        |
        merge candidates across languages (best rank wins on duplicate)
        → top 5 delivered on main thread via onSuggestions callback

### Key classes

| Class | Responsibility |
|---|---|
| `WordSuggestionProvider` | Public façade; manages `WordCorrector` instances per language; owns background `CoroutineScope`; `acousticLookup` wired to the `InferenceRepository` by the IME service |
| `WordCorrector` | Log-domain AM+LM rescoring (`acousticLogProb + λ·ln10·lmLog10`); lexicon-filters acoustic alternatives to dictionary words; dictionary fallback with fixed prior |
| `CandidateGenerator` | Loads dict file; phonetic index + edit-distance sweep (dictionary fallback); case-insensitive `isKnownWord` membership (lexicon filter) |
| `ArpaLanguageModel` | Loads ARPA file; trigram/bigram/unigram scoring with standard backoff, raw log10 |
| `WordAlternative` / `AcousticCandidateCache` | Per-word acoustic alternative (word + length-normalised log-prob); bounded thread-safe cache, cleared per recording session |
| `SuggestionFileManager` | On-disk path constants; `isLanguageReady()` check |
| `SuggestionFileDownloader` | OkHttp download with range-resume + SHA-256 verification; emits `SuggestionDownloadState` flow |
| `SuggestionBar` | Compose chip row; animated appear/disappear (300 ms ease-out / ease-in) |

### Invariants

- No data leaves the device once files are downloaded.
- `WordSuggestionProvider` loads only the languages explicitly selected by the user via `AppPreferences.suggestionBarLanguages`.
- Downloads come **only** from `github.com/minburg/outspoke-data`. Do not add other external hosts.
- The feature is a strict no-op (no background work, no memory overhead) when `suggestionBarEnabled = false`.
- Acoustic capture runs inside the decode loop (the logits are already in memory) — top-K token swaps are near-free; the local beam is gated on per-word confidence (< 0.6) and capped at 3 beams per chunk, so confident speech (the common case) adds no beam cost.
- The acoustic cache is process-local to the `InferenceRepository` (the long-lived bound service), bounded to 200 words (oldest evicted first), and cleared at the start of each recording session — RAM stays flat (tens of KB) and stale evidence never leaks across utterances.
- Rescoring is in the log domain (`acousticLogProb + λ·ln10·lmLog10`, no clamping): the acoustic term is the grounded signal and dominates; the LM differentiates.
- Acoustic alternatives are lexicon-filtered to genuine dictionary words (case-insensitive, deduplicated case-insensitively with the highest-scoring casing kept) before rescoring — the TDT decoder is lexicon-unconstrained and its beam can emit acoustically plausible strings that are not words. When nothing survives the filter, the dictionary fallback applies.

---

## 13.6 Acoustic Word-Alternative Capture

The decode-time half of the word-correction subsystem (see §13.5). Lives in the `inference` package because only the inference layer has the logits, the encoder features, and the decoder state.

### What is captured

`ParakeetEngine.decodeRange` records, for every non-blank emission: the argmax token, its log-softmax probability (the per-token confidence), and the top-3 tokens with their log-softmax probabilities — the acoustic runner-ups the greedy decode previously discarded. It also snapshots the LSTM state + previous token before each emitting joint call (`FrameState`), so a word beam can restart from the exact state the greedy decode had at the word's first token.

`InferenceRepository` (the Parakeet streaming path) segments each chunk's emissions into words — a SentencePiece word starts at a `▁`-initial token; the chunk's final segment is deferred across the chunk boundary because a word may continue. For every completed word it builds alternatives:

1. **Top-K token swaps** (every word, near-free): swap each runner-up token into each position, detokenise, keep the best length-normalised log-prob per resulting word.
2. **Bounded local beam** (gated + capped): for words with per-word confidence < 0.6 that are fully contained in the chunk, `ParakeetEngine.localWordBeam` re-runs the joint model over the word's frame range from the greedy state snapshot, branching on the top-3 tokens per step (plus blank, which ends the word), carrying the LSTM state per beam. At most 8 live states, 200 joint calls, 5 hypotheses. This finds alternative *token sequences* (a different word shape) that single-token swapping cannot reach.

The best ≤ 5 alternatives per word (the word itself excluded, case-insensitively) are written to the `AcousticCandidateCache`, keyed by the lower-cased word.

### Tuning knobs (constants, no structural change)

| Constant | Value | Meaning |
|---|---|---|
| `TOP_K_TOKENS` | 3 | runner-up tokens captured per emission |
| `BEAM_WIDTH` | 8 | live states in the local word beam |
| `MAX_BEAM_STEPS` | 200 | joint-call cap per word beam |
| `MAX_ALTERNATIVES` | 5 | word hypotheses returned per beam |
| `ACUSTIC_CONF_GATE` | 0.6 | per-word confidence below which the beam runs |
| `MAX_BEAMS_PER_CHUNK` | 3 | beam cap per decoded chunk |
| `MAX_CACHED_ALTERNATIVES` | 5 | alternatives cached per word |
| `AcousticCandidateCache.DEFAULT_CAPACITY` | 200 | distinct words held in the cache |
| `WordCorrector.LM_WEIGHT` (λ) | 0.5 | LM weight in the rescoring combination |
| `WordCorrector.DICT_ACOUSTIC_PRIOR` | -2.0 | acoustic prior for dictionary-fallback candidates |

### Invariants
- Capture never changes the decoded text — it only reads logits/state the decode loop already computed; the greedy path is untouched.
- A beam failure (logged, caught) degrades that word to token-swap candidates only; it never fails the transcription.
- The cache is read from IME threads via a single lock-protected map read; the writer is the inference worker.

---

## 14. Testing Strategy

All unit tests in app/src/test/kotlin/dev/brgr/outspoke/.
testOptions { unitTests.isReturnDefaultValues = true }.

| Test file | What it covers |
|---|---|
| audio/AudioChunkTest | AudioChunk data class, normalisation |
| audio/MicScorerTest | Mic quality scoring (HF fraction, percentile SNR) |
| audio/RMSVadFilterTest | Energy VAD onset/hangover logic |
| ime/TranscriptAlignerTest | All 3 layers of findNewContent |
| ime/TextInjectorTest | Legacy composing span management, trim resets |
| ime/TextInjectorRecordingTest | Single insertion with separators and selected-text replacement |
| ui/keyboard/RecordingTranscriptTest | Revisable utterances, silence boundaries and legacy trims |
| ui/keyboard/KeyboardViewModelRecordingTest | Preview-only recording, final insertion, failures and input-session cancellation |
| ime/ReplaceCursorWordTest | TextInjector.replaceCursorWord: replace at cursor, insert between words |
| ime/WordAtCursorTest | TextInjector.wordAtCursor: left-fragment extraction, sentence context |
| ime/WordSuggestionProviderTest | Acoustic-first facade: lookup, context extraction, main-thread delivery |
| ime/correction/WordCorrectorTest | Log-domain AM+LM rescoring, acoustic dominance, lexicon filter (non-word drop, case dedup), dictionary fallback |
| ime/correction/ArpaLanguageModelTest | Trigram scoring, ARPA backoff, production CRLF/literal-`\n` format |
| ime/correction/CorrectionRecallTest | recall@5 harness: noisy-audio ASR errors vs acoustic+LM pipeline (real model) |
| inference/InferenceRepositoryTest | Sliding-window buffering strategy (stride/trim constants) |
| inference/InferenceRepositoryPipelineTest | Full pipeline integration |
| inference/HumanSpeechPipelineTest | Human speech patterns: pauses, restarts, trailing-off, force-trim |
| inference/ShortUtterancePipelineTest | Short-utterance decode-context retry: no-lead-in word, VAD path, silence/noise no-Final (real model) |
| inference/ParakeetEngineRealAudioTest | One-shot engine path against WAV fixtures (real model) |
| inference/RealAudioPipelineTest | Full-stack streaming dictation, pipe-* no-loss/no-duplication battery (real model) |
| inference/StreamingParityTest | Streaming vs one-shot output parity across fixtures (real model) |
| inference/AcousticCaptureIntegrationTest | Top-K capture, local beam, cache population (real model) |
| inference/AcousticCandidateCacheTest | Bounded thread-safe acoustic cache: eviction, case, concurrency |
| inference/CleanTranscriptTest | All 8 post-processing steps |
| inference/CollapsePhrasesTest | Phrase-loop deduplication |
| inference/CollapseStuttersTest | Stutter collapse |
| inference/ScriptHallucinationTest | Script-consistency hallucination filter, 20% threshold boundary |
| inference/ParakeetAllLanguagesTest | Post-processing does not block or corrupt any Parakeet-supported language's script |
| inference/ParakeetLanguageTest | Language forcing threads into cleanTranscript (EN vs DE filler/disfluency handling) |
| inference/NumberNormaliserTest | Number-word to digit normalisation |
| inference/GrammarCorrectorTest | GrammarCorrector implementations and repository integration |
| inference/TranscriptResultTest | TranscriptResult sealed class |
| inference/WavReaderTest | WAV parsing, resampling, channel downmix |
| e2e/GoldenPathTest | End-to-end repository + TextInjector golden paths |

Test helpers: RealAudioTestUtils (model-dir resolution, WER), WavReader, FakeSpeechEngine, FakeInputConnection.

**CI behaviour:** the real-model tests resolve the model directory (-Dtest.model.dir > $OUTSPOKE_TEST_MODEL_DIR > ~/.cache/outspoke-test-model/parakeet-tdt-0.6b-v3/) and skip themselves (JUnit Assume) when it is absent — so the CI pipeline always runs the model-free suite, while a local machine with the model present runs everything.

Instrumented tests (device/emulator required) in app/src/androidTest/, run with ./gradlew connectedAndroidTest.

---

## 15. Extension Guide: Adding a New Engine

1. **ModelId**: add a new enum value with a stable storageDirName. Never change existing values.
2. **ModelRegistry**: add a ModelInfo (display name, DownloadSource, RemoteFile list with SHA-256 hashes, size estimate). Add to ModelRegistry.all only when confirmed to run within acceptable resource limits on real devices.
3. **New engine class** under inference/: implement SpeechEngine.
4. **SpeechEngineFactory**: add a branch for the new ModelId.

No changes required in InferenceRepository, InferenceService, TextInjector, or any IME class.

---

## 16. Key Conventions and Invariants

| Rule | Details |
|---|---|
| No field injection | Constructor injection only throughout |
| val over var | Avoid nullable types unless genuinely optional |
| No external telemetry | No analytics, crash reporters, or SDKs that phone home |
| Model storage | <filesDir>/models/<storageDirName>/ - no external storage permission |
| SHA-256 verification | Required for all downloaded model files; add hashes to RemoteFile entries |
| storageDirName immutability | Changing it breaks existing installations |
| WindowTrimmed handling | RecordingTranscript preserves the dropped prefix in memory and reconciles the retained overlap; no editor write |
| Final(isUtteranceBoundary=true) | Do NOT stop audio capture - used for mid-session sentence boundaries in long dictation |
| InferenceService stop | Unbinding does not stop the service; shuts down only when explicitly stopped or after the 30 s keyboard-hidden timer |
| Kotlin code style | kotlin.code.style=official |

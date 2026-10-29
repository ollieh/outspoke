package dev.brgr.outspoke.inference

import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ComponentCallbacks2
import android.content.pm.ServiceInfo
import android.content.res.Configuration
import android.os.*
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import dev.brgr.outspoke.R
import dev.brgr.outspoke.settings.model.ModelId
import dev.brgr.outspoke.settings.model.ModelRegistry
import dev.brgr.outspoke.settings.model.ModelStorageManager
import dev.brgr.outspoke.settings.preferences.AppPreferences
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

private const val TAG = "InferenceService"
private const val CHANNEL_ID = "outspoke_inference"
private const val NOTIFICATION_ID = 1001

/**
 * A foreground [LifecycleService] that owns the [SpeechEngine] lifecycle.
 *
 * The IME binds to it and also starts it, so the service and its loaded model outlive
 * the IME: switching to another keyboard destroys the IME and unbinds, and the started
 * service keeps the model warm for the next time Outspoke opens.
 *
 * Responsibilities:
 *  - Show a persistent low-priority notification so Android keeps the process alive
 *    while the keyboard is active
 *  - Observe [AppPreferences.selectedModelId] and load the appropriate [SpeechEngine]
 *    via [SpeechEngineFactory] whenever the selection changes
 *  - Watch the `models/` directory for file-system changes so the engine auto-reloads
 *    after a download completes and auto-unloads when model files are deleted
 *  - Expose [InferenceRepository] to bound clients (the IME) via [InferenceBinder]
 *  - Cleanly close the engine on [onDestroy]
 */
class InferenceService : LifecycleService() {

    /** Currently active grammar corrector. Tied to the service lifecycle. */
    private val grammarCorrector: GrammarCorrector = NoOpGrammarCorrector

    /** The currently loaded engine, or `null` while loading / unloaded. */
    @Volatile
    private var currentEngine: SpeechEngine? = null

    /** Repository wrapping [currentEngine]. Rebuilt each time the engine changes. */
    @Volatile
    private var currentRepository: InferenceRepository? = null

    /**
     * The model id currently selected (or being loaded). Tracked so [reloadIfNeeded] can
     * reload exactly the same model after a memory-pressure unload without re-reading
     * preferences.
     */
    @Volatile
    private var currentModelId: ModelId? = null
    private var loadedModelId: ModelId? = null

    /**
     * `true` after the engine was proactively closed under memory pressure (see
     * [memoryCallback]). The next [reloadIfNeeded] call reloads the model and clears it.
     */
    @Volatile
    private var memoryUnloaded: Boolean = false

    /** Mutex preventing concurrent [reloadForModel] calls from racing. */
    private val engineLoadMutex = Mutex()

    private val _engineState = MutableStateFlow<EngineState>(EngineState.Loading)

    /** Observable loading / runtime state observed by the IME. */
    val engineState: StateFlow<EngineState> = _engineState.asStateFlow()

    inner class InferenceBinder : Binder() {
        /** Returns the active [InferenceRepository], or `null` while the engine is loading. */
        fun getRepository(): InferenceRepository? = currentRepository
        fun getEngineState(): StateFlow<EngineState> = engineState

        /**
         * Reload the previously-selected model if it was closed under memory pressure.
         *
         * Called by the IME on [android.inputmethodservice.InputMethodService.onWindowShown]
         * (and as a safety net from the keyboard VM when the user presses record while the
         * engine is Unloaded). No-op when the engine is still loaded or already loading.
         */
        fun reloadIfNeeded() {
            if (!memoryUnloaded) return
            lifecycleScope.launch(Dispatchers.Default) {
                val modelId = currentModelId
                    ?: AppPreferences(applicationContext).selectedModelId.first()
                Log.i(TAG, "reloadIfNeeded() - reloading $modelId after memory-pressure unload")
                reloadForModel(modelId, reason = "reload after memory-pressure unload")
            }
        }
    }

    private val binder = InferenceBinder()

    override fun onBind(intent: android.content.Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    override fun onStartCommand(intent: android.content.Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        try {
            ServiceCompat.startForeground(
                this, NOTIFICATION_ID,
                buildNotification(if (_engineState.value == EngineState.Ready)
                    getString(R.string.notif_engine_ready, ModelRegistry[currentModelId ?: ModelId.DEFAULT].displayName)
                else getString(R.string.notif_engine_loading)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
            RecordingTrace.log { "service start foreground=true startId=$startId engine=${_engineState.value.javaClass.simpleName}" }
        } catch (e: SecurityException) {
            // Stay started anyway: a started background service survives the IME unbinding
            // when the user switches keyboards, until Android stops idle background services.
            Log.w(TAG, "Foreground promotion unavailable; staying started in the background", e)
            RecordingTrace.log { "service start foreground=false startId=$startId error=${e.javaClass.simpleName}" }
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Foreground promotion unavailable; staying started in the background", e)
            RecordingTrace.log { "service start foreground=false startId=$startId error=${e.javaClass.simpleName}" }
        }
        return START_NOT_STICKY
    }

    override fun onCreate() {
        super.onCreate()
        RecordingTrace.log { "service create" }

        createNotificationChannel()
        try {
            ServiceCompat.startForeground(
                this,
                NOTIFICATION_ID,
                buildNotification(getString(R.string.notif_engine_loading)),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } catch (e: SecurityException) {
            // startForeground() is rejected when the service is started via the IME binding
            // (no Activity in the foreground).  We continue running as a plain bound service -
            // no persistent notification, but the engine loads and the keyboard stays functional.
            // The notification will appear the next time the user opens the companion Activity.
            Log.w(TAG, "startForeground rejected - running as bound service without notification", e)
        } catch (e: IllegalStateException) {
            Log.w(TAG, "Foreground start unavailable - running as bound service", e)
        }

        // Log device and memory info at service startup
        val am = getSystemService(ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo().also { am?.getMemoryInfo(it) }
        Log.i(
            TAG,
            "Service created on device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})"
        )
        Log.i(
            TAG,
            "Total RAM: ${memInfo.totalMem / (1024 * 1024)} MB, Avail: ${memInfo.availMem / (1024 * 1024)} MB, LowMem: ${memInfo.lowMemory}"
        )

        Log.d(TAG, "Service created - observing selected model preference")

        // Observe the selected model preference and reload the engine whenever it changes.
        lifecycleScope.launch(Dispatchers.Default) {
            AppPreferences(applicationContext).selectedModelId.distinctUntilChanged().collect { modelId ->
                currentModelId = modelId
                memoryUnloaded = false  // a model change supersedes any memory-pressure unload
                reloadForModel(modelId, reason = "selected model")
            }
        }

        // Keep the model warm across keyboard hide/show, app switches and keyboard
        // switches (the IME starts this service, so it outlives the IME's binding). To
        // avoid being OOM-killed with ~700 MB resident, close the engine when the OS
        // signals critical memory pressure and let the IME reload it on the next
        // onWindowShown via InferenceBinder.reloadIfNeeded().
        registerMemoryCallback()

        // Watch the models/root directory to detect external changes (e.g. download complete).
        startModelWatcher()
    }

    /**
     * Registered on the application context in [onCreate]; closes the loaded engine when
     * the OS reports critical memory pressure so the ~700 MB Parakeet model is reclaimed
     * cooperatively instead of via an OOM kill. Unregistered in [onDestroy].
     *
     * Only [ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL] and [onLowMemory] trigger an
     * unload — the levels at which the whole system is genuinely short of memory. The
     * background levels (UI_HIDDEN, BACKGROUND, MODERATE, COMPLETE) and RUNNING_LOW are
     * routine for a process whose keyboard was just hidden or switched away; unloading on
     * them meant a 4–5 s reload nearly every time the user came back to Outspoke.
     */
    private var memoryCallback: ComponentCallbacks2? = null

    @Suppress("DEPRECATION", "OVERRIDE_DEPRECATION")
    private fun registerMemoryCallback() {
        // Trim-memory levels are not strictly monotonic in severity:
        //   UI_HIDDEN(20), BACKGROUND(40)        — routine backgrounding, NOT real pressure.
        //   MODERATE(60), COMPLETE(80)          — position in the cached-process LRU list.
        //   RUNNING_LOW(10), RUNNING_CRITICAL(15) — process in foreground + pressure.
        // Only RUNNING_CRITICAL (the system is about to kill background processes) and
        // onLowMemory unload. Reload is handled lazily by InferenceBinder.reloadIfNeeded()
        // on the next onWindowShown.
        val cb = object : ComponentCallbacks2 {
            override fun onTrimMemory(level: Int) {
                val unload = level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL
                RecordingTrace.log { "service trim memory level=$level unload=$unload engine=${_engineState.value.javaClass.simpleName}" }
                if (unload) unloadDueToMemoryPressure("trim memory level=$level")
            }

            override fun onLowMemory() {
                RecordingTrace.log { "service low memory unload=true" }
                unloadDueToMemoryPressure("low memory")
            }

            override fun onConfigurationChanged(newConfig: Configuration) = Unit
        }
        memoryCallback = cb
        applicationContext.registerComponentCallbacks(cb)
    }

    /**
     * Closes the currently loaded engine cooperatively under memory pressure and marks the
     * engine as memory-unloaded so [InferenceBinder.reloadIfNeeded] reloads it on next use.
     * No-op if the engine is not currently loaded / ready (e.g. still loading or already
     * unloaded). Serialized with [reloadForModel] via [engineLoadMutex].
     */
    private fun unloadDueToMemoryPressure(reason: String) {
        if (currentEngine == null || _engineState.value != EngineState.Ready) return
        Log.w(TAG, "Memory pressure ($reason) - cooperatively closing engine to free RAM")
        lifecycleScope.launch(Dispatchers.Default) {
            engineLoadMutex.withLock {
                if (currentEngine == null) return@withLock
                if (currentRepository?.closeEngine(onlyIfIdle = true) == false) {
                    Log.i(TAG, "Keeping model loaded until the active recording finishes")
                    RecordingTrace.log { "engine unload skipped reason=$reason recording=true" }
                    return@withLock
                }
                RecordingTrace.log { "engine unloaded model=$loadedModelId reason=$reason" }
                currentEngine = null
                currentRepository = null
                memoryUnloaded = true
                _engineState.value = EngineState.Unloaded
                updateNotification(getString(R.string.notif_model_not_downloaded))
                logMemoryUsage()
            }
        }
    }

    override fun onDestroy() {
        RecordingTrace.log { "service destroy engine=${_engineState.value.javaClass.simpleName} model=$loadedModelId" }
        modelFileObserver?.stopWatching()
        modelFileObserver = null
        currentWatchDir = null
        memoryCallback?.let { applicationContext.unregisterComponentCallbacks(it) }
        memoryCallback = null
        super.onDestroy()
        CoroutineScope(Dispatchers.Default).launch {
            engineLoadMutex.withLock {
                val hadEngine = currentRepository != null
                currentRepository?.closeEngine()
                currentEngine = null
                currentRepository = null
                grammarCorrector.close()
                Log.d(TAG, "Service destroyed - engine closed")
                if (hadEngine) RecordingTrace.log { "engine unloaded model=$loadedModelId reason=service destroyed" }
            }
        }
    }

    /**
     * Closes any existing engine and loads a fresh one for [modelId].
     * Always called inside [engineLoadMutex] to serialise concurrent invocations.
     * [reason] only labels the trace log.
     */
    private suspend fun reloadForModel(modelId: ModelId, reason: String) {
        engineLoadMutex.withLock {
            if (currentEngine != null && loadedModelId == modelId &&
                ModelStorageManager.isModelReady(applicationContext, modelId)) {
                RecordingTrace.log { "engine load skipped model=$modelId reason=$reason alreadyLoaded=true" }
                return@withLock
            }
            // We are (re)loading on purpose, so any prior memory-pressure unload no
            // longer applies — clear the flag so reloadIfNeeded() does not re-trigger.
            memoryUnloaded = false

            // Close the current engine before replacing it.
            _engineState.value = EngineState.Loading
            if (currentRepository != null) {
                RecordingTrace.log { "engine unloaded model=$loadedModelId reason=replaced ($reason)" }
            }
            currentRepository?.closeEngine()
            currentEngine = null
            currentRepository = null

            if (!ModelStorageManager.isModelReady(applicationContext, modelId)) {
                Log.w(TAG, "Model $modelId files not present - engine stays Unloaded")
                RecordingTrace.log { "engine load skipped model=$modelId reason=$reason filesPresent=false" }
                _engineState.value = EngineState.Unloaded
                updateNotification(getString(R.string.notif_model_not_downloaded))
                return
            }

            // Log model file size and available memory before loading
            val modelDir = ModelStorageManager.getModelDir(applicationContext, modelId)
            val modelSizeMB = modelDir.walkTopDown().filter { it.isFile }.sumOf { it.length() } / (1024 * 1024)
            val am = getSystemService(ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo().also { am?.getMemoryInfo(it) }
            Log.i(TAG, "Preparing to load model $modelId, modelDir=${modelDir.path}, size=${modelSizeMB}MB")
            Log.i(TAG, "Available RAM: ${memInfo.availMem / (1024 * 1024)} MB, LowMem: ${memInfo.lowMemory}")
            if (modelSizeMB > 500 && memInfo.availMem < modelSizeMB * 2 * 1024 * 1024) {
                Log.w(
                    TAG,
                    "Device RAM may be insufficient for large model $modelId (model: ${modelSizeMB}MB, avail: ${memInfo.availMem / (1024 * 1024)}MB)"
                )
            }

            _engineState.value = EngineState.Loading
            updateNotification(getString(R.string.notif_engine_loading))

            val startTime = System.currentTimeMillis()
            RecordingTrace.log { "engine load begin model=$modelId reason=$reason" }
            var loadingEngine: SpeechEngine? = null
            try {
                val engine = SpeechEngineFactory.create(modelId)
                loadingEngine = engine
                engine.load(modelDir)

                // Apply forced-language preference so post-processing uses the correct locale.
                // For Parakeet TDT, this does not affect ONNX inference (no language tensor in
                // the current export) but controls filler removal and number normalisation.
                val lang = AppPreferences(applicationContext).forcedLanguage.first()
                if (lang != null) {
                    engine.setLanguage(lang)
                    Log.d(TAG, "Forced language '$lang' applied to engine for $modelId")
                }

                loadedModelId = modelId
                currentEngine = engine
                currentRepository = InferenceRepository(engine, grammarCorrector)
                loadingEngine = null
                _engineState.value = EngineState.Ready
                updateNotification(getString(R.string.notif_engine_ready, ModelRegistry[modelId].displayName))
                val elapsed = System.currentTimeMillis() - startTime
                Log.i(TAG, "Engine loaded successfully for $modelId in ${elapsed}ms")
                RecordingTrace.log { "engine load end model=$modelId reason=$reason durationMs=$elapsed ok=true" }
                logMemoryUsage()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                val msg = e.localizedMessage ?: "Unknown error"
                Log.e(TAG, "Engine load failed for $modelId: $msg", e)
                RecordingTrace.log { "engine load end model=$modelId reason=$reason durationMs=${System.currentTimeMillis() - startTime} ok=false" }
                _engineState.value = EngineState.Error(msg)
                updateNotification(getString(R.string.notif_engine_failed, msg))
                logMemoryUsage()
            } finally {
                loadingEngine?.close()
            }
        }
    }

    /**
     * Watches the `models/` root directory so we detect both deletion and installation of
     * model files for any model - triggering an engine reload for the selected model.
     *
     * The observer is only recreated when the watched directory actually changes.
     */
    private var modelFileObserver: FileObserver? = null
    private var currentWatchDir: File? = null

    private fun startModelWatcher() {
        val context = applicationContext
        // Watch the models/ root; fall back to filesDir if models/ doesn't exist yet.
        val modelsRoot = ModelStorageManager.getModelsRoot(context).also { it.mkdirs() }
        val watchDir = if (modelsRoot.exists()) modelsRoot else context.filesDir

        if (watchDir == currentWatchDir) return

        modelFileObserver?.stopWatching()
        currentWatchDir = watchDir

        modelFileObserver = object : FileObserver(watchDir, ALL_EVENTS) {
            override fun onEvent(event: Int, path: String?) {
                val mask = event and ALL_EVENTS
                if (mask and (DELETE or DELETE_SELF or MOVED_FROM or
                            CREATE or MOVED_TO or CLOSE_WRITE) != 0
                ) {
                    onModelDirectoryChanged()
                }
            }
        }.also { it.startWatching() }

        Log.d(TAG, "FileObserver started on: $watchDir")
    }

    /**
     * Called (on a FileObserver background thread) whenever anything changes under the
     * `models/` directory. Checks whether the selected model's readiness changed and
     * reloads or unloads the engine accordingly.
     */
    private fun onModelDirectoryChanged() {
        lifecycleScope.launch(Dispatchers.Default) {
            val context = applicationContext
            val selectedModelId = AppPreferences(context).selectedModelId.first()
            val isReady = ModelStorageManager.isModelReady(context, selectedModelId)
            val currentState = _engineState.value

            when {
                !isReady && currentState == EngineState.Ready -> {
                    Log.w(TAG, "Selected model files removed - unloading engine")
                    reloadForModel(selectedModelId, reason = "model files removed") // will detect !isReady and set Unloaded
                    startModelWatcher()
                }

                isReady && currentState == EngineState.Unloaded -> {
                    Log.d(TAG, "Selected model files appeared - loading engine")
                    reloadForModel(selectedModelId, reason = "model files appeared")
                    startModelWatcher()
                }
            }
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notif_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.notif_channel_desc)
            setShowBadge(false)
        }
        getSystemService(NotificationManager::class.java)?.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Outspoke")
            .setContentText(text)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()

    private fun updateNotification(text: String) {
        val nm = getSystemService(NotificationManager::class.java) ?: return
        nm.notify(NOTIFICATION_ID, buildNotification(text))
    }

    private fun logMemoryUsage() {
        val runtime = Runtime.getRuntime()
        val usedMemMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val maxMemMB = runtime.maxMemory() / (1024 * 1024)
        Log.i(TAG, "App memory usage: used=${usedMemMB}MB, max=${maxMemMB}MB")
        val debugMem = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugMem)
        Log.i(
            TAG,
            "Debug memory: dalvik=${debugMem.dalvikPrivateDirty}KB, native=${debugMem.nativePrivateDirty}KB, totalPss=${debugMem.totalPss}KB"
        )
    }
}

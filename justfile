set shell := ["nu", "-c"]

# Compare 1, 2 and 4 CPU threads on a connected device with Unified downloaded.
benchmark-unified:
    ^./gradlew assembleDebug assembleDebugAndroidTest
    ^adb install -r app/build/outputs/apk/debug/app-arm64-v8a-debug.apk
    ^adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
    ^adb shell am instrument -w -e class dev.brgr.outspoke.UnifiedLatencyBenchmarkTest -e unifiedBenchmark true dev.brgr.outspoke.test/androidx.test.runner.AndroidJUnitRunner
    ^adb logcat -d -s 'OutspokeBenchmark:I' '*:S'

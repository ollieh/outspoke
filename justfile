set shell := ["nu", "-c"]

# compileSdk 37 isn't in a stock Nix Android SDK, so compose one with androidenv.
sdk_expr := 'let pkgs = import (builtins.getFlake "nixpkgs") { config = { allowUnfree = true; android_sdk.accept_license = true; }; }; in (pkgs.androidenv.composeAndroidPackages { platformVersions = [ "37" "36" ]; buildToolsVersions = [ "36.0.0" "35.0.0" ]; includeNDK = false; includeEmulator = false; includeSystemImages = false; }).androidsdk'

# Link the Nix-built Android SDK at .android-sdk (also a GC root).
android-sdk:
    if not ('.android-sdk' | path exists) { ^nix build --impure --out-link .android-sdk --expr '{{sdk_expr}}' }

# Run Gradle with that SDK and JDK 17, e.g. `just gradle testDebugUnitTest`.
gradle *args: android-sdk
    with-env { ANDROID_HOME: ($env.PWD | path join .android-sdk libexec android-sdk) } { ^nix shell nixpkgs#jdk17 -c ./gradlew {{args}} }

# Update the phone in place; ~/.android/debug.keystore signed its install, so its data is kept.
install: (gradle "assembleDebug")
    ^adb install -r app/build/outputs/apk/debug/app-arm64-v8a-debug.apk

# Compare V3's live and single-pass speed at 2/4/6 threads on a connected device with V3 downloaded.
benchmark-v3: (gradle "assembleDebug" "assembleDebugAndroidTest")
    ^adb install -r app/build/outputs/apk/debug/app-arm64-v8a-debug.apk
    ^adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
    ^adb shell am instrument -w -e class dev.brgr.outspoke.V3LatencyBenchmarkTest -e v3Benchmark true dev.brgr.outspoke.test/androidx.test.runner.AndroidJUnitRunner
    ^adb logcat -d -s 'OutspokeBenchmark:I' '*:S'

# Compare 1, 2 and 4 CPU threads on a connected device with Unified downloaded.
benchmark-unified: (gradle "assembleDebug" "assembleDebugAndroidTest")
    ^adb install -r app/build/outputs/apk/debug/app-arm64-v8a-debug.apk
    ^adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
    ^adb shell am instrument -w -e class dev.brgr.outspoke.UnifiedLatencyBenchmarkTest -e unifiedBenchmark true dev.brgr.outspoke.test/androidx.test.runner.AndroidJUnitRunner
    ^adb logcat -d -s 'OutspokeBenchmark:I' '*:S'

# Unified streaming latency and a Handy-style recording flow

Investigated on 2026-09-30 against the working tree before the changes below. The reported model is
Parakeet Unified EN. The intended product direction is Handy for Android: show a
revisable transcript in the keyboard, then insert the completed recording once.

## Implemented this pass

The keyboard now owns the recording preview. Partials can revise the active
utterance, and silence-boundary finals accumulate without writing to the editor.
Stop keeps that preview visible with a “Finishing…” label. Successful flow
completion inserts all finalized utterances with one editor commit, including
the configured separators. Changing input sessions or clearing the field cancels
pending insertion. Decode and insertion failures retain the preview.

Unified now uses `ALL_OPT`, sequential execution and up to four intra-op CPU
threads. The trained window and final-tail algorithm are unchanged. These settings
improved the desktop replay below; Pixel speed and thermal behavior remain unmeasured.
Unified remains experimental pending device validation, and this pass does not
introduce transcribe.cpp.

Debug `OutspokeTrace` logs include monotonic timestamps, recording/utterance IDs,
captured and consumed audio, queue age/duration, encoder and final-tail timing,
and editor insertion timing/acceptance. Audio and transcript contents are excluded.

With Java, Android SDK and `adb` configured, connect an arm64 device with USB
debugging enabled and run `just benchmark-unified`. The recipe builds and installs
the debug app and test APK, then compares 1/2/4 threads using the downloaded
Unified model. Download Unified in the debug app before running it. Test-only WAV
fixtures cover short speech and 18 seconds of numbered sentences. The benchmark
warms each configuration, reports streaming work, finalization and real-time
factor to `OutspokeBenchmark`, and checks recognized words. Its unpaced replay
measures compute capacity; live recording traces are needed for Stop-to-insert,
microphone shutdown and sustained thermal behavior.

Validation: full JVM suite and debug app/test APK builds passed, including the
available real Unified audio fixtures. Instrumented tests and the benchmark have
not run because no Android device is connected.

## Findings

There was a reproducible performance problem before Android text
insertion entered the picture. The baseline Unified engine was close to or slower
than real time on this Linux host, and short recordings trigger substantial
inference at Stop. A slower phone can amplify both costs. No Android device was
connected, so the exact contribution of each cost on the Pixel 10 Pro remains
unmeasured.

The tests used the actual pinned INT8 model, the production `ParakeetUnifiedEngine`
and checked-in WAV fixtures. This host exposes eight AMD EPYC Genoa CPU cores.
These are host measurements, not Framework 16 or Pixel measurements. Audio was
replayed as 30 ms inputs without wall-clock pacing; the figures measure the work
that must fit between microphone arrivals.

| Fixture | Audio duration | Baseline streaming work | Baseline Stop flush | Encoder passes at Stop |
| --- | ---: | ---: | ---: | ---: |
| `single-yes.wav` | 0.720 s | 0.002–0.048 s | 1.17–1.37 s | 2 |
| `hello-world.wav` | 0.952 s | 0.003–0.015 s | 1.13–1.28 s | 2 |
| `natural-counting.wav` | 18.369 s | 19.772 s | 1.340 s | 2 |

The long recording performed 31 live encoder passes. Typical passes took about
0.63–0.66 s while advancing only 0.56 s of audio. Total computation was 21.111 s
for 18.369 s of input, a real-time factor of 1.149. A temporary replay assertion
requiring a factor below 1 failed with that value. A separate initial run produced
similar results. The short examples contain no live encoder pass before Stop.

### Why short speech is expensive

The export requires a fixed window of 560 left-context, 56 center and 56
right-context feature frames. At 10 ms per frame, each encoder call evaluates
6.72 s of features to advance 0.56 s of speech. Missing context is zero padded,
including for a subsecond recording; the tensor still has the full size.

`addFrames()` waits for center plus right context before decoding. Startup needs
about 1.12 s of input, plus feature framing and computation. At Stop, `finish()`
decodes the remaining frames in batches of at most 56. The 72-frame "yes" and
95-frame "hello world" recordings therefore each need two full encoder calls.
Steady streaming also retains 56 right-context frames plus a residual center
chunk, so finishing commonly needs two calls even without a queue backlog.

See [window scheduling and finalization](../app/src/main/kotlin/outspoke/inference/ParakeetUnifiedEngine.kt#L124)
and [encoder invocation](../app/src/main/kotlin/outspoke/inference/ParakeetUnifiedEngine.kt#L151).
This follows the inspected sherpa center-only scheduling approach. It is expensive;
the two calls alone do not establish an incorrect implementation.

Unified already bypasses the old 30-second sliding-window path. Both our export
and Handy's runtime use buffered streaming with context. Streaming does not imply
that each arriving piece of speech is encoded only once.

### Why a recording can fall further behind

The repository buffers microphone input with `Channel.UNLIMITED` and processes
it in order. Stop ends capture, then inference drains accepted audio before
finalizing. When computing a 0.56-second stride costs more than 0.56 seconds,
that queue grows. Waiting at Stop includes unfinished inference plus the final
tail work. This is consistent with the user's long-recording explanation, but
the subsecond replay demonstrates a separate fixed finalization cost.

See [stream collection](../app/src/main/kotlin/outspoke/inference/InferenceRepository.kt#L1987)
and [Stop handling](../app/src/main/kotlin/outspoke/ui/keyboard/KeyboardViewModel.kt#L709).
Stopping must continue to preserve accepted audio. Cancelling decoding or dropping
queued microphone frames would hide delay by losing speech.

### Runtime settings help, but are not the complete fix

The baseline sessions used two intra-op threads, sequential execution and
`BASIC_OPT`. Profiling the encoder showed only `CPUExecutionProvider`.
Quantized matrix multiplication and convolution accounted for roughly 71% of
recorded basic-optimization operator time. Neither a GPU nor the Pixel's AI
accelerator is requested by this code.

With identical fixed-size encoder inputs, ONNX profiling enabled, and four calls
per configuration, mean encoder times were:

| Settings | Mean encoder call |
| --- | ---: |
| Basic optimization, 2 threads | 0.697 s |
| Full optimization, 2 threads | 0.643 s |
| Full optimization, 4 threads | 0.485 s |
| Full optimization, 1 thread | 0.928 s |

A further real-audio replay replaced sessions inside the temporary test with
four-thread `ALL_OPT` sessions, leaving production source untouched. For the long
fixture, streaming work fell to 14.857 s and finalization to 0.962 s. Total work
was 15.819 s, a factor of 0.861, and the same real-time assertion passed.
Short finalization remained around 0.96–1.02 s. All fixtures returned nonempty
text. This experiment establishes a speed improvement on this host; a full
transcript-parity suite, ARM measurements, and thermal testing are still required
to validate the new defaults on Android.

## Differences from Handy

Handy's transcription manager exposes a stable prefix and a tentative suffix to
its overlay, then finalizes the stream on Stop. Its action handler pastes the
finished text afterward. It also queues incoming audio ahead of finalization,
so Handy can wait if its runtime falls behind. Its immediate-looking paste is
observed behavior on the user's laptop, not a zero-cost finalization guarantee.
[Transcription manager](https://github.com/cjpais/Handy/blob/ba10ce1943ef34e93c09494027fc0b9ced2e8a44/src-tauri/src/managers/transcription.rs),
[recording action](https://github.com/cjpais/Handy/blob/ba10ce1943ef34e93c09494027fc0b9ced2e8a44/src-tauri/src/actions.rs).

Before this pass, Outspoke sent each `Partial` to both the keyboard and `TextInjector.setPartial()`.
That injector permanently froze words outside its six-word mutable tail.
Silence-boundary `Final` results also committed to the editor while recording
continued. Our Unified stream appends decoded tokens; it has no separate
speculative suffix/state rollback equivalent to Handy's general preview API.
Keeping all preview text inside the keyboard removes premature editor commits
and allows later revisions without Android composing-span alignment.

Handy uses the GGUF/transcribe.cpp implementation of Unified. Outspoke uses a
separate ONNX export, Kotlin feature extraction and ONNX predictor/joiner calls.
The model family is shared, but the runtime, quantization and scheduling differ.
transcribe.cpp uses variable stride and folds retained right context plus the
remaining tail into a final encoder pass. Its default Unified context is also
different from our selected 1.12-second profile.
[Unified runtime documentation](https://github.com/handy-computer/transcribe.cpp/blob/4807edaf210d0d7e8a6f7fb2a44b65966a2797f0/docs/models/parakeet-unified-en-0.6b.md),
[finalization implementation](https://github.com/handy-computer/transcribe.cpp/blob/4807edaf210d0d7e8a6f7fb2a44b65966a2797f0/src/arch/parakeet/model.cpp#L3180).

This difference is a reason to benchmark that runtime on Android. It is not
permission to decode extra right-context frames with our current graph without
checking the attention mask, normalization and predictor progression. Blindly
switching to a shorter-lookahead export can also worsen throughput by increasing
the number of expensive context re-encodes per second.

## Investigation plan and remaining device work

1. **Establish Pixel measurements.** Extend the existing debug `RecordingTrace`
   with monotonic timestamps, recording and utterance IDs, captured/consumed
   samples, oldest queued audio age, encoder duration and pending center/right
   frames. Measure tap to capture-end, capture-end to queue-empty, final encoder
   work, and final-result to editor-commit separately. Record runtime settings.
   Keep audio and transcript contents out of these logs. Replay the same short,
   10-second and 60-second recordings with and without pauses, cold and warm, then
   repeat sustained dictation to detect thermal slowdown.

2. **Adopt preview followed by one insertion.** Add recording-owned transcript
   state in `KeyboardViewModel`. Partials update the preview; utterance-boundary
   finals accumulate completed text in that same recording state. Neither writes
   to the editor. After successful recording flow completion, combine finalized
   utterances and insert once using the existing separator and trailing-space
   preferences. Trigger suggestions and automatic keyboard return after that
   insertion. Preserve the preview during finalization instead of replacing it
   with a textless `Transcribing` state. Test multiple silence boundaries, fast
   repeated recordings, punctuation, selected-text replacement and failure.

   Bind the pending result to the originating editor/session. On field change,
   cancellation or clear, never paste old speech into a new target. The existing
   `commitPartialAndStop()` input-finish behavior also needs changing because it
   explicitly commits a partial today. Define failure recovery so the preview is
   retained or discarded deliberately and is never silently committed twice.

3. **Tune and benchmark the current runtime first.** Compare full graph
   optimization and sensible CPU thread counts on the Pixel with the real
   transcript fixtures. Tune the encoder independently from the smaller
   predictor/joiner if useful. Measure end-to-end latency, accuracy and sustained
   throughput. The desktop result makes this an inexpensive first experiment,
   but it does not establish the best thread count for heterogeneous phone cores.

4. **Benchmark Handy's runtime through a small JNI bridge.** Compare pinned
   transcribe.cpp Unified against the tuned ONNX path on identical Pixel audio.
   Exercise its trained window configurations and final-tail behavior. If it
   materially improves accuracy/throughput/finalization, adopt it behind
   `StreamingSpeechEngine`; otherwise keep ONNX and address the export/scheduling
   bottleneck with parity tests. See [Android runtime options](unified-android-runtime-options.md)
   for build and binding evidence. Do not promise GPU acceleration before
   validating the actual Android backend and model operators.

5. **Promote Unified for English after Android validation.** The experimental
   label and debug-only listing describe our unfinished device validation, not an
   experimental NVIDIA model. Handy already recommends Unified for English.
   Remove the label/debug gate when accuracy and sustained speed meet the product
   requirements, and make Unified the recommended English choice. Keep V3
   available for its multilingual coverage and existing users. Unified is
   English-only, so replacing all users' selected models would be inappropriate.
   [Handy 0.9 release notes](https://github.com/cjpais/Handy/blob/main/src/content/release-notes/0.9.0.md),
   [current registry gate](../app/src/main/kotlin/outspoke/settings/model/ModelRegistry.kt#L98).

Suggested validation goals, to confirm on the Pixel: responsive Stop feedback,
zero editor writes before completion, exactly one final insertion, sustained
real-time factor below 1 with headroom, no growing queue during a minute of
speech, and a warm Stop-to-insert target below 500 ms when inference is caught
up. The current two-pass ONNX flush misses that latency target on this host.

## Investigation artifacts

The disposable test was run with `:app:testDebugUnitTest --tests
dev.brgr.outspoke.inference.UnifiedLatencyProbeTest`. Its real-time assertion failed
with production sessions and passed with the four-thread/full-optimization
session substitution. The test was removed from app test sources afterward.
The investigation itself did not modify production behavior; the subsequent
implementation is described above.

Local diagnostic snapshots are in `/tmp/UnifiedLatencyProbeTest.kt`,
`/tmp/outspoke-unified-latency-baseline.xml` and
`/tmp/outspoke-unified-latency-tuned.xml`; ONNX profiles are
`/tmp/outspoke-unified-{basic2,all1,all2,all4}_2026-09-30_*.json`. These temporary
files are not release artifacts. Desktop timings are useful evidence of work
cost, not stable CI performance thresholds.

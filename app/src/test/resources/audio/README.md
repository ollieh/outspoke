# Test Audio Fixtures

WAV fixtures for the real-model test classes: `ParakeetEngineRealAudioTest`,
`RealAudioPipelineTest`, `ShortUtterancePipelineTest`, `StreamingParityTest` and
`AcousticCaptureIntegrationTest`.

## Format Requirements

All files must be **16-bit PCM, mono**. Sample rate can be anything (WavReader handles
resampling to 16 kHz). Files must be standard RIFF WAV — WavReader parses the WAV spec
(little-endian numeric fields, raw-ASCII chunk tags).

## File List

All speech fixtures are **synthetic** — Piper neural TTS or generated tone/noise. No
real human voice recordings are committed to this repository.

| File | Content | Duration | Source |
|---|---|---|---|
| `silence-only.wav` | 3 s of digital silence | 3.0 s | `./gradlew generateTestAudio copyGeneratedTestAudio` |
| `noise-only.wav` | 3 s of white noise (±1000) | 3.0 s | `./gradlew generateTestAudio copyGeneratedTestAudio` |
| `whispered.wav` | 0.5 s of near-silence noise (±20) | 0.5 s | `./gradlew generateTestAudio copyGeneratedTestAudio` |
| `silence-then-speech.wav` | 0.6 s silence + "Hello." | ~1.2 s | Piper TTS + ffmpeg |
| `hello.wav` | "Hello." | ~0.5 s | Piper TTS |
| `hello-world.wav` | "Hello, world." | ~1.0 s | Piper TTS |
| `single-yes.wav` | "Yes." | ~0.7 s | Piper TTS |
| `single-no.wav` | "No." | ~0.5 s | Piper TTS |
| `long-two-sentences.wav` | long-sentence + 0.5 s pause + long-sentence (trim-path fixture) | ~9.5 s | Piper TTS + ffmpeg |
| `medium-sentence.wav` | "The quick brown fox jumps over the lazy dog." | ~2.5 s | Piper TTS |
| `long-sentence.wav` | "I was wondering if you could help me with something that I found really interesting this morning." | ~4.5 s | Piper TTS |
| `long-two-sentences.wav` | long-sentence + 0.5 s pause + long-sentence (trim-path fixture) | ~9.5 s | Piper TTS + ffmpeg |
| `long-diff-two-sentences.wav` | two distinct sentences (encoder / decoder) + 0.5 s pause — streaming-parity regression anchor | ~10.4 s | Piper TTS + ffmpeg |
| `med-groceries.wav` | "I need to order some groceries before the store closes at six." | ~3.5 s | Piper TTS |
| `med-report.wav` | "Could you please send me the report by end of day tomorrow?" | ~3.1 s | Piper TTS |
| `med-meeting.wav` | "The meeting was moved to Thursday afternoon at three o'clock." | ~2.9 s | Piper TTS |
| `long-appointment.wav` | "I would like to schedule an appointment with the doctor for next week if possible." | ~4.0 s | Piper TTS |
| `long-approach.wav` | "I think we should reconsider our approach to the project before we continue." | ~3.9 s | Piper TTS |
| `long-callback.wav` | "She said that she would call us back as soon as she got home." | ~2.9 s | Piper TTS |
| `long-furniture.wav` | "He promised to help us move the furniture on Saturday morning." | ~3.2 s | Piper TTS |
| `long-package.wav` | "The package arrived yesterday but the contents were damaged during shipping." | ~3.5 s | Piper TTS |
| `long-plants.wav` | "Please remember to water the plants while I am away on my business trip." | ~3.6 s | Piper TTS |
| `long-pizza.wav` | "The restaurant on the corner serves the best pizza in the whole city." | ~3.3 s | Piper TTS |
| `long-reservation.wav` | "I am writing to confirm that our reservation for two people has been received." | ~3.9 s | Piper TTS |
| `long-slides.wav` | "We need to finish the presentation slides before the client arrives." | ~3.6 s | Piper TTS |
| `long-train.wav` | "Can you tell me what time the train leaves from the central station?" | ~3.5 s | Piper TTS |
| `long-weather.wav` | "The weather forecast predicts rain tomorrow so we should stay indoors." | ~3.7 s | Piper TTS |
| `long-wallet.wav` | "I forgot my wallet at home so I cannot pay for the taxi now." | ~3.3 s | Piper TTS |
| `two-words.wav` | "How are you?" | ~0.7 s | Piper TTS |
| `single-please.wav` | "Please." | ~0.6 s | Piper TTS |
| `single-thanks.wav` | "Thanks." | ~0.5 s | Piper TTS |
| `single-thankyou.wav` | "Thank you." | ~0.7 s | Piper TTS |
| `german-hallo.wav` | German utterance containing "hallo" | ~1.7 s | Piper TTS (de_DE-thorsten-medium) |
| `multi-4sent.wav` | four sentences concatenated with pauses (no-loss/no-duplication battery) | ~10.9 s | Piper TTS + ffmpeg |
| `numbered-sentences.wav` | "This is the first sentence" through "This is the tenth sentence" | ~15 s | Piper TTS (en_US-lessac-medium) |
| `paced-numbered-sentences.wav` | Same numbered sentences with 1.2 s pauses | 26.65 s | Piper TTS (en_US-lessac-medium) |
| `natural-counting.wav` | Ten varied sentences numbered one through ten, with 200 ms leading silence | 18.37 s | Piper TTS (en_US-lessac-medium), line-by-line synthesis, linear resampling to 16 kHz |
| `natural-counting-paused.wav` | Same ten sentences, with 1.8 s silence after each | 36.37 s | Same source as `natural-counting.wav` |
| `blank-chunk-recovery.wav` | Last four natural-counting sentences, preserving the paused recording's chunk alignment | 16.37 s | Same source, with 17,235 leading silence samples |

The natural-counting fixtures say: "This is sentence number one. Now we're on
sentence number two. This is sentence number three. We've got the sentence number
four. This is the fifth sentence. This is the sixth sentence. This is actually the
seventh sentence. This is the eighth sentence. And this sentence is number nine.
And now we've finally made it to sentence number ten."
The unpaused recording reproduced stripped digit tokens; the paused recordings
reproduced a whole missing ninth sentence with the carried streaming predictor.

## Multi-Sentence Pipeline Fixtures

Concatenations of the single-sentence recordings above with 0.5 s pauses, used by the
full-pipeline no-loss / no-duplication battery in `RealAudioPipelineTest` (the
`pipe-*` tests). Each drives a multi-sentence dictation through the complete production
stack and asserts every sentence's words survive and no phrase is committed twice.

| File | Sentences (source recordings) | Duration |
|---|---|---|
| `pipe-a.wav` | report + approach + package (3) | ~11.5 s |
| `pipe-c.wav` | appointment + reservation + slides (3) | ~12.5 s |
| `pipe-e.wav` | approach + plants + package + weather + wallet (5) | ~20.0 s |
| `pipe-f.wav` | medium-sentence (pangram) + long-sentence (2) | ~7.5 s |
| `pipe-g.wav` | train + pizza + appointment + reservation (4) | ~16.2 s |
| `pipe-j.wav` | weather + wallet + pizza (3) | ~11.3 s |
| `pipe-k.wav` | weather + furniture + pizza + train (4) | ~15.2 s |
| `pipe-l.wav` | report + appointment + weather + furniture (4) | ~15.5 s |
| `pipe-n.wav` | meeting + furniture + package + slides (4) | ~14.8 s |
| `pipe-o.wav` | weather + furniture + appointment + pizza (4) | ~15.7 s |
Speech fixtures are neural-TTS (Piper, `en_US-lessac-medium` / `de_DE-thorsten-medium`)
recordings. They are clean, studio-grade speech — a harsher test of the model than
noisy human dictation, but sufficient to anchor the pipeline against regressions.
Replace with real human recordings for higher-fidelity coverage; any standard WAV works.

## Running the Real-Model Tests

The model is **not** checked in. The real-audio tests skip gracefully when the model
directory is absent. To run them:

```bash
# 1. Download the model files (same files the app downloads, SHA-256 verified):
mkdir -p ~/.cache/outspoke-test-model/parakeet-tdt-0.6b-v3 && cd ~/.cache/outspoke-test-model/parakeet-tdt-0.6b-v3
BASE=https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main
for f in vocab.txt config.json nemo128.onnx decoder_joint-model.int8.onnx encoder-model.int8.onnx; do
  curl -sL -O "$BASE/$f"
done

# 2. Run the tests (model dir resolution: -Dtest.model.dir > $OUTSPOKE_TEST_MODEL_DIR > cache dir)
./gradlew testDebugUnitTest
```

Individual recordings take seconds; the full real-model suites take several
minutes on a desktop CPU (INT8, 2 intra-op threads).

The human counting regression uses an external, private recording rather than a
checked-in voice sample. After converting the supplied recording to mono PCM WAV,
run it with:

```bash
./gradlew testDebugUnitTest -Ptest.model.dir=/path/to/parakeet-v3 \
  -Ptest.counting.wav=/private/path/counting.wav \
  --tests '*RealAudioPipelineTest.supplied*'
```

These tests expect the specific ten-sentence recording supplied during debugging.
The first checks every numbered sentence in the one-shot transcript and in streaming
output and editor text at five offsets across a two-second chunk. A second test
checks silence filtering on and off, recovered sentence punctuation, and an unrelated
custom vocabulary. Both skip when the recording is not supplied. Keep the voice
recording outside the repository.

## Generating Synthetic Fixtures

```bash
./gradlew generateTestAudio copyGeneratedTestAudio
```

This regenerates only `silence-only.wav`, `noise-only.wav` and `whispered.wav` — the
speech fixtures above are pre-recorded and are NOT overwritten.

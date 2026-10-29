# Server transcription race

This fork can send each finished recording to the user's own OpenAI-compatible
server, reached over Tailscale, and insert whichever transcript wins the race below.
It stays off while the server URL in Settings → Speech Processing is empty.
The server runs Parakeet Unified EN 0.6B, so recordings in a forced non-English
language stay on the phone.

## Flow

1. When a recording starts, `KeyboardViewModel` asks `ServerTranscriber.startSession`
   for a `ServerRaceSession`. Null means off.
2. `AudioCaptureManager.startCapture(pcmTap = …)` passes each raw `AudioRecord` read,
   before VAD, to `RecordingPcmBuffer`. Pauses therefore stay in the upload.
3. When the capture flow completes normally (`uploadWhenComplete`), the session
   encodes and uploads the whole recording while the phone finishes decoding.
   A cancelled capture uploads nothing.
4. After the phone's flow completes, `ServerRaceSession.choose(localText)` picks the
   text to insert; `localText` is null when the phone failed or heard nothing.

## Race rule (`ServerRacePolicy`)

`stoppedAt` is when the capture completed, which is also when the upload starts.
The server wins if its non-blank text arrives by `max(stoppedAt + grace, localReadyAt)`.
Grace defaults to 2000 ms.

- Server error, timeout or empty text: the phone's text is used as soon as that is known.
- Phone failed or empty: the keyboard waits for the server; the upload is bounded at 35 s.
- Both fail: existing failure behaviour, with the preview kept.

The diagnostics badge shows "via server" or "on phone" for raced recordings.

## Text processing (`ServerTextFormatter`)

Server text receives only the user's preference-driven steps, in the phone's order:
number formatting (when post-processing and digits are on), then custom words.
The phone's other cleanup targets streaming artefacts the offline server decode does
not produce, and it would damage good punctuation. For example, the spurious-period
filter deletes a genuine short sentence's period. `TextInjector`'s display clean is
a pass-through for the streaming engines; a legacy engine would re-clean server text.

## Upload

- Opus in Ogg at 32 kbit/s via `MediaCodec` + `MediaMuxer`; WAV if that fails.
  Temp files live in `cacheDir/server-upload/` and are deleted after each upload.
- `POST {base}/v1/audio/transcriptions` with `file`, `model`, `response_format=json`,
  `language=en`. An empty model setting uses the first id from `GET /v1/models`,
  cached per server for the process.
- OkHttp: 1.5 s connect timeout, 30 s per transcription call, 5 s for the model list.
- `ServerEndpoint.parse` adds `https://` to a bare host and `/v1` when missing.

## Storage and network policy

- URL, model and grace are in the shared DataStore (`ServerTranscriptionPreferences`).
- The API key is AES-GCM encrypted with a non-exportable Android Keystore key in
  `outspoke_server_secret.xml`, which is excluded from backup and device transfer.
- `network_security_config.xml` refuses cleartext app-wide. The server needs an https
  address, e.g. a tailnet certificate through Caddy or `tailscale serve`.

## Recording memory

`RecordingPcmBuffer` keeps up to 15 minutes (about 29 MB). A longer recording releases
its audio and is transcribed on the phone only, because uploading the beginning would
let a truncated server transcript win.

## Debug traces

`OutspokeTrace` lines start with `server`: upload size and duration, encode time,
codec, server latency and status, and the winner with its reason. They contain
lengths and timings only, never audio, transcript text or server error text.

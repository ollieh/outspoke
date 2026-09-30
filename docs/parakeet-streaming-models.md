# Parakeet Unified streaming model research

Verified 2026-09-25 against upstream source and downloadable model metadata. This note records the model and inference contract; Android performance still requires device measurement.

## Model identification

Handy's newer English streaming model is **Parakeet Unified EN 0.6B**, based on `nvidia/parakeet-unified-en-0.6b`. Handy lists it separately from Parakeet TDT v2 and Nemotron Streaming 3.5. Its 0.9 release notes recommend Unified for English and Nemotron 3.5 for multilingual speech. The current catalog advertises English, streaming and token timestamps. [Handy release notes](https://github.com/cjpais/Handy/blob/main/src/content/release-notes/0.9.0.md), [pinned catalog](https://github.com/cjpais/Handy/blob/8f9cf53cd1410cda26beea39ff802ac306e39585/src-tauri/src/catalog/catalog.json).

Unified has a 24-layer FastConformer encoder and an RNN-T decoder, about 600 million parameters, and punctuation/capitalization support. NVIDIA describes buffered streaming that recomputes left context, rather than cached encoder streaming. The model was trained for both offline and streaming inference. [NVIDIA model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b).

Handy uses `transcribe-cpp` 0.2.3 for this GGUF model; `transcribe-rs` 0.3.8 remains for legacy ONNX models. Its catalog pins GGUF revision `7e948f21b7bdbac698d3318db9d350f1096f3b6c`. Q8_0 is 731,357,568 bytes with SHA-256 `4b50b6dd862bf6e346929aaf4f5eaacec003bfa3f56462d6c874b41ef2f38795`. [Handy dependencies](https://github.com/cjpais/Handy/blob/8f9cf53cd1410cda26beea39ff802ac306e39585/src-tauri/Cargo.toml), [GGUF download](https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/7e948f21b7bdbac698d3318db9d350f1096f3b6c/parakeet-unified-en-0.6b-Q8_0.gguf).

The current NVIDIA card specifies NVIDIA Open Model License, while Handy/transcribe.cpp metadata says CC-BY-4.0 for an older upstream revision. Preserve the originating license information; do not infer the current model license solely from Handy's catalog. [NVIDIA card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), [transcribe.cpp model notes](https://github.com/handy-computer/transcribe.cpp/blob/4807edaf210d0d7e8a6f7fb2a44b65966a2797f0/docs/models/parakeet-unified-en-0.6b.md).

## ONNX artifacts suitable for streaming

The sherpa-onnx streaming export is the appropriate ONNX starting point. Its exporter calls `set_default_att_context_size([left, chunk, right])` before exporting the encoder. Generic offline exports cannot be assumed to preserve this streaming attention configuration. Sherpa provides 1120 ms, 560 ms and 240 ms exports; its streaming implementation merged on May 9, 2026. [Exporter](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/scripts/nemo/parakeet-unified-en-0.6b/export_onnx_streaming.py), [merged PR](https://github.com/k2-fsa/sherpa-onnx/pull/3575).

Use the 1120 ms INT8 export at Hugging Face repository `csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-1120ms`, pinned revision `615bf8c5ad39692dc96cf1105afde73ec3eb2a59`. The base download URL is `https://huggingface.co/csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-1120ms/resolve/615bf8c5ad39692dc96cf1105afde73ec3eb2a59/`. [Repository files](https://huggingface.co/csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-1120ms/tree/615bf8c5ad39692dc96cf1105afde73ec3eb2a59).

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder.int8.onnx` | 654046391 | `1c03f1192de41771384af22972ca10203613ba56197a024f275b86727cd35911` |
| `decoder.int8.onnx` | 7257777 | `34fea72425d2506600772ba191a6d3f99c0710abdb68d9a3dc89fa8cb2aa473a` |
| `joiner.int8.onnx` | 1735860 | `869f43f7d24595c55581ad3bf249a935fb8a71389fbdaa7504b9f46f93140f8a` |
| `tokens.txt` | 8952 | `dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff` |

ONNX hashes and sizes come from Hugging Face's LFS metadata; the token hash was calculated from the pinned downloaded file. Total required download is 663,048,980 bytes. [Repository API](https://huggingface.co/api/models/csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-1120ms?blobs=true).

The equivalent GitHub release archive is 501,356,335 bytes, SHA-256 `4788229a6dd03be33f8243ccee48e33a8d15df7b448cb99150b0ccddd1b02d74`. Direct individual Hugging Face files fit Outspoke's model downloader without archive extraction. [Archive](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-1120ms.tar.bz2), [release metadata](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/asr-models).

## Streaming inference contract

The 1120 ms preset uses 70 left, 7 center and 7 right encoder frames. With 8x subsampling and 10 ms feature frames, this means 560 left, 56 center and 56 right feature frames. Each encoder call takes a fixed 672-frame feature window. A chunk advances 8960 audio samples at 16 kHz, with another 8960 samples of right lookahead. Startup and final missing context use zero-valued **feature frames**, before normalization. Decode only encoder frames 70 through 76; preserve predictor state across chunks. There are no encoder cache tensors. Final chunks decode `ceil(valid_center_feature_frames / 8)` frames. [Runtime implementation](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/sherpa-onnx/csrc/online-recognizer-transducer-nemo-parakeet-unified-impl.h).

The encoder takes float features in `[batch,128,time]` and int64 lengths. It returns `[batch,1024,time_out]` plus encoded lengths. The predictor takes int32 tokens `[batch,1]`, int32 lengths and two float LSTM tensors `[2,batch,640]`. The joiner takes an encoder frame and predictor output. There are 1024 text tokens plus blank at 1024; metadata `vocab_size=1024` counts text tokens, while joiner logits have 1025 entries. Initialize LSTM tensors to zero and the previous token to blank. Retain predictor state on blank; advance it on emitted tokens. The upstream decoder limits emissions to 10 tokens per frame. [ONNX wrapper](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/sherpa-onnx/csrc/online-transducer-nemo-parakeet-unified-model.cc), [decoder](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/sherpa-onnx/csrc/online-transducer-greedy-search-nemo-parakeet-unified-decoder.cc).

The separate decoder and joiner differ from Outspoke's existing combined Parakeet TDT decoder graph. This is a model adapter change, not a new native runtime requirement. Existing ONNX Runtime Android can load this graph format in principle; actual operator support, memory consumption and real-time speed must be verified on the supported Android runtime and devices.

## Exact audio frontend

Sherpa's frontend uses normalized float PCM, 16 kHz mono, 400-sample windows, 160-sample shifts, FFT size 512, 128 Slaney mel bands spanning 0–8000 Hz, Slaney area normalization, no dither and no DC removal. It computes power spectra. Preemphasis is 0.97 and applies independently within each extracted frame, including `frame[0] *= 0.03`. Hann is periodic, `0.5 - 0.5*cos(2*pi*i/400)`. With `snip_edges=false`, frame `f` begins at `f*160-120`; boundaries reflect `-1` to `0` and `N` to `N-1`. These details differ from a generic centered PyTorch STFT. [Sherpa configuration](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/sherpa-onnx/csrc/features.h), [frame implementation](https://github.com/csukuangfj/kaldi-native-fbank/blob/b09e686fe2084732ddd30d1ef80acfc0f13eaf01/kaldi-native-fbank/csrc/feature-window.cc), [mel implementation](https://github.com/csukuangfj/kaldi-native-fbank/blob/b09e686fe2084732ddd30d1ef80acfc0f13eaf01/kaldi-native-fbank/csrc/mel-computations.cc).

Log mel is `ln(max(mel_energy, float_epsilon))`, with epsilon approximately `1.1920929e-7`. After assembling and padding the entire feature window, normalize each band over all 672 frames. Use population variance of centered values and denominator `sqrt(variance)+1e-5`. Do not normalize each arriving audio chunk independently. [Log calculation](https://github.com/csukuangfj/kaldi-native-fbank/blob/b09e686fe2084732ddd30d1ef80acfc0f13eaf01/kaldi-native-fbank/csrc/feature-fbank.cc), [normalization](https://github.com/k2-fsa/sherpa-onnx/blob/040afe360a38e25daaa325ce8889abf93ea02609/sherpa-onnx/csrc/math.cc).

For independent frontend verification, a reference fixture was generated using the actual `kaldi-native-fbank` 1.22.3 Python wheel. Input was 1000 signed PCM samples `((i*7919)%60001)-30000`, converted to float by dividing by 32768. With the configuration above the library exposes five frames before `input_finished()` and six afterward. All 128 bands of every frame were handed to implementation work as a golden reference.

To regenerate `app/src/test/resources/audio/unified-fbank-golden.txt`, run this Python script in an environment with `kaldi-native-fbank==1.22.3` and NumPy installed:

```python
import kaldi_native_fbank as k

opts = k.FbankOptions()
opts.frame_opts.dither = 0
opts.frame_opts.remove_dc_offset = False
opts.frame_opts.window_type = "hann"
opts.frame_opts.snip_edges = False
opts.mel_opts.low_freq = 0
opts.mel_opts.high_freq = 8000
opts.mel_opts.num_bins = 128
opts.mel_opts.is_librosa = True
fbank = k.OnlineFbank(opts)
fbank.accept_waveform(16000, [
    ((i * 7919) % 60001 - 30000) / 32768 for i in range(1000)
])
before = fbank.num_frames_ready
fbank.input_finished()
path = "app/src/test/resources/audio/unified-fbank-golden.txt"
with open(path, "w") as out:
    out.write("kaldi-native-fbank 1.22.3; PCM=((i*7919)%60001)-30000 divided by32768; N=1000\n")
    out.write(str(opts) + "\n")
    out.write(f"before_finish={before}; after_finish={fbank.num_frames_ready}\n")
    for i in range(fbank.num_frames_ready):
        values = " ".join(format(float(v), ".9g") for v in fbank.get_frame(i))
        out.write(f"{i} {values}\n")
```

## Alternative exports and runtime

`eschmidbauer/parakeet-unified-en-0.6b-onnx` supplies self-contained INT8 encoder and combined decoder/joint graphs, but its export script does not set streaming attention context. `nub235/parakeet-unified-en-0.6b-onnx-int8` adds a text vocabulary and is used by parakeet-rs. Neither should replace the explicitly streaming sherpa encoder without validating attention behavior. [eschmidbauer exporter](https://huggingface.co/eschmidbauer/parakeet-unified-en-0.6b-onnx/blob/7a16ff98b72e5b0beba6f182a5357c704d977b15/onnx_export.py), [nub235 export](https://huggingface.co/nub235/parakeet-unified-en-0.6b-onnx-int8/tree/e03056f2e873182f7e36e19d2d3bf7a3fcb6f79f).

The GGUF alternative has an upstream C API and documented buffered-streaming parity with NeMo. A JNI/NDK bridge would be needed for this Android application. The runtime has official Python, Rust, Swift and JavaScript bindings; the inspected source does not establish a maintained Android Java binding. ONNX therefore avoids introducing another native inference stack. [transcribe.cpp C API](https://github.com/handy-computer/transcribe.cpp/blob/4807edaf210d0d7e8a6f7fb2a44b65966a2797f0/include/transcribe.h), [streaming model documentation](https://github.com/handy-computer/transcribe.cpp/blob/4807edaf210d0d7e8a6f7fb2a44b65966a2797f0/docs/models/parakeet-unified-en-0.6b.md).

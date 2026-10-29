package dev.brgr.outspoke.remote

import android.media.MediaCodec
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.util.concurrent.TimeUnit

private const val TAG = "RecordingEncoder"
private const val OPUS_BIT_RATE = 32_000
private const val CODEC_TIMEOUT_US = 10_000L
private val STALE_UPLOAD_AGE_MS = TimeUnit.HOURS.toMillis(1)

/** A recording ready to upload. The caller deletes [file]. */
class EncodedAudio(val file: File, val mimeType: String, val codec: String)

/**
 * Encodes a recording compactly for upload: Opus in Ogg (about 4 kB per second of
 * speech), or 16-bit WAV (32 kB per second) when the platform encoder fails.
 */
internal class RecordingEncoder(
    private val directory: File,
    private val opus: suspend (PcmRecording, File) -> Unit = OggOpusEncoder::encode,
) {
    suspend fun encode(recording: PcmRecording): EncodedAudio {
        directory.mkdirs()
        // Leftovers from a process that died mid-upload.
        val now = System.currentTimeMillis()
        directory.listFiles()?.filter { now - it.lastModified() > STALE_UPLOAD_AGE_MS }?.forEach { it.delete() }

        val ogg = File.createTempFile("recording-", ".ogg", directory)
        try {
            opus(recording, ogg)
            if (ogg.length() > 0) return EncodedAudio(ogg, "audio/ogg", "opus")
            Log.w(TAG, "Opus encoder produced no output; uploading WAV")
        } catch (e: CancellationException) {
            ogg.delete()
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Opus encoding failed; uploading WAV", e)
        }
        ogg.delete()
        val wav = File.createTempFile("recording-", ".wav", directory)
        try {
            WavWriter.write(recording, wav)
        } catch (e: Exception) {
            wav.delete()
            throw e
        }
        return EncodedAudio(wav, "audio/wav", "wav")
    }
}

internal object WavWriter {
    fun write(recording: PcmRecording, out: File) {
        val dataBytes = recording.sampleCount * 2
        out.outputStream().buffered().use { stream ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray(Charsets.US_ASCII))
            header.putInt((36 + dataBytes).toInt())
            header.put("WAVE".toByteArray(Charsets.US_ASCII))
            header.put("fmt ".toByteArray(Charsets.US_ASCII))
            header.putInt(16)
            header.putShort(1) // PCM
            header.putShort(1) // mono
            header.putInt(recording.sampleRate)
            header.putInt(recording.sampleRate * 2)
            header.putShort(2) // block align
            header.putShort(16) // bits per sample
            header.put("data".toByteArray(Charsets.US_ASCII))
            header.putInt(dataBytes.toInt())
            stream.write(header.array())
            for (block in recording.blocks) {
                val bytes = ByteBuffer.allocate(block.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                bytes.asShortBuffer().put(block)
                stream.write(bytes.array())
            }
        }
    }
}

/** Platform Opus encoder (Android 10+) into an Ogg container. */
internal object OggOpusEncoder {
    suspend fun encode(recording: PcmRecording, out: File) {
        val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_OPUS, recording.sampleRate, 1)
        format.setInteger(MediaFormat.KEY_BIT_RATE, OPUS_BIT_RATE)
        val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_OPUS)
        var muxer: MediaMuxer? = null
        var muxerStarted = false
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            muxer = MediaMuxer(out.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_OGG)
            val input = PcmCursor(recording.blocks)
            val info = MediaCodec.BufferInfo()
            var track = -1
            var inputDone = false
            var queuedSamples = 0L
            while (true) {
                currentCoroutineContext().ensureActive()
                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(CODEC_TIMEOUT_US)
                    if (index >= 0) {
                        val buffer = checkNotNull(codec.getInputBuffer(index))
                        buffer.clear()
                        val ptsUs = queuedSamples * 1_000_000L / recording.sampleRate
                        val samples = input.fill(buffer.order(ByteOrder.nativeOrder()).asShortBuffer())
                        if (samples == 0) {
                            codec.queueInputBuffer(index, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(index, 0, samples * 2, ptsUs, 0)
                            queuedSamples += samples
                        }
                    }
                }
                val index = codec.dequeueOutputBuffer(info, CODEC_TIMEOUT_US)
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    check(!muxerStarted) { "Opus output format changed twice" }
                    // The output format carries the Opus header the Ogg muxer needs.
                    track = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                    muxerStarted = true
                } else if (index >= 0) {
                    val buffer = checkNotNull(codec.getOutputBuffer(index))
                    val isConfig = info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                    if (!isConfig && info.size > 0) {
                        check(muxerStarted) { "Opus data arrived before its format" }
                        buffer.position(info.offset)
                        buffer.limit(info.offset + info.size)
                        muxer.writeSampleData(track, buffer, info)
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
            check(muxerStarted) { "Opus encoder produced no format" }
            muxerStarted = false
            muxer.stop()
        } finally {
            runCatching { codec.stop() }
            codec.release()
            if (muxerStarted) runCatching { muxer?.stop() }
            muxer?.release()
        }
    }

    private class PcmCursor(private val blocks: List<ShortArray>) {
        private var block = 0
        private var offset = 0

        fun fill(target: ShortBuffer): Int {
            var written = 0
            while (target.hasRemaining() && block < blocks.size) {
                val current = blocks[block]
                val count = minOf(target.remaining(), current.size - offset)
                target.put(current, offset, count)
                offset += count
                written += count
                if (offset == current.size) {
                    block++
                    offset = 0
                }
            }
            return written
        }
    }
}

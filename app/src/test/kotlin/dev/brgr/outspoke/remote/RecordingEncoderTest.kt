package dev.brgr.outspoke.remote

import kotlinx.coroutines.runBlocking
import org.assertj.core.api.Assertions.assertThat
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder

class RecordingEncoderTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val recording = PcmRecording(listOf(shortArrayOf(1, -2, 3), shortArrayOf(Short.MAX_VALUE, Short.MIN_VALUE)), 16_000)

    @Test
    fun `uses Opus when the platform encoder succeeds`() = runBlocking<Unit> {
        val encoder = RecordingEncoder(folder.root, opus = { _, out -> out.writeText("OggS") })

        val audio = encoder.encode(recording)

        assertThat(audio.mimeType).isEqualTo("audio/ogg")
        assertThat(audio.codec).isEqualTo("opus")
        assertThat(audio.file.name).endsWith(".ogg")
        assertThat(audio.file.readText()).isEqualTo("OggS")
    }

    @Test
    fun `falls back to WAV when Opus fails or writes nothing`() = runBlocking<Unit> {
        for (opus in listOf<suspend (PcmRecording, java.io.File) -> Unit>({ _, _ -> error("no encoder") }, { _, _ -> })) {
            val audio = RecordingEncoder(folder.root, opus).encode(recording)

            assertThat(audio.mimeType).isEqualTo("audio/wav")
            assertThat(audio.file.name).endsWith(".wav")
            assertThat(folder.root.listFiles()!!.filter { it.name.endsWith(".ogg") }).isEmpty()
            audio.file.delete()
        }
    }

    @Test
    fun `writes a 16 kHz mono 16-bit WAV with every sample`() {
        val out = folder.newFile("recording.wav")

        WavWriter.write(recording, out)

        val bytes = ByteBuffer.wrap(out.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        assertThat(out.length()).isEqualTo(44L + 5 * 2)
        assertThat(String(out.readBytes(), 0, 4, Charsets.US_ASCII)).isEqualTo("RIFF")
        assertThat(bytes.getInt(4)).isEqualTo(36 + 10)
        assertThat(String(out.readBytes(), 8, 8, Charsets.US_ASCII)).isEqualTo("WAVEfmt ")
        assertThat(bytes.getShort(20)).isEqualTo(1.toShort())
        assertThat(bytes.getShort(22)).isEqualTo(1.toShort())
        assertThat(bytes.getInt(24)).isEqualTo(16_000)
        assertThat(bytes.getInt(28)).isEqualTo(32_000)
        assertThat(bytes.getShort(34)).isEqualTo(16.toShort())
        assertThat(String(out.readBytes(), 36, 4, Charsets.US_ASCII)).isEqualTo("data")
        assertThat(bytes.getInt(40)).isEqualTo(10)
        val samples = (0 until 5).map { bytes.getShort(44 + it * 2) }
        assertThat(samples).containsExactly(1, -2, 3, Short.MAX_VALUE, Short.MIN_VALUE)
    }
}

package dev.brgr.outspoke.ui.keyboard

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class RecordingTranscriptTest {
    @Test
    fun `the whole active utterance remains revisable`() {
        val recording = RecordingTranscript()
        recording.partial("You can use this interface while speaking")
        recording.partial("UI can use this interface while speaking")
        assertThat(recording.text).isEqualTo("UI can use this interface while speaking")
        assertThat(recording.hasFinalText).isFalse()
        recording.final("UI can use this interface while speaking.")
        assertThat(recording.text).isEqualTo("UI can use this interface while speaking.")
    }

    @Test
    fun `silence boundaries retain completed utterances without duplicating partials`() {
        val recording = RecordingTranscript()
        recording.partial("First sentence")
        recording.final("First sentence.")
        recording.partial("Next")
        recording.partial("Next sentence")
        assertThat(recording.text).isEqualTo("First sentence. Next sentence")
        recording.final("Next sentence.")
        assertThat(recording.text).isEqualTo("First sentence. Next sentence.")
        assertThat(recording.hasFinalText).isTrue()
    }

    @Test
    fun `stopping after a boundary retains the recording even with an empty last utterance`() {
        val recording = RecordingTranscript()
        recording.final("Already done.")
        recording.windowTrimmed()
        recording.final("")
        assertThat(recording.text).isEqualTo("Already done.")
    }

    @Test
    fun `a repeated utterance after silence is preserved`() {
        val recording = RecordingTranscript()
        repeat(2) { recording.final("One two three.") }
        assertThat(recording.text).isEqualTo("One two three. One two three.")
    }

    @Test
    fun `legacy trims preserve the dropped prefix and allow overlap refinements`() {
        val recording = RecordingTranscript()
        recording.partial("Here are some words about the interface")
        recording.windowTrimmed()
        recording.partial("about the interface and more words")
        recording.windowTrimmed()
        recording.final("and more words.")
        assertThat(recording.text).isEqualTo("Here are some words about the interface and more words.")
    }

    @Test
    fun `a sentence boundary followed by a trim does not overlap the next sentence`() {
        val recording = RecordingTranscript()
        recording.final("This is a sentence.")
        recording.windowTrimmed()
        recording.final("This is another sentence.")
        assertThat(recording.text).isEqualTo("This is a sentence. This is another sentence.")
    }
}

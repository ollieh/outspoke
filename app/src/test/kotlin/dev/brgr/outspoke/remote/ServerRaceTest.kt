package dev.brgr.outspoke.remote

import dev.brgr.outspoke.inference.TranscriptSource
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test
import java.io.IOException

class ServerRacePolicyTest {
    private fun decide(local: String?, server: ServerOutcome?, localReadyAt: Long) =
        ServerRacePolicy.decide(local, server, stoppedAtMs = 1_000, graceMs = 2_000, localReadyAtMs = localReadyAt)

    @Test
    fun `server text before the phone is ready wins`() {
        val decision = decide("phone", ServerOutcome.Text("Server.", atMs = 1_400), localReadyAt = 1_500)
        assertThat(decision).isEqualTo(RaceDecision("Server.", TranscriptSource.SERVER, "server-first"))
    }

    @Test
    fun `server text within the grace period after the phone wins`() {
        val decision = decide("phone", ServerOutcome.Text("Server.", atMs = 3_000), localReadyAt = 1_500)
        assertThat(decision).isEqualTo(RaceDecision("Server.", TranscriptSource.SERVER, "server-in-grace"))
    }

    @Test
    fun `server text after the grace period loses`() {
        val decision = decide("phone", ServerOutcome.Text("Server.", atMs = 3_001), localReadyAt = 1_500)
        assertThat(decision).isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-late"))
    }

    @Test
    fun `a slow phone extends the deadline to its own ready time`() {
        assertThat(ServerRacePolicy.deadlineMs(stoppedAtMs = 1_000, graceMs = 2_000, localReadyAtMs = 5_000))
            .isEqualTo(5_000)
        val decision = decide("phone", ServerOutcome.Text("Server.", atMs = 4_500), localReadyAt = 5_000)
        assertThat(decision.source).isEqualTo(TranscriptSource.SERVER)
    }

    @Test
    fun `server errors, empty text and pending uploads keep the phone's text`() {
        assertThat(decide("phone", ServerOutcome.Failed("timeout", 1_200), 1_500))
            .isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-error:timeout"))
        assertThat(decide("phone", ServerOutcome.Text("  ", 1_200), 1_500))
            .isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-empty"))
        assertThat(decide("phone", null, 1_500))
            .isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-late"))
    }

    @Test
    fun `server text rescues a failed phone however late`() {
        val decision = decide(null, ServerOutcome.Text("Server.", atMs = 60_000), localReadyAt = 1_500)
        assertThat(decision).isEqualTo(RaceDecision("Server.", TranscriptSource.SERVER, "phone-failed"))
    }

    @Test
    fun `nothing is chosen when both fail`() {
        assertThat(decide(null, ServerOutcome.Failed("http-500", 1_200), 1_500))
            .isEqualTo(RaceDecision(null, null, "both-failed"))
        assertThat(decide(null, ServerOutcome.Text("", 1_200), 1_500))
            .isEqualTo(RaceDecision(null, null, "both-failed"))
    }
}

@OptIn(ExperimentalCoroutinesApi::class)
class ServerRaceSessionTest {
    private fun TestScope.session(graceMs: Long = 2_000, transcribe: suspend (PcmRecording) -> String) =
        ServerRaceSession(backgroundScope, graceMs, recordingId = 1, transcribe = transcribe, clock = { currentTime })

    private fun ServerRaceSession.record() {
        pcm.onPcm(ShortArray(1_600))
        onAudioComplete()
    }

    @Test
    fun `server first wins without waiting`() = runTest {
        val race = session { delay(500); "Server." }
        race.record()
        advanceTimeBy(1_000)

        val decision = race.choose("phone")

        assertThat(decision.text).isEqualTo("Server.")
        assertThat(decision.reason).isEqualTo("server-first")
        assertThat(currentTime).isEqualTo(1_000)
    }

    @Test
    fun `phone first waits for a server answer inside the grace period`() = runTest {
        val race = session { delay(1_500); "Server." }
        race.record()
        advanceTimeBy(300)

        val decision = race.choose("phone")

        assertThat(decision).isEqualTo(RaceDecision("Server.", TranscriptSource.SERVER, "server-in-grace"))
        assertThat(currentTime).isEqualTo(1_500)
    }

    @Test
    fun `a late server is abandoned when the grace period ends`() = runTest {
        var uploadCancelled = false
        val race = session {
            try {
                delay(5_000); "Server."
            } finally {
                uploadCancelled = true
            }
        }
        race.record()
        advanceTimeBy(300)

        val decision = race.choose("phone")
        runCurrent()

        assertThat(decision).isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-late"))
        assertThat(currentTime).isEqualTo(2_000)
        assertThat(uploadCancelled).isTrue()
    }

    @Test
    fun `a phone ready after the grace period does not wait`() = runTest {
        val race = session { delay(4_000); "Server." }
        race.record()
        advanceTimeBy(3_000)

        val decision = race.choose("phone")

        assertThat(decision.reason).isEqualTo("server-late")
        assertThat(currentTime).isEqualTo(3_000)
    }

    @Test
    fun `a server error ends the wait immediately`() = runTest {
        val race = session { delay(600); throw IOException("refused") }
        race.record()
        advanceTimeBy(100)

        val decision = race.choose("phone")

        assertThat(decision).isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-error:IOException"))
        assertThat(currentTime).isEqualTo(600)
    }

    @Test
    fun `server error kinds are reported without server text`() = runTest {
        val race = session {
            throw ServerTranscriptionException(ServerTranscriptionException.Kind.HTTP, 503, detail = "secret detail")
        }
        race.record()
        runCurrent()

        assertThat(race.choose("phone").reason).isEqualTo("server-error:http-503")
    }

    @Test
    fun `a failed phone waits for the server`() = runTest {
        val race = session { delay(4_000); "Server." }
        race.record()
        advanceTimeBy(100)

        val decision = race.choose(null)

        assertThat(decision).isEqualTo(RaceDecision("Server.", TranscriptSource.SERVER, "phone-failed"))
        assertThat(currentTime).isEqualTo(4_000)
    }

    @Test
    fun `a hung upload is bounded`() = runTest {
        val race = session { CompletableDeferred<String>().await() }
        race.record()

        val decision = race.choose(null)

        assertThat(decision).isEqualTo(RaceDecision(null, null, "both-failed"))
        assertThat(currentTime).isEqualTo(35_000)
    }

    @Test
    fun `a capture that never completed uploads nothing`() = runTest {
        var uploads = 0
        val race = session { uploads++; "Server." }
        race.pcm.onPcm(ShortArray(1_600))

        val decision = race.choose("phone")
        race.onAudioComplete()
        runCurrent()

        assertThat(decision).isEqualTo(RaceDecision("phone", TranscriptSource.PHONE, "server-skipped:audio-incomplete"))
        assertThat(uploads).isZero()
    }

    @Test
    fun `an overlong recording stays on the phone`() = runTest {
        var uploads = 0
        val race = ServerRaceSession(
            backgroundScope, 2_000, 1, transcribe = { uploads++; "Server." }, clock = { currentTime },
            pcm = RecordingPcmBuffer(maxSamples = 1_000),
        )
        race.pcm.onPcm(ShortArray(1_001))
        race.onAudioComplete()
        runCurrent()

        assertThat(race.choose("phone").reason).isEqualTo("server-skipped:over-15-minutes")
        assertThat(race.choose(null).text).isNull()
        assertThat(uploads).isZero()
    }

    @Test
    fun `the upload starts when capture completes, before the phone finishes`() = runTest {
        val started = CompletableDeferred<Long>()
        val race = session { started.complete(currentTime); "Server." }
        advanceTimeBy(700)
        race.record()
        runCurrent()

        assertThat(started.isCompleted).isTrue()
        assertThat(started.await()).isEqualTo(700)
        assertThat(race.choose("phone").source).isEqualTo(TranscriptSource.SERVER)
    }
}

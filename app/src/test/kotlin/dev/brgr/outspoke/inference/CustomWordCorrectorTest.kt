package dev.brgr.outspoke.inference

import android.text.InputType
import android.view.inputmethod.EditorInfo
import dev.brgr.outspoke.audio.AudioChunk
import dev.brgr.outspoke.ime.FakeInputConnection
import dev.brgr.outspoke.ime.TextInjector
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class CustomWordCorrectorTest {
    @Test
    fun `saved spelling replaces split names and retains surrounding punctuation`() {
        val corrector = CustomWordCorrector(listOf("Astro.js", "MacBook Pro", "CJ"))
        assertThat(corrector.correct("Try astro js, on a mac book pro with c j!"))
            .isEqualTo("Try Astro.js, on a MacBook Pro with CJ!")
    }

    @Test
    fun `close spellings and phonetic matches are corrected`() {
        val corrector = CustomWordCorrector(listOf("Kubernetes", "Steven"))
        assertThat(corrector.correct("Kubernates and Stephen"))
            .isEqualTo("Kubernetes and Steven")
    }

    @Test
    fun `short ordinary words and numbers do not get fuzzy replacements`() {
        val corrector = CustomWordCorrector(listOf("CJ", "June", "R2D2", "Outspoke"))
        val original = "See Jay in July. R2D3 counted 1 2 3 and went outside."
        assertThat(corrector.correct(original)).isEqualTo(original)
    }

    @Test
    fun `ambiguous names are left unchanged`() {
        assertThat(CustomWordCorrector(listOf("Mariana", "Mariena")).correct("Mariona"))
            .isEqualTo("Mariona")
    }

    @Test
    fun `matching does not cross punctuation or newlines`() {
        val corrector = CustomWordCorrector(listOf("MacBook Pro"))
        val original = "Mac, book pro. Mac\nbook pro."
        assertThat(corrector.correct(original)).isEqualTo(original)
    }

    @Test
    fun `unicode names keep accents and preferred case`() {
        assertThat(CustomWordCorrector(listOf("Michaëla", "Łukasz")).correct("michaëla and łukasz"))
            .isEqualTo("Michaëla and Łukasz")
    }

    @Test
    fun `empty vocabulary leaves whitespace and punctuation untouched`() {
        val original = "  Hello,\n  world! "
        assertThat(CustomWordCorrector(emptyList()).correct(original)).isEqualTo(original)
    }

    @Test
    fun `stored vocabulary is normalized deduplicated and bounded`() {
        assertThat(CustomWordCorrector.normalizeWords(listOf(" Mac  Book Pro ", "macbookpro", "!!!", "123", "a b c d")))
            .containsExactly("Mac Book Pro")
        assertThat(CustomWordCorrector.normalizeWords((1..150).map { "term$it" })).hasSize(100)
        assertThat(CustomWordCorrector.normalizeWords(listOf("x".repeat(51)))).isEmpty()
    }

    @Test
    fun `saved punctuation is not duplicated when correction repeats`() {
        val corrector = CustomWordCorrector(listOf("C++", "Astro.js", ".NET"))
        val once = corrector.correct("Use C++ and Astro.js.")
        assertThat(once).isEqualTo("Use C++ and Astro.js.")
        assertThat(corrector.correct(once)).isEqualTo(once)
        assertThat(corrector.correct("Use C++.")).isEqualTo("Use C++.")
        assertThat(corrector.correct("Use (.NET). ")).isEqualTo("Use (.NET). ")
    }

    @Test
    fun `custom phrases survive streaming partials freezing and final insertion`() = runBlocking {
        val words = "We are testing this keyboard today with a mac book pro and Kubernates while continuing to dictate several more words.".split(' ')
        val engine = FakeStreamingParakeet(words.mapIndexed { index, word -> index * 8000 to word })
        val field = FakeInputConnection()
        val injector = TextInjector(field, EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT })
        val results = InferenceRepository(engine).transcribe(
            flowOf(*List(350) { AudioChunk(ShortArray(640)) }.toTypedArray()),
            customWords = listOf("MacBook Pro", "Kubernetes"),
        ).toList()
        assertThat(results.filterIsInstance<TranscriptResult.Partial>()).isNotEmpty()
        results.forEach {
            when (it) {
                is TranscriptResult.Partial -> injector.setPartial(it.text)
                is TranscriptResult.Final -> injector.commitFinal(it.text)
                else -> Unit
            }
        }
        assertThat(field.fieldText).isEqualTo(
            "We are testing this keyboard today with a MacBook Pro and Kubernetes while continuing to dictate several more words."
        )
        Unit
    }
}

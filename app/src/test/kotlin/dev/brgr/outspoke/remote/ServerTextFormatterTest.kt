package dev.brgr.outspoke.remote

import org.assertj.core.api.Assertions.assertThat
import org.junit.Test

class ServerTextFormatterTest {
    @Test
    fun `formats numbers and keeps the server's punctuation`() {
        val formatter = ServerTextFormatter(formatNumbersAsDigits = true, customWords = emptyList())

        assertThat(formatter.format(" I need twenty five apples, please. Okay. ")).isEqualTo("I need 25 apples, please. Okay.")
        assertThat(formatter.format("Ten, nine, eight.")).isEqualTo("10, 9, 8.")
    }

    @Test
    fun `leaves number words alone when formatting is off`() {
        val formatter = ServerTextFormatter(formatNumbersAsDigits = false, customWords = emptyList())

        assertThat(formatter.format("I need twenty five apples.")).isEqualTo("I need twenty five apples.")
    }

    @Test
    fun `applies custom words after numbers`() {
        val formatter = ServerTextFormatter(formatNumbersAsDigits = true, customWords = listOf("Outspoke"))

        assertThat(formatter.format("I use out spoke on three phones.")).isEqualTo("I use Outspoke on 3 phones.")
    }

    @Test
    fun `blank server text stays blank`() {
        assertThat(ServerTextFormatter(true, listOf("Outspoke")).format("  ")).isEmpty()
    }
}

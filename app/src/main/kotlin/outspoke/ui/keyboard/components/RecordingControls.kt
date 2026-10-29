package dev.brgr.outspoke.ui.keyboard.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.brgr.outspoke.R
import dev.brgr.outspoke.ui.theme.MyIcons
import dev.brgr.outspoke.ui.theme.OutspokeKeyboardTheme

/** Height of the controls row; replaces the 28 dp waveform row while dictating. */
private val CONTROLS_HEIGHT = 32.dp

/**
 * Shown in place of the waveform row while a dictation is in progress: discard on the
 * left and the waveform. The talk button finishes and inserts; the Enter key turns into
 * Send (finish, insert, then send).
 */
@Composable
fun RecordingControls(
    amplitude: Float,
    onDiscard: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(CONTROLS_HEIGHT),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        IconButton(onClick = onDiscard, modifier = Modifier.size(CONTROLS_HEIGHT)) {
            Icon(
                imageVector = MyIcons.Close,
                contentDescription = stringResource(R.string.cd_discard_dictation),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp),
            )
        }
        BoxWithConstraints(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
            // Default bars are 3 dp wide with 2 dp gaps.
            val bars = ((maxWidth + 2.dp) / 5.dp).toInt().coerceIn(4, 60)
            WaveformBar(amplitude = amplitude, maxBarHeight = 24.dp, barCount = bars)
        }
        // Balances the discard button so the waveform stays centred.
        Spacer(modifier = Modifier.size(CONTROLS_HEIGHT))
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF111111, widthDp = 380)
@Composable
private fun RecordingControlsPreview() {
    OutspokeKeyboardTheme {
        RecordingControls(amplitude = 0.6f, onDiscard = {})
    }
}

@Preview(showBackground = true, backgroundColor = 0xFF111111, widthDp = 320)
@Composable
private fun RecordingControlsNarrowPreview() {
    OutspokeKeyboardTheme {
        RecordingControls(amplitude = 0.3f, onDiscard = {})
    }
}

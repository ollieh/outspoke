package dev.brgr.outspoke.settings.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.brgr.outspoke.R
import dev.brgr.outspoke.remote.ServerTranscriptionException
import dev.brgr.outspoke.settings.preferences.ServerTranscriptionViewModel
import dev.brgr.outspoke.settings.preferences.ServerTranscriptionViewModel.TestState

/** Optional transcription on the user's own server, raced against the phone. */
@Composable
fun ServerTranscriptionSection(viewModel: ServerTranscriptionViewModel = viewModel()) {
    val form by viewModel.form.collectAsState()
    val saved by viewModel.saved.collectAsState()
    val testState by viewModel.testState.collectAsState()
    val keyStored by viewModel.keyStored.collectAsState()
    var showKey by rememberSaveable { mutableStateOf(false) }
    val loaded = saved != null

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(stringResource(R.string.pref_server_title), style = MaterialTheme.typography.titleMedium)
        Text(
            stringResource(R.string.pref_server_summary),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        OutlinedTextField(
            value = form.baseUrl,
            onValueChange = { value -> viewModel.update { it.copy(baseUrl = value) } },
            label = { Text(stringResource(R.string.pref_server_url_label)) },
            placeholder = { Text(stringResource(R.string.pref_server_url_example)) },
            singleLine = true,
            enabled = loaded,
            isError = !form.urlValid,
            supportingText = {
                Text(stringResource(when {
                    !form.urlValid -> R.string.pref_server_url_invalid
                    form.baseUrl.isBlank() -> R.string.pref_server_off
                    form.cleartext -> R.string.pref_server_url_cleartext
                    else -> R.string.pref_server_url_on
                }))
            },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = form.apiKey,
            onValueChange = { value -> viewModel.update { it.copy(apiKey = value) } },
            label = { Text(stringResource(R.string.pref_server_key_label)) },
            singleLine = true,
            enabled = loaded,
            visualTransformation = if (showKey) VisualTransformation.None else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
            trailingIcon = {
                TextButton(onClick = { showKey = !showKey }) {
                    Text(stringResource(if (showKey) R.string.pref_server_key_hide else R.string.pref_server_key_show))
                }
            },
            isError = !keyStored,
            supportingText = if (keyStored) null else {
                { Text(stringResource(R.string.pref_server_key_not_stored)) }
            },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = form.model,
            onValueChange = { value -> viewModel.update { it.copy(model = value) } },
            label = { Text(stringResource(R.string.pref_server_model_label)) },
            singleLine = true,
            enabled = loaded,
            supportingText = { Text(stringResource(R.string.pref_server_model_hint)) },
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = form.graceMs,
            onValueChange = { value -> viewModel.update { it.copy(graceMs = value) } },
            label = { Text(stringResource(R.string.pref_server_grace_label)) },
            singleLine = true,
            enabled = loaded,
            isError = form.grace == null,
            supportingText = {
                Text(stringResource(
                    if (form.grace == null) R.string.pref_server_grace_invalid else R.string.pref_server_grace_hint
                ))
            },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth(),
        )
        TestStatus(testState)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
        ) {
            OutlinedButton(
                onClick = viewModel::testConnection,
                enabled = loaded && form.baseUrl.isNotBlank() && testState != TestState.Running,
            ) {
                Text(stringResource(
                    if (testState == TestState.Running) R.string.pref_server_testing else R.string.pref_server_test
                ))
            }
            Button(onClick = viewModel::save, enabled = loaded && form.valid && form != saved) {
                Text(stringResource(R.string.pref_server_save))
            }
        }
    }
}

@Composable
private fun TestStatus(state: TestState) {
    val message = when (state) {
        TestState.Idle, TestState.Running -> return
        is TestState.Connected -> state.filledModel?.let { stringResource(R.string.pref_server_test_filled, it) }
            ?: stringResource(R.string.pref_server_test_ok, state.models.joinToString(", "))
        is TestState.Failed -> errorMessage(state.error)
    }
    Text(
        message,
        style = MaterialTheme.typography.bodyMedium,
        color = if (state is TestState.Failed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
    )
}

@Composable
private fun errorMessage(error: ServerTranscriptionException): String {
    val base = when (error.kind) {
        ServerTranscriptionException.Kind.INVALID_URL -> stringResource(R.string.pref_server_url_invalid)
        ServerTranscriptionException.Kind.UNREACHABLE -> stringResource(R.string.pref_server_error_unreachable)
        ServerTranscriptionException.Kind.TIMEOUT -> stringResource(R.string.pref_server_error_timeout)
        ServerTranscriptionException.Kind.CLEARTEXT_BLOCKED -> stringResource(R.string.pref_server_error_cleartext)
        ServerTranscriptionException.Kind.UNAUTHORIZED ->
            stringResource(R.string.pref_server_error_unauthorized, error.httpStatus)
        ServerTranscriptionException.Kind.HTTP -> stringResource(R.string.pref_server_error_http, error.httpStatus)
        ServerTranscriptionException.Kind.BAD_RESPONSE -> stringResource(R.string.pref_server_error_bad_response)
        ServerTranscriptionException.Kind.NO_MODELS -> stringResource(R.string.pref_server_error_no_models)
    }
    return error.detail?.let { "$base\n$it" } ?: base
}

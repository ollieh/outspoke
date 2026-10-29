package dev.brgr.outspoke.settings.preferences

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

private const val TAG = "ServerApiKeyStore"
private const val FILE = "outspoke_server_secret"
private const val ENTRY = "api_key"
private const val KEY_ALIAS = "outspoke_server_api_key"
private const val GCM_IV_BYTES = 12
private const val GCM_TAG_BITS = 128

/**
 * The server API key, encrypted with a non-exportable Android Keystore key. Its file is
 * excluded from backups (see `backup_rules.xml`), where it could not be decrypted anyway.
 * Blocking: call off the main thread.
 */
class ServerApiKeyStore(context: Context) {
    private val prefs by lazy { context.getSharedPreferences(FILE, Context.MODE_PRIVATE) }

    /** The stored key, or "" when none is stored or it can no longer be decrypted. */
    fun get(): String {
        val stored = prefs.getString(ENTRY, null) ?: return ""
        return try {
            val bytes = Base64.decode(stored, Base64.NO_WRAP)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(GCM_TAG_BITS, bytes, 0, GCM_IV_BYTES))
            String(cipher.doFinal(bytes, GCM_IV_BYTES, bytes.size - GCM_IV_BYTES), Charsets.UTF_8)
        } catch (e: Exception) {
            Log.w(TAG, "Stored server API key cannot be decrypted; ignoring it", e)
            ""
        }
    }

    /** Stores [apiKey], or removes it when blank. Returns false if encryption failed. */
    fun set(apiKey: String): Boolean {
        val trimmed = apiKey.trim()
        if (trimmed.isEmpty()) {
            prefs.edit().remove(ENTRY).apply()
            return true
        }
        return try {
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, secretKey())
            val sealed = cipher.iv + cipher.doFinal(trimmed.toByteArray(Charsets.UTF_8))
            prefs.edit().putString(ENTRY, Base64.encodeToString(sealed, Base64.NO_WRAP)).apply()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Cannot encrypt the server API key", e)
            false
        }
    }

    private fun secretKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (keyStore.getKey(KEY_ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }
}

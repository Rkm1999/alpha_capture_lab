package com.ryu.sonyremote

import android.app.Activity
import android.content.Intent
import android.os.Bundle

class LutShareActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        forwardToAlphaCaptureLab(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        forwardToAlphaCaptureLab(intent)
    }

    private fun forwardToAlphaCaptureLab(sharedIntent: Intent) {
        startActivity(Intent(sharedIntent).apply {
            setClass(this@LutShareActivity, MainActivity::class.java)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        })
        finish()
    }
}

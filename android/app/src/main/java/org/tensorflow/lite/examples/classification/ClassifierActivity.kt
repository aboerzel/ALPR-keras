/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.classification

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ImageReader
import android.os.SystemClock
import android.util.Size
import org.tensorflow.lite.examples.classification.env.Logger
import org.tensorflow.lite.examples.classification.tflite.LicenseRecognizer
import java.io.IOException

class ClassifierActivity : CameraActivity(), ImageReader.OnImageAvailableListener {
    private var rgbFrameBitmap: Bitmap? = null
    private var licenseRecognizer: LicenseRecognizer? = null
    private var lastProcessingTimeMs: Long = 0
    override val layoutId: Int
        get() = R.layout.camera_connection_fragment
    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        recreateClassifier()
        if (licenseRecognizer == null) {
            LOGGER.e("No licenseRecognizer on preview!")
            return
        }
        previewWidth = size.width
        previewHeight = size.height
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
    }

    override fun processImage() {
        rgbFrameBitmap!!.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)
        
        runInBackground(
                Runnable {
                    if (licenseRecognizer != null) {
                        val startTime = SystemClock.uptimeMillis()
                        val result = licenseRecognizer!!.recognize(correctImageOrientation(rgbFrameBitmap!!))
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                        LOGGER.v("Detect: %s", result)
                        LOGGER.v("Processing time: %d ms", lastProcessingTimeMs)
                        runOnUiThread { showResult(result) }
                    }
                    readyForNextImage()
                })
    }

    private fun correctImageOrientation(bitmap: Bitmap): Bitmap {
        val angle = 90 - screenOrientation
        val matrix = Matrix()
        matrix.setRotate(angle.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }

    private fun recreateClassifier() {

        LOGGER.d("Closing licenseRecognizer.")
        licenseRecognizer?.close()
        licenseRecognizer = null

        try {
            LOGGER.d("Creating licenseRecognizer")
            licenseRecognizer = LicenseRecognizer(this)
        } catch (e: IOException) {
            LOGGER.e(e, "Failed to create licenseRecognizer.")
        }
    }

    companion object {
        private val LOGGER = Logger()
    }
}
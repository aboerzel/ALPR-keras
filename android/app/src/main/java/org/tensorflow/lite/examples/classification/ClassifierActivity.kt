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

import android.graphics.*
import android.media.ImageReader
import android.os.SystemClock
import android.util.Size
import android.view.View
import org.tensorflow.lite.examples.classification.customview.OverlayView
import org.tensorflow.lite.examples.classification.env.Logger
import org.tensorflow.lite.examples.classification.tflite.LicenseRecognizer
import java.io.IOException
import kotlin.math.min
import kotlin.math.roundToInt


class ClassifierActivity : CameraActivity(), ImageReader.OnImageAvailableListener {
    private var rgbFrameBitmap: Bitmap? = null
    private var licenseRecognizer: LicenseRecognizer? = null
    private var lastProcessingTimeMs: Long = 0
    override val layoutId: Int
        get() = R.layout.camera_connection_fragment
    override val desiredPreviewFrameSize: Size?
        get() = Size(640, 480)

    private lateinit var trackingOverlay: OverlayView
    private var roi: BoundingBox = BoundingBox(0, 0, 0, 0)

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

        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView

        trackingOverlay.addCallback { canvas ->

            val boxPaint = Paint()
            boxPaint.color = Color.RED
            boxPaint.alpha = 200
            boxPaint.style = Paint.Style.STROKE
            boxPaint.strokeWidth = 4.0f

            val r = canvas.width - 2 * roi.X
            val b = canvas.height - roi.HEIGHT - roi.Y

            val trackedPos = RectF(roi.X.toFloat(), roi.Y.toFloat(), r.toFloat(), b.toFloat())
            val cornerSize = min(trackedPos.width(), trackedPos.height()) / 8.0f
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint)
        }
    }

    override fun processImage() {
        rgbFrameBitmap!!.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        var bmp = correctImageOrientation(rgbFrameBitmap!!)
        roi = getROI(bmp)
        bmp = cropROI(bmp, roi)

        trackingOverlay.postInvalidate()

        runInBackground(
                Runnable {
                    if (licenseRecognizer != null) {
                        val startTime = SystemClock.uptimeMillis()
                        val result = licenseRecognizer!!.recognize(bmp)
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime
                        LOGGER.v("Detect: %s", result)
                        LOGGER.v("Processing time: %d ms", lastProcessingTimeMs)
                        runOnUiThread { showResult(result) }
                    }
                    readyForNextImage()
                })
    }

    private fun correctImageOrientation(bitmap: Bitmap): Bitmap {
        val matrix = Matrix()
        matrix.setRotate(screenOrientationCorrectionAngle)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }

    private fun cropROI(bitmap: Bitmap, box: BoundingBox) : Bitmap {
        return Bitmap.createBitmap(bitmap, box.X, box.Y, box.WIDTH, box.HEIGHT,null, false)
    }

    private fun getROI(bitmap: Bitmap) : BoundingBox
    {
        val marginX = 40
        val width = bitmap.width - (2 * marginX)
        val ratio = licenseRecognizer!!.getRatio()
        val height = (width * ratio).roundToInt()
        val top = (bitmap.height - height) / 2

        return BoundingBox(marginX, top, width, height)
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
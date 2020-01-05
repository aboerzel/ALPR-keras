package org.tensorflow.lite.examples.classification.tflite

import android.graphics.Bitmap
import androidx.annotation.NonNull
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.TensorImage


class GlprProprocessingOp(private val targetHeight: Int, private val targetWidth: Int) : ImageOperator {
    override fun apply(@NonNull image: TensorImage): TensorImage {
        //val scaled = Bitmap.createScaledBitmap(image.bitmap, targetWidth, targetHeight, useBilinear)
        //image.load(scaled)

        val bmp = image.bitmap

        val rgb = Mat() //Size(targetHeight.toDouble(), targetWidth.toDouble()), CvType.CV_8U)
        //val bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true)
        Utils.bitmapToMat(bmp, rgb)

        val resized = Mat()
        Imgproc.resize(rgb, resized, Size(targetWidth.toDouble(), targetHeight.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

        val gray = Mat()
        Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY)

        val resultBitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888)

        Utils.matToBitmap(gray, resultBitmap)
        image.load(resultBitmap)
        return image
    }
}

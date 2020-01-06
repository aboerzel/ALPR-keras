package org.tensorflow.lite.examples.classification.tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Build.VERSION_CODES.N
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Extracts the license from an image of a car license plate as plain text
 */
class LicenseRecognizer @Throws(IOException::class)
constructor(context: Context) {

    // TensorFlow Lite interpreter for running inference with the tflite model
    private var interpreter: Interpreter

    private var inputImageBuffer: TensorBuffer
    private var outputProbabilityBuffer: TensorBuffer

    // Initialize TFLite interpreter
    init {

        // Load TFLite model
        val assetManager = context.assets
        val model = loadModelFile(assetManager)

        // Configure TFLite Interpreter options
        val options = Interpreter.Options()
        options.setNumThreads(3)
        //options.setUseNNAPI(true)

        // Create & initialize TFLite interpreter
        interpreter = Interpreter(model, options)

        // Reads type and shape of input and output tensors, respectively.
        val imageTensorIndex = 0
        val imageShape = interpreter.getInputTensor(imageTensorIndex).shape() // {DIM_BATCH_SIZE, DIM_INPUT_WIDTH, DIM_INPUT_HEIGHT, DIM_INPUT_DEPTH}
        val imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType()

        // Creates the input tensor.
        inputImageBuffer = TensorBuffer.createFixedSize(imageShape, imageDataType)

        val outputTensorIndex = 0
        val outputShape = interpreter.getOutputTensor(outputTensorIndex).shape()
        val outputDataType = interpreter.getOutputTensor(outputTensorIndex).dataType()

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(intArrayOf(DIM_BATCH_SIZE, TEXT_LENGTH, ALPHABET_LENGTH), outputDataType)
    }

    // Memory-map the model file in Assets
    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * To classify an image, follow these steps:
     * 1. pre-process the input image
     * 2. run inference with the model
     * 3. post-process the output result for displaying in UI
     *
     * @param bitmap
     * @return car licesne as plain text
     */
    fun classify(bitmap: Bitmap): String {

        // 1. Pre-processing
        val inputByteBuffer = preprocess(bitmap)

        // 2. Run inference
        interpreter.run(inputByteBuffer, outputProbabilityBuffer.getBuffer().rewind())

        // 3. Post-processing
        return postprocess(outputProbabilityBuffer)
    }

    fun close() {
        interpreter.close()
    }

    /**
     * Preprocess the bitmap by converting it to ByteBuffer & grayscale
     *
     * @param bitmap
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {

        val rgb = Mat()
        Utils.bitmapToMat(bitmap, rgb)

        val resized = Mat()
        Imgproc.resize(rgb, resized, Size(DIM_INPUT_WIDTH.toDouble(), DIM_INPUT_HEIGHT.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)

        val gray = Mat()
        Imgproc.cvtColor(resized, gray, Imgproc.COLOR_BGR2GRAY)

        val resultBitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888)

        Utils.matToBitmap(gray, resultBitmap)

        return convertBitmapToByteBuffer(resultBitmap)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        // Create input image bytebuffer
        val byteBuffer = ByteBuffer.allocateDirect(4
                * DIM_BATCH_SIZE    // 1
                * DIM_INPUT_WIDTH   // 128
                * DIM_INPUT_HEIGHT  // 64
                * DIM_INPUT_DEPTH)   // 1
        byteBuffer.order(ByteOrder.nativeOrder())

        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        val initialStream = ByteArrayInputStream(stream.toByteArray())
        while (initialStream.available() > 0) {
            byteBuffer.put(initialStream.read().toByte())
        }

        return byteBuffer
    }

    /**
     * Find digit prediction with the highest probability
     *
     * @return
     */
    private fun postprocess(outputArray: TensorBuffer): String {
        val shape = outputArray.shape
        val floatArray = outputArray.getFloatArray()

        var best = IntArray(shape[1])
        for (i in 0 until shape[1]) {
            var max = -1
            for (j in 0 until shape[2]) {
                if (floatArray[i+j] > max)
                    max = j
            }
            best[i] = max
        }

        //out_best = [k for k, g in itertools.groupby(argmax)]
        var result = ""
        for (c in best) {
            if (c < ALPHABET.length && c >= 0) {
                result += ALPHABET[c]
            }
        }

        return result
    }

    companion object {

        private val LOG_TAG = LicenseRecognizer::class.java.simpleName

        private val ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789- "

        // Name of the model file (under /assets folder)
        private val MODEL_PATH = "glpr-model.tflite"

        // Input size
        private val DIM_BATCH_SIZE = 1      // batch size
        private val DIM_INPUT_WIDTH = 128   // input image width
        private val DIM_INPUT_HEIGHT = 64   // input image height
        private val DIM_INPUT_DEPTH = 1     // 1 for gray scale & 3 for color images

        /* Output*/
        private val TEXT_LENGTH = 32
        private val ALPHABET_LENGTH = 42
    }
}

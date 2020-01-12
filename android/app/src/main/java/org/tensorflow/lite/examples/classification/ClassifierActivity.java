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

package org.tensorflow.lite.examples.classification;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Matrix;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;

import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.LicenseRecognizer;

import java.io.IOException;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private Bitmap rgbFrameBitmap = null;
  private LicenseRecognizer licenseRecognizer;
  private long lastProcessingTimeMs;

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

    recreateClassifier();
    if (licenseRecognizer == null) {
      LOGGER.e("No licenseRecognizer on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    /*
    Mat image = null;
    try {
      image = Utils.loadResource(this, R.drawable.license1);
    } catch (IOException e) {
      e.printStackTrace();
    }
    */

    runInBackground(
            () -> {
              if (licenseRecognizer != null) {
                final long startTime = SystemClock.uptimeMillis();
                String result = licenseRecognizer.recognize(correctImageOrientation(rgbFrameBitmap));
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.v("Detect: %s", result);
                LOGGER.v("Processing time: %d ms", lastProcessingTimeMs);

                runOnUiThread(
                        () -> showResult(result));
              }
              readyForNextImage();
            });
  }

  private Bitmap correctImageOrientation(Bitmap bitmap) {
    int angle = 90 - getScreenOrientation();
    Matrix matrix = new Matrix();
    matrix.setRotate(angle);
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, false);
  }

  private void recreateClassifier() {
    if (licenseRecognizer != null) {
      LOGGER.d("Closing licenseRecognizer.");
      licenseRecognizer.close();
      licenseRecognizer = null;
    }

    try {
      LOGGER.d("Creating licenseRecognizer");
      licenseRecognizer = new LicenseRecognizer(this);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create licenseRecognizer.");
    }
  }
}

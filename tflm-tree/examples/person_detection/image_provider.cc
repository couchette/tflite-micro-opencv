/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "image_provider.h"

#include "model_settings.h"

CameraManager::CameraManager(int device_id, int image_width, int image_height)
    : cap(device_id), width(image_width), height(image_height) {
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open camera");
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

CameraManager::~CameraManager() {
    cap.release();
}

TfLiteStatus CameraManager::GetImage(int channels, int8_t* image_data) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        return kTfLiteError;
    }

    cv::resize(frame, frame, cv::Size(width, height));

    if (channels == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }

    int index = 0;
    for (int i = 0; i < frame.rows; ++i) {
        for (int j = 0; j < frame.cols; ++j) {
            if (channels == 1) {
                image_data[index++] = frame.at<uint8_t>(i, j) - 128;
            } else {
                cv::Vec3b pixel = frame.at<cv::Vec3b>(i, j);
                image_data[index++] = pixel[0] - 128;  // Blue channel
                image_data[index++] = pixel[1] - 128;  // Green channel
                image_data[index++] = pixel[2] - 128;  // Red channel
            }
        }
    }

    return kTfLiteOk;
}

TfLiteStatus GetImage(int image_width, int image_height, int channels,
                      int8_t* image_data) {
  for (int i = 0; i < image_width * image_height * channels; ++i) {
    image_data[i] = 0;
  }
  return kTfLiteOk;
}

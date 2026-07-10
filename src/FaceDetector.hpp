#pragma once

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

struct FaceResult {
    cv::Rect box;
    float score = 0.0f;
    cv::Point center;
    std::vector<cv::Point> landmarks;
};

struct DetectorConfig {
    std::string modelPath = "models/face_detection_yunet_2023mar.onnx";
    float scoreThreshold = 0.85f;
    float nmsThreshold = 0.30f;
    int topK = 5000;
};

class FaceDetector {
public:
    bool load(const DetectorConfig& config, cv::Size inputSize, std::string& error);
    void setInputSize(cv::Size inputSize);
    std::vector<FaceResult> detect(const cv::Mat& frame);
    const DetectorConfig& config() const { return config_; }

private:
    DetectorConfig config_;
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Size inputSize_;
    std::vector<cv::Rect> previousBoxes_;

    cv::Rect smoothBox(std::size_t index, const cv::Rect& current);
};

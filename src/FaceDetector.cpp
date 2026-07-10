#include "FaceDetector.hpp"

#include <algorithm>
#include <filesystem>

bool FaceDetector::load(const DetectorConfig& config, cv::Size inputSize, std::string& error) {
    config_ = config;
    inputSize_ = inputSize;

    if (!std::filesystem::exists(config_.modelPath)) {
        error = "Model not found: " + config_.modelPath;
        return false;
    }

    detector_ = cv::FaceDetectorYN::create(
        config_.modelPath,
        "",
        inputSize_,
        config_.scoreThreshold,
        config_.nmsThreshold,
        config_.topK
    );

    if (detector_.empty()) {
        error = "Unable to create YuNet detector.";
        return false;
    }

    return true;
}

void FaceDetector::setInputSize(cv::Size inputSize) {
    if (!detector_.empty() && inputSize != inputSize_) {
        inputSize_ = inputSize;
        detector_->setInputSize(inputSize_);
    }
}

std::vector<FaceResult> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<FaceResult> results;
    if (detector_.empty() || frame.empty()) {
        return results;
    }

    setInputSize(frame.size());
    cv::Mat faces;
    detector_->detect(frame, faces);

    for (int row = 0; row < faces.rows; ++row) {
        const float* data = faces.ptr<float>(row);
        const float score = data[14];
        if (score < config_.scoreThreshold) {
            continue;
        }

        cv::Rect box(
            cv::saturate_cast<int>(data[0]),
            cv::saturate_cast<int>(data[1]),
            cv::saturate_cast<int>(data[2]),
            cv::saturate_cast<int>(data[3])
        );
        box &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.empty()) {
            continue;
        }

        FaceResult result;
        result.box = smoothBox(results.size(), box);
        result.score = score;
        result.center = cv::Point(result.box.x + result.box.width / 2, result.box.y + result.box.height / 2);

        for (int i = 0; i < 5; ++i) {
            result.landmarks.emplace_back(
                cv::saturate_cast<int>(data[4 + i * 2]),
                cv::saturate_cast<int>(data[5 + i * 2])
            );
        }

        results.push_back(result);
    }

    previousBoxes_.resize(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        previousBoxes_[i] = results[i].box;
    }

    return results;
}

cv::Rect FaceDetector::smoothBox(std::size_t index, const cv::Rect& current) {
    if (index >= previousBoxes_.size()) {
        return current;
    }

    const cv::Rect& previous = previousBoxes_[index];
    const double alpha = 0.65;
    return cv::Rect(
        cv::saturate_cast<int>(previous.x * alpha + current.x * (1.0 - alpha)),
        cv::saturate_cast<int>(previous.y * alpha + current.y * (1.0 - alpha)),
        cv::saturate_cast<int>(previous.width * alpha + current.width * (1.0 - alpha)),
        cv::saturate_cast<int>(previous.height * alpha + current.height * (1.0 - alpha))
    );
}

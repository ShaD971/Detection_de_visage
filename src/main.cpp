#include "FaceDetector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

struct Options {
    int camera = 0;
    std::string model = "models/face_detection_yunet_2023mar.onnx";
    float confidence = 0.85f;
    int width = 1280;
    int height = 720;
    bool mirror = false;
    bool blur = false;
    bool pixelate = false;
    bool help = false;
    bool legacyHaar = false;
    std::string snapshotDir = "snapshots";
    std::string imagePath;
    std::string videoPath;
};

void printHelp() {
    std::cout
        << "Usage: face_detection [options]\n\n"
        << "Options:\n"
        << "  --camera N              Camera index, default 0\n"
        << "  --model PATH            YuNet ONNX model path\n"
        << "  --confidence VALUE      Score threshold, default 0.85\n"
        << "  --width VALUE           Capture width, default 1280\n"
        << "  --height VALUE          Capture height, default 720\n"
        << "  --mirror                Mirror camera frame\n"
        << "  --blur                  Blur detected faces\n"
        << "  --pixelate              Pixelate detected faces\n"
        << "  --snapshot-dir PATH     Directory for S key snapshots\n"
        << "  --image PATH            Detect faces in one image\n"
        << "  --video PATH            Detect faces in a video file\n"
        << "  --legacy-haar           Print legacy Haar information\n"
        << "  --help                  Show this help\n\n"
        << "Keys: S snapshot, B blur, P pause, H help, Q/Esc quit\n";
}

bool parseArgs(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                error = "Missing value for " + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help") options.help = true;
        else if (arg == "--mirror") options.mirror = true;
        else if (arg == "--blur") options.blur = true;
        else if (arg == "--pixelate") options.pixelate = true;
        else if (arg == "--legacy-haar") options.legacyHaar = true;
        else if (arg == "--camera") { const char* v = requireValue(arg); if (!v) return false; options.camera = std::stoi(v); }
        else if (arg == "--model") { const char* v = requireValue(arg); if (!v) return false; options.model = v; }
        else if (arg == "--confidence") { const char* v = requireValue(arg); if (!v) return false; options.confidence = std::stof(v); }
        else if (arg == "--width") { const char* v = requireValue(arg); if (!v) return false; options.width = std::stoi(v); }
        else if (arg == "--height") { const char* v = requireValue(arg); if (!v) return false; options.height = std::stoi(v); }
        else if (arg == "--snapshot-dir") { const char* v = requireValue(arg); if (!v) return false; options.snapshotDir = v; }
        else if (arg == "--image") { const char* v = requireValue(arg); if (!v) return false; options.imagePath = v; }
        else if (arg == "--video") { const char* v = requireValue(arg); if (!v) return false; options.videoPath = v; }
        else { error = "Unknown option: " + arg; return false; }
    }

    if (options.blur && options.pixelate) {
        error = "--blur and --pixelate cannot be used together.";
        return false;
    }
    if (!options.imagePath.empty() && !options.videoPath.empty()) {
        error = "--image and --video cannot be used together.";
        return false;
    }
    if (options.confidence < 0.0f || options.confidence > 1.0f) {
        error = "--confidence must be between 0 and 1.";
        return false;
    }
    return true;
}

void applyPrivacy(cv::Mat& frame, const cv::Rect& box, bool blur, bool pixelate) {
    cv::Mat roi = frame(box);
    if (blur) {
        cv::GaussianBlur(roi, roi, cv::Size(45, 45), 0);
    } else if (pixelate) {
        cv::Mat small;
        cv::resize(roi, small, cv::Size(12, 12), 0, 0, cv::INTER_LINEAR);
        cv::resize(small, roi, roi.size(), 0, 0, cv::INTER_NEAREST);
    }
}

void drawDetections(cv::Mat& frame, const std::vector<FaceResult>& faces, bool blur, bool pixelate) {
    for (const FaceResult& face : faces) {
        applyPrivacy(frame, face.box, blur, pixelate);
        cv::rectangle(frame, face.box, cv::Scalar(0, 220, 255), 2);
        cv::circle(frame, face.center, 3, cv::Scalar(255, 255, 255), cv::FILLED);
        for (const cv::Point& point : face.landmarks) {
            cv::circle(frame, point, 3, cv::Scalar(0, 255, 0), cv::FILLED);
        }

        std::ostringstream label;
        label << "score " << std::fixed << std::setprecision(2) << face.score;
        cv::putText(frame, label.str(), face.box.tl() + cv::Point(0, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 220, 255), 2);
    }
}

std::string timestampedPath(const std::string& directory) {
    std::filesystem::create_directories(directory);
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &seconds);
#else
    localtime_r(&seconds, &tm);
#endif
    std::ostringstream name;
    name << directory << "/snapshot_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
    return name.str();
}

int runImage(const Options& options, FaceDetector& detector) {
    cv::Mat frame = cv::imread(options.imagePath);
    if (frame.empty()) {
        std::cerr << "Unable to read image: " << options.imagePath << "\n";
        return 1;
    }
    detector.setInputSize(frame.size());
    const auto faces = detector.detect(frame);
    drawDetections(frame, faces, options.blur, options.pixelate);
    std::cout << "Faces detected: " << faces.size() << "\n";
    cv::imshow("Face detection", frame);
    cv::waitKey(0);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    if (!parseArgs(argc, argv, options, error)) {
        std::cerr << error << "\n";
        printHelp();
        return 1;
    }
    if (options.help) {
        printHelp();
        return 0;
    }
    if (options.legacyHaar) {
        std::cout << "Legacy Haar demo kept in legacy/detecteFace_haar.cpp\n";
        return 0;
    }

    DetectorConfig detectorConfig;
    detectorConfig.modelPath = options.model;
    detectorConfig.scoreThreshold = options.confidence;
    detectorConfig.nmsThreshold = 0.30f;
    detectorConfig.topK = 5000;

    FaceDetector detector;
    if (!detector.load(detectorConfig, cv::Size(options.width, options.height), error)) {
        std::cerr << error << "\nDownload the official model from OpenCV Zoo into models/.\n";
        return 1;
    }

    if (!options.imagePath.empty()) {
        return runImage(options, detector);
    }

    cv::VideoCapture capture;
    if (!options.videoPath.empty()) {
        capture.open(options.videoPath);
    } else {
        capture.open(options.camera);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
    }
    if (!capture.isOpened()) {
        std::cerr << "Unable to open camera or video source.\n";
        return 1;
    }

    bool paused = false;
    bool showHelp = false;
    bool blur = options.blur;
    bool pixelate = options.pixelate;
    cv::Mat frame;
    double fps = 0.0;
    auto last = std::chrono::steady_clock::now();

    while (true) {
        if (!paused) {
            if (!capture.read(frame) || frame.empty()) {
                break;
            }
            if (options.mirror) {
                cv::flip(frame, frame, 1);
            }

            const auto faces = detector.detect(frame);
            drawDetections(frame, faces, blur, pixelate);

            const auto now = std::chrono::steady_clock::now();
            const double currentFps = 1000.0 / std::max(1.0, static_cast<double>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count()));
            fps = fps == 0.0 ? currentFps : fps * 0.9 + currentFps * 0.1;
            last = now;

            cv::putText(frame, "Faces: " + std::to_string(faces.size()), {16, 28},
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), {16, 58},
                        cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
        }

        if (showHelp) {
            cv::putText(frame, "S snapshot | B blur | P pause | H help | Q/Esc quit", {16, frame.rows - 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        cv::imshow("Face detection", frame);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key == 'p' || key == 'P') paused = !paused;
        if (key == 'h' || key == 'H') showHelp = !showHelp;
        if (key == 'b' || key == 'B') { blur = !blur; if (blur) pixelate = false; }
        if (key == 's' || key == 'S') {
            const std::string path = timestampedPath(options.snapshotDir);
            cv::imwrite(path, frame);
            std::cout << "Snapshot saved: " << path << "\n";
        }
    }

    return 0;
}

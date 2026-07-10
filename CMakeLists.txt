cmake_minimum_required(VERSION 3.16)

project(face_detection_opencv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(face_detection
    src/main.cpp
    src/FaceDetector.cpp
)

target_include_directories(face_detection PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    src
)

target_link_libraries(face_detection PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(face_detection PRIVATE /W4)
else()
    target_compile_options(face_detection PRIVATE -Wall -Wextra -Wpedantic)
endif()

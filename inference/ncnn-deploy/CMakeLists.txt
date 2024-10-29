cmake_minimum_required(VERSION 3.5)

project(deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
set(OpenCV_DIR "D:/home/3rdparty/opencv48/x64/vc16/lib")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("FOUND opencv")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
# ncnn
set(NCNN_DIR "D:/home/3rdparty/ncnn")
include_directories(${NCNN_DIR}/include)
set(NCNN_STATIC ${NCNN_DIR}/lib/ncnn.lib -fopenmp)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SRC_FILES
    ${SRC_DIR}/*.h
    ${SRC_DIR}/*.hpp
    ${SRC_DIR}/*.cpp)
add_executable(deploy ${SRC_FILES})

target_link_libraries(deploy PRIVATE ${OpenCV_LIBS} ${NCNN_STATIC})

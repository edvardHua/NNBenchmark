cmake_minimum_required(VERSION 3.4.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/libs/ncnn-vulkan-0524/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnn_wrapper SHARED src/main/cpp/ncnn_classifier_jni.cpp)

target_link_libraries(ncnn_wrapper
        ncnn

        jnigraphics
        )



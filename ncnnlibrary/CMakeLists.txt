project(styletransferncnn)

cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_SOURCE_DIR}/libs/ncnn-cpu-0507/${ANDROID_ABI}/include/ncnn)
#include_directories(${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/include/glslang)
#include_directories(${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/include/glslang/Include)
#include_directories(${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/include/glslang/MachineIndependent)
#include_directories(${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/include/glslang/Public)
#include_directories(${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/include/glslang/SPIRV)


# import ncnn library
add_library(ncnn STATIC IMPORTED)

#add_library(glslang STATIC IMPORTED)
#add_library(OGLCompiler STATIC IMPORTED)
#add_library(OSDependent STATIC IMPORTED)
#add_library(SPIRV STATIC IMPORTED)
#
#add_library(Generic STATIC IMPORTED)
#add_library(Machine STATIC IMPORTED)

#set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libglslang.a)
#set_target_properties(OGLCompiler PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libOGLCompiler.a)
#set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libOSDependent.a)
#set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libSPIRV.a)
#
#set_target_properties(Generic PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libGenericCodeGen.a)
#set_target_properties(Machine PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan-lib/${ANDROID_ABI}/lib/libMachineIndependent.a)

set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/ncnn-cpu-0507/${ANDROID_ABI}/lib/libncnn.a)

# openmp
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")

if(DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 24)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffast-math")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

# disable rtti and exceptions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

add_library(ncnn_wrapper SHARED src/main/cpp/ncnn_classifier_jni.cpp)

target_link_libraries(ncnn_wrapper -static-openmp -fopenmp
        ncnn

        #        Generic Machine glslang SPIRV OGLCompiler OSDependent

        android
        z
        log
        jnigraphics
        #        vulkan
        )

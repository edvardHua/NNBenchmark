//
// Created by tencent on 2020-04-30.
//
#include "Classifier.h"
#include "classifier_jni.h"
#include "helper_jni.h"
#include "tnn_sdk.h"
#include "kannarotate.h"
#include "yuv420sp_to_rgb_fast_asm.h"
#include <android/bitmap.h>

static std::shared_ptr<TNN_NS::Classifier> gDetector;
static int gComputeUnitType = 0;

JNIEXPORT JNICALL jint
TNN_CLASSIFY(init)(JNIEnv *env, jobject thisObj, jstring modelPath, jint computeUnitType) {
    gDetector = std::make_shared<TNN_NS::Classifier>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + ".tnnproto");
    modelContent = fdLoadFile(modelPathStr + ".tnnmodel");
    LOGI("proto content size %d model content size %d", (int) protoContent.length(),
         (int) modelContent.length());
    TNN_NS::Status status;
    gComputeUnitType = computeUnitType;
    if (gComputeUnitType == 0) {
        status = gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsCPU);
    } else {
        status = gDetector->Init(protoContent, modelContent, "", TNN_NS::TNNComputeUnitsGPU);
    }

    if (status != TNN_NS::TNN_OK) {
        LOGE("detector init failed %d", (int) status);
        return -1;
    }
    return 0;
}

JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thisObj) {
    gDetector = nullptr;
    return 0;
}

JNIEXPORT JNICALL jint
TNN_CLASSIFY(predictFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width,
                                jint height, jint rotate) {
    unsigned char *yuvData = new unsigned char[height * width * 3 / 2];
    jbyte *yuvDataRef = env->GetByteArrayElements(yuv420sp, 0);
    int ret = kannarotate_yuv420sp((const unsigned char *) yuvDataRef, (int) width, (int) height,
                                   (unsigned char *) yuvData, (int) rotate);
    env->ReleaseByteArrayElements(yuv420sp, yuvDataRef, 0);
    unsigned char *rgbaData = new unsigned char[height * width * 4];
    yuv420sp_to_rgba_fast_asm((const unsigned char *) yuvData, height, width,
                              (unsigned char *) rgbaData);

    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 4, height, width};

    auto rgbTNN = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims, rgbaData);

    TNN_NS::Status status = gDetector->infer(rgbTNN);

    if (status != TNN_NS::TNN_OK) {
        return 1;
    }

    delete[] yuvData;
    delete[] rgbaData;
    return 0;
}

JNIEXPORT JNICALL jint
TNN_CLASSIFY(predictFromBitmap)(JNIEnv *env, jobject thiz, jobject bmp, jint width, jint height) {
    AndroidBitmapInfo sourceInfocolor;
    void *sourcePixelscolor;

    if (AndroidBitmap_getInfo(env, bmp, &sourceInfocolor) < 0) {
        return 0;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return 0;
    }

    if (AndroidBitmap_lockPixels(env, bmp, &sourcePixelscolor) < 0) {
        return 0;
    }

    std::vector<int> origin_dims = {1, 4, height, width};
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, origin_dims,
                                                   sourcePixelscolor);

    TNN_NS::Status status = gDetector->infer(input_mat);

    if (status != TNN_NS::TNN_OK) {
        return 0;
    }

    return 1;
}
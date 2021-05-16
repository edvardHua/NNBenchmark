//
// Created by tencent on 2020-04-30.
//

#ifndef ANDROID_IMAGECLASSIFY_JNI_H
#define ANDROID_IMAGECLASSIFY_JNI_H
#include <jni.h>
#define TNN_CLASSIFY(sig) Java_com_tencent_tnn_interpret_Classifier_##sig
#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT JNICALL jint TNN_CLASSIFY(init)(JNIEnv *env, jobject thisObj, jstring modelPath, jint computeUnitType);
JNIEXPORT JNICALL jint TNN_CLASSIFY(deinit)(JNIEnv *env, jobject thisObj);
JNIEXPORT JNICALL jint TNN_CLASSIFY(predictFromStream)(JNIEnv *env, jobject thiz, jbyteArray yuv420sp, jint width, jint height, jint rotate);
JNIEXPORT JNICALL jint TNN_CLASSIFY(predictFromBitmap)(JNIEnv *env, jobject thiz, jobject bmp, jint width, jint height);
#ifdef __cplusplus
}
#endif
#endif //ANDROID_IMAGECLASSIFY_JNI_H

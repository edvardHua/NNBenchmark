//
// Created by edvardzeng on 2021/5/16.
//

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net mobilenet;
static ncnn::Net mobilenet_int8;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_edvardzeng_ncnnlibrary_NCNNINterpreter_Init(JNIEnv *env, jobject thiz, jobject assetManager) {

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    // float model
    mobilenet.load_param(mgr, "mobilenet_v2_opt.param");
    mobilenet.load_model(mgr, "mobilenet_v2_opt.bin");

    mobilenet_int8.load_param(mgr, "mobilenet_v2_int8.param");
    mobilenet_int8.load_model(mgr, "mobilenet_v2_int8.bin");

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_edvardzeng_ncnnlibrary_NCNNINterpreter_ClassifierNcnn(JNIEnv *env, jobject thiz,
                                                               jobject bitmap, jboolean use_gpu, jboolean use_int8) {
    // TODO: implement MobilenetNcnn()

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return JNI_FALSE;

    // seg infer
    ncnn::Mat inS = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB2BGR, 224,
                                                          224);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
    inS.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Mat out2;
    {
        ncnn::Extractor ex = mobilenet.create_extractor();
        if (use_int8){
            ex = mobilenet_int8.create_extractor();
        }

        if(use_gpu){
            ex.set_vulkan_compute(true);
        }else{
            ex.set_num_threads(4);
            ex.set_vulkan_compute(false);
        }

        ex.input("data", inS);
        ex.extract("prob", out2);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "MobilenetNcnn", "%.2fms   MobilenetNcnn", elasped);

    return JNI_TRUE;
}
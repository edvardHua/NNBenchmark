package com.edvardzeng.ncnnlibrary;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

/**
 * description :
 * author : edvardzeng
 * email : edvardzeng@tencent.com
 * date : 2021/5/16 12:30
 */
public class NCNNINterpreter {

	public native boolean Init(AssetManager mgr);
	public native boolean ClassifierNcnn(Bitmap bitmap, boolean use_gpu);

	static {
		try {
			System.loadLibrary("ncnn_wrapper");
		} catch (Exception e) {
		} catch (Error e) {
		} finally {
		}
	}
}

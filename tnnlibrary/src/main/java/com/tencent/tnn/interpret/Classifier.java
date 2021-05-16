package com.tencent.tnn.interpret;

import android.graphics.Bitmap;

public class Classifier {
	static {
		try {
			System.loadLibrary("tnn_wrapper");
		} catch (Exception e) {
		} catch (Error e) {
		} finally {
		}
	}

	public native int init(String modelPath, int computeUnitType);

	public native int deinit();

	public native int predictFromStream(byte[] yuv420sp, int width, int height, int rotate);

	public native int predictFromBitmap(Bitmap bmp, int width, int height);
}

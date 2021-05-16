package com.tencent.tnn.interpret;

import android.content.Context;
import android.graphics.Bitmap;

/**
 * description :
 * author : edvardzeng
 * email : edvardzeng@tencent.com
 * date : 2021/1/5 16:29
 */
public class TNNInterpreter {

	private static final Logger LOGGER = new Logger();

	private Classifier mClassifier = new Classifier();

	public TNNInterpreter(Context context){
		this(context, false);
	}

	public TNNInterpreter(Context context, boolean useGpu) {
		String targetDir = context.getFilesDir().getAbsolutePath();
		String[] modelPathsDetector = {
				"Models/mobilenet_v2",
				"Models/model.quantized"
		};

		String[] interModelFilePath = new String[2];
		for (int i = 0; i < modelPathsDetector.length; i++) {
			String modelFilePath = modelPathsDetector[i];
			String interPath = targetDir + "/" + modelFilePath.replace("/", "_");
			FileUtils.copyAsset(context.getAssets(), modelFilePath + ".tnnmodel", interPath + ".tnnmodel");
			FileUtils.copyAsset(context.getAssets(), modelFilePath + ".tnnproto", interPath + ".tnnproto");
			interModelFilePath[i] = interPath;
		}
		int flag = 0;
		if (useGpu) {
			flag = 1;
		}

		// 0 是 float 模型，1 是 int8 模型
		int initCode = mClassifier.init(interModelFilePath[1], flag);

		if (initCode != 0) {
			throw new IllegalStateException("Init model failed.");
		}
		LOGGER.i("CreateEngine result = %d", initCode);
	}


	public void infer(byte[] yuvData, int width, int height) {
		/**
		 * @param yuvData: byte 类型的 yuv 图像数据
		 * @param width: 图像数据的宽
		 * @param height: 图像数据的高
		 */
		int code = mClassifier.predictFromStream(yuvData, width, height, 1);
	}

	public void inferBmp(Bitmap bmp, int width, int height) {
		int code = mClassifier.predictFromBitmap(bmp, width, height);
	}

	public void dinit() {
		if (mClassifier != null) {
			mClassifier.deinit();
		}
	}
}

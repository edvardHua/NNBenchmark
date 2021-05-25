package com.edvardzeng.nnbenchmark;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.edvardzeng.ncnnlibrary.NCNNINterpreter;
import com.tencent.tnn.interpret.TNNInterpreter;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;

/**
 * description :
 * author : edvardzeng
 * email : edvardzeng@tencent.com
 * date : 2021/5/15 10:14
 * <p>
 * 简单用来测试 NCNN 和 TNN 运行 MobileNet V2 的速度
 * 模型放在 library 对应的 assets 里面
 */
public class MainActivity extends AppCompatActivity {

	private TextView tv;
	private Bitmap img;
	private int TEST_NUMS = 100;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		tv = findViewById(R.id.tvResult);
		img = BitmapFactory.decodeResource(getResources(), R.drawable.test);

		new Thread(new Runnable() {
			public void run() {
				try {
					// 不能同时执行
//					testBmpTNN();
//					testBmpNCNN();
					testTFlite();
				} catch (InterruptedException | IOException e) {
					e.printStackTrace();
				}
			}
		}).start();
	}

	public void testBmpTNN() throws InterruptedException {
		TNNInterpreter tm = new TNNInterpreter(getApplicationContext(), true, false);
		// GPU test
		tm.inferBmp(img, 224, 224);
		Long st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			tm.inferBmp(img, 224, 224);
		}
		Long costGPU = System.currentTimeMillis() - st;
		tm.dinit();

		// cpu test
		tm = new TNNInterpreter(getApplicationContext(), false, false);
		tm.inferBmp(img, 224, 224);
		st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			tm.inferBmp(img, 224, 224);
		}
		Long costCPU = System.currentTimeMillis() - st;
		tm.dinit();

		// int8 test
		tm = new TNNInterpreter(getApplicationContext(), false, true);
		tm.inferBmp(img, 224, 224);
		st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			tm.inferBmp(img, 224, 224);
		}
		Long costInt8 = System.currentTimeMillis() - st;
		tm.dinit();

		runOnUiThread(new Runnable() {
			@Override
			public void run() {

				tv.append("\n\n------------ TNN Benchmark ---------------- \n");

				tv.append("CPU avg. cost = " + (costCPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("INT8 avg. cost = " + (costInt8 - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("GPU avg. cost = " + (costGPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
			}
		});

	}

	public void testBmpNCNN() throws InterruptedException {
		NCNNINterpreter ni = new NCNNINterpreter();
		ni.Init(getAssets());

		// warm up 一下
		ni.ClassifierNcnn(img, false, false);
		ni.ClassifierNcnn(img, true, false);

		Long st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			ni.ClassifierNcnn(img, true, false);
		}
		Long costGPU = System.currentTimeMillis() - st;

		st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			ni.ClassifierNcnn(img, false, false);
		}
		Long costCPU = System.currentTimeMillis() - st;

		st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			ni.ClassifierNcnn(img, false, true);
		}
		Long costInt8 = System.currentTimeMillis() - st;

		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				tv.append("\n\n------------ NCNN Benchmark ---------------- \n");
				tv.append("CPU avg. cost = " + (costCPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("INT8 avg. cost = " + (costInt8 - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("GPU avg. cost = " + (costGPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
			}
		});
	}

	public Long runTFLite(boolean cpu, boolean quant) throws IOException, InterruptedException {
		Interpreter.Options tfliteOptions = new Interpreter.Options();
		MappedByteBuffer tfliteModel;
		if (quant) {
			tfliteModel = FileUtil.loadMappedFile(this, "mobilenet_v2_1.0_224_quant.tflite");
		} else {
			tfliteModel = FileUtil.loadMappedFile(this, "mobilenet_v2_1.0_224.tflite");
		}

		tfliteOptions.setNumThreads(4);
		if (!cpu) {
			tfliteOptions.addDelegate(new GpuDelegate());
		}
		Interpreter tflite = new Interpreter(tfliteModel, tfliteOptions);

		// Reads type and shape of input and output tensors, respectively.
		int imageTensorIndex = 0;
		int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
		DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
		int probabilityTensorIndex = 0;
		int[] probabilityShape =
				tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
		DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

		// Creates the input tensor.
		TensorImage inputImageBuffer = new TensorImage(imageDataType);
		// Loads bitmap into a TensorImage.
		inputImageBuffer.load(img);

		// Creates processor for the TensorImage.
		int cropSize = Math.min(img.getWidth(), img.getHeight());

		ImageProcessor imageProcessor;
		if (!quant) {
			imageProcessor =
					new ImageProcessor.Builder()
							.add(new ResizeWithCropOrPadOp(cropSize, cropSize))
							.add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
							.add(new NormalizeOp(127.5f, 127.5f))
							.build();
		} else {
			imageProcessor =
					new ImageProcessor.Builder()
							.add(new ResizeWithCropOrPadOp(cropSize, cropSize))
							.add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
							.build();
		}

		TensorImage inputImage = imageProcessor.process(inputImageBuffer);
		TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

		// warmup
		tflite.run(inputImage.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
		Long st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i++) {
			Thread.sleep(41);
			tflite.run(inputImage.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
		}
		Long cost = System.currentTimeMillis() - st;
		return cost;
	}

	public void testTFlite() throws IOException, InterruptedException {

		Long costCPU = runTFLite(true, false);
		Long costGPU = runTFLite(false, false);
		Long costInt8 = runTFLite(true, false);

		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				tv.append("\n\n------------ TFLite Benchmark ---------------- \n");
				tv.append("CPU avg. cost = " + (costCPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("INT8 avg. cost = " + (costInt8 - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
				tv.append("GPU avg. cost = " + (costGPU - 41 * TEST_NUMS) / TEST_NUMS + " ms. \n");
			}
		});

	}

}
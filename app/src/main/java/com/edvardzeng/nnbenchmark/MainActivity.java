package com.edvardzeng.nnbenchmark;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;

import com.edvardzeng.ncnnlibrary.NCNNINterpreter;
import com.tencent.tnn.interpret.TNNInterpreter;

import java.nio.ByteBuffer;

/**
 * description :
 * author : edvardzeng
 * email : edvardzeng@tencent.com
 * date : 2021/5/15 10:14
 *
 * 简单用来测试 NCNN 和 TNN 运行 MobileNet V2 的速度
 * 模型放在 library 对应的 assets 里面
 *
 */
public class MainActivity extends AppCompatActivity {

	private TextView tv;
	private Bitmap img;
	private int TEST_NUMS = 200;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		tv = findViewById(R.id.tvResult);
		img = BitmapFactory.decodeResource(getResources(), R.drawable.test);

		new Thread(new Runnable() {
			public void run() {
				testBmpNCNN();
//				testBmpTNN();
			}
		}).start();
	}

	public Long testBmpTNN() {
		TNNInterpreter tm = new TNNInterpreter(getApplicationContext(), false);
		// 使用 Int8 模型，会显示 model 初始化的错误
		// 模型切换和加载在 TNNInterpreter 里
		// warm up
		tm.inferBmp(img, 224, 224);

		Long st = System.currentTimeMillis();
		for(int i = 0; i < TEST_NUMS; i ++){
			tm.inferBmp(img, 224, 224);
		}
		Long cost = System.currentTimeMillis() - st;
		tm.dinit();
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				tv.setText("TNN acg. cost = " + cost / TEST_NUMS + " ms.");
			}
		});
		return cost;
	}

	public Long testBmpNCNN() {
		NCNNINterpreter ni = new NCNNINterpreter();
		ni.Init(getAssets());
		// CAUTION: 目前用的是 cpu 版本的包，无 GPU
		// 但是使用 int8 模型，会报错
		// warm up 一下
		ni.ClassifierNcnn(img, false);

		Long st = System.currentTimeMillis();
		for (int i = 0; i < TEST_NUMS; i ++){
			ni.ClassifierNcnn(img, false);
		}
		Long cost = System.currentTimeMillis() - st;
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				tv.setText("NCNN avg. cost = " + cost / TEST_NUMS + " ms.");
			}
		});

		return cost;
	}
}
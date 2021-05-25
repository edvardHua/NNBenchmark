### NNBenchmark

简单测试 TNN 和 NCNN 运行 MobileNet V2 模型（CPU，GPU，INT8）

***

其中，NCNN 转换 Mobilenet V2 是来自于 https://github.com/shicai/MobileNet-Caffe

TNN 转换 Mobilenet V2 是来自于 https://github.com/Tencent/TNN/blob/master/model/download_model.sh

TFLite 的 Mobilenet V2 则是来自于 https://www.tensorflow.org/lite/guide/hosted_models

***

```bash
环境如下
测试设备：魅族 16th
Flyme 版本：8.20.6.30
安卓版本：8.1.0

ndk: 21
sdk: 30, (min > 24)
platform: MacOS 10.15.6
abiFilters: arm64-v8a
TNN: bf615508bee2b815e9243ae07f4b9f220682d29e
NCNN: 0524
TFLite: 2.4
```


// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "Classifier.h"
#include "tnn/utils/dims_vector_utils.h"
#include <sys/time.h>
#include <cmath>

namespace TNN_NS {
    Classifier::~Classifier() {}

    int Classifier::infer(std::shared_ptr<TNN_NS::Mat> image) {
        if (!image || !image->GetData()) {
            std::cout << "image is empty ,please check!" << std::endl;
            return -1;
        }

        // step 1. set input mat, bgr input

        TNN_NS::MatConvertParam input_cvt_param;
        input_cvt_param.scale = {1.0 / (127.5), 1.0 / (127.5), 1.0 / (127.5), 0.0};
        input_cvt_param.bias = {-1, -1, -1, 0.0};
        input_cvt_param.reverse_channel = true;

        TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
        TNN_NS::DimsVector resize_dim = {1, 4, input_height, input_width};
        auto resize_mat = std::make_shared<Mat>(dt, TNN_NS::N8UC4, resize_dim);
        TNN_NS::ResizeParam param;
        param.type = INTERP_TYPE_NEAREST;
        auto resize_status = TNN_NS::MatUtils::Resize(*image, *resize_mat, param, NULL);
        RETURN_ON_NEQ(resize_status, TNN_NS::TNN_OK);

        auto status = instance_->SetInputMat(resize_mat, input_cvt_param, "input");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

        status = instance_->ForwardAsync(nullptr);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

        std::shared_ptr<TNN_NS::Mat> fg = nullptr;

        status = instance_->GetOutputMat(fg, TNN_NS::MatConvertParam(), "prob");
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
        return 0;
    }
}


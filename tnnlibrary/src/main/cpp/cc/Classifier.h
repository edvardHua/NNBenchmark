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

#ifndef ImageClassifier_hpp
#define ImageClassifier_hpp

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "tnn_sdk.h"


namespace TNN_NS {
    class Classifier : public TNN_NS::TNNSDKSample {
    public:
        const int input_width;
        const int input_height;

        Classifier() : input_width(224), input_height(224) {}

        ~Classifier();

        int infer(std::shared_ptr<TNN_NS::Mat> image);

    private:
        std::shared_ptr<Mat> GenerateAlphaImage(std::shared_ptr<Mat> alpha);

    };
}


#endif /* ImageClassifier_hpp */

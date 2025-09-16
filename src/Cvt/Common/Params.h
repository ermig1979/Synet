/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include "Synet/Params.h"

namespace Synet
{
    struct OnnxParam
    {
        CPL_PARAM_VALUE(Strings, toNchwHints, Strings());
        CPL_PARAM_VALUE(Strings, toNhwcHints, Strings());
        CPL_PARAM_VALUE(Strings, shapeV2s, Strings());
        CPL_PARAM_VALUE(bool, transpose0312PermuteToNhwc, false);
        CPL_PARAM_VALUE(bool, globalPoolingPermuteToNchw, true);
        CPL_PARAM_VALUE(bool, addToEltwise, true);
        CPL_PARAM_VALUE(bool, mulToEltwise, true);
        CPL_PARAM_VALUE(bool, setReshapeAxis1, false);
    };

    struct Bf16OptParam
    {
        CPL_PARAM_VALUE(bool, enable, false);
        CPL_PARAM_VALUE(uint32_t, minSrcC, 32);
        CPL_PARAM_VALUE(uint32_t, minDstC, 8);
        CPL_PARAM_VALUE(LowPrecisionType, addType, LowPrecisionTypeActive);
        CPL_PARAM_VALUE(LowPrecisionType, reluType, LowPrecisionTypePassive);
        CPL_PARAM_VALUE(LowPrecisionType, depthwiseType, LowPrecisionTypeNone);
        CPL_PARAM_VALUE(Strings, exclude, Strings());
        CPL_PARAM_VALUE(Strings, manualActive, Strings());
    };

    struct OptimizerParam
    {
        CPL_PARAM_VALUE(bool, mergeTwoConvolutions, true);
        CPL_PARAM_VALUE(uint32_t, mergeTwoConvolutionsOutputNumMax, 256);
        CPL_PARAM_VALUE(bool, mergeQuantizedConvolutions, false);
        CPL_PARAM_VALUE(bool, mergeInt8Convolutions, true);
        CPL_PARAM_VALUE(bool, saveUnoptimized, false);
        CPL_PARAM_VALUE(int, convToNhwc, 0);
        CPL_PARAM_VALUE(bool, skipPermute, false);
        CPL_PARAM_VALUE(bool, reuseLayers, true);
        CPL_PARAM_VALUE(bool, reuseEltwise, false);
        CPL_PARAM_STRUCT(Bf16OptParam, bf16);
    };

    CPL_PARAM_HOLDER(OptimizerParamHolder, OptimizerParam, optimizer);
}
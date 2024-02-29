/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "TestCommon.h"

#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"
#include "Synet/Converters/OnnxRuntime.h"
#include "Synet/Decoders/Anchor.h"
#include "Synet/Decoders/Ultraface.h"
#include "Synet/Decoders/YoloV5.h"
#include "Synet/Decoders/YoloV7.h"
#include "Synet/Decoders/YoloV8.h"
#include "Synet/Decoders/Iim.h"

namespace Test
{
    using Synet::Shp;

    typedef Synet::Shape Shape;
    typedef Synet::Shapes Shapes;
    typedef Synet::Floats Floats;

    struct SizeParam
    {
        CPL_PARAM_VALUE(int32_t, size, 0);
    };

    struct InputParam
    {
        CPL_PARAM_VALUE(String, name, String());
        CPL_PARAM_VALUE(Shape, dims, Shape());
        CPL_PARAM_VECTOR(SizeParam, shape);
        CPL_PARAM_VALUE(String, from, String());
        CPL_PARAM_VALUE(Floats, data, Floats());
    };

    struct OutputParam
    {
        CPL_PARAM_VALUE(String, name, String());
        CPL_PARAM_VALUE(String, compare, "");
    };

    struct DetectionParam
    {
        CPL_PARAM_VALUE(float, confidence, 0.5f);
        CPL_PARAM_VALUE(float, overlap, 0.5f);
        CPL_PARAM_VALUE(String, decoder, String());
        CPL_PARAM_STRUCT_MOD(Synet::AnchorParam, epsilon, Synet::GetEpsilonParam());
        CPL_PARAM_STRUCT_MOD(Synet::AnchorParam, retina, Synet::GetRetinaParam());
        CPL_PARAM_STRUCT(Synet::UltrafaceParam, ultraface);
        CPL_PARAM_STRUCT(Synet::YoloV5Param, yoloV5);
        CPL_PARAM_STRUCT(Synet::IimParam, iim);
    };

    struct IdParam
    {
        CPL_PARAM_VALUE(String, name, "");
        CPL_PARAM_VALUE(int, id, 0);
    };

    struct IndexParam
    {
        CPL_PARAM_VALUE(String, type, "");
        CPL_PARAM_VALUE(String, name, "index.txt");
        CPL_PARAM_VECTOR(IdParam, ids);
    };

    struct TestParam
    {
        CPL_PARAM_VALUE(String, inputType, "images");
        CPL_PARAM_VALUE(String, images, String());
        CPL_PARAM_VALUE(bool, smartResize, false);
        CPL_PARAM_VALUE(Floats, lower, Floats(1, 0.0f));
        CPL_PARAM_VALUE(Floats, upper, Floats(1, 1.0f));
        CPL_PARAM_VALUE(String, model, String());
        CPL_PARAM_VALUE(String, order, String());
        CPL_PARAM_VALUE(bool, dynamicOutput, false);
        CPL_PARAM_VECTOR(InputParam, input);
        CPL_PARAM_VECTOR(OutputParam, output);
        CPL_PARAM_STRUCT(DetectionParam, detection);
        CPL_PARAM_STRUCT(IndexParam, index);
        CPL_PARAM_STRUCT(Synet::OptimizerParam, optimizer);
        CPL_PARAM_STRUCT(Synet::OnnxParam, onnx);
    };

    CPL_PARAM_HOLDER(TestParamHolder, TestParam, test);
}


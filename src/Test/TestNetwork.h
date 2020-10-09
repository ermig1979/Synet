/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

namespace Test
{
    using Synet::Shp;

    typedef Synet::Shape Shape;
    typedef Synet::Shapes Shapes;
    typedef Synet::Floats Floats;

    struct SizeParam
    {
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VALUE(int32_t, size, 0);
    };

    struct ShapeParam
    {
        SYNET_PARAM_VALUE(String, name, String());
        SYNET_PARAM_VECTOR(SizeParam, shape);
        SYNET_PARAM_VALUE(int32_t, size, 0);
    };

    struct EpsilonParam
    {
        SYNET_PARAM_VALUE(bool, enable, false);
        SYNET_PARAM_VALUE(Floats, variance, Floats({ 0.1f, 0.2f }));
        SYNET_PARAM_VALUE(Shape, step, Shape({ 8, 16, 32, 64 }));
        SYNET_PARAM_VALUE(Shape, minSize, Shape({ 10, 16, 24, 32, 48, 0, 64, 96, 0, 128, 192, 256 }));
        SYNET_PARAM_VALUE(bool, clip, false);
    };

    struct TestParam
    {
        SYNET_PARAM_VALUE(String, images, String());
        SYNET_PARAM_VALUE(Floats, lower, Floats(1, 0.0f));
        SYNET_PARAM_VALUE(Floats, upper, Floats(1, 1.0f));
        SYNET_PARAM_VECTOR(ShapeParam, input);
        SYNET_PARAM_VECTOR(ShapeParam, output);
        SYNET_PARAM_STRUCT(EpsilonParam, epsilon);
    };

    SYNET_PARAM_HOLDER(TestParamHolder, TestParam, test);

    //-------------------------------------------------------------------------

    typedef Synet::Region<float> Region;
    typedef std::vector<Region> Regions;
    typedef Synet::Floats Floats;
    typedef Synet::Tensor<float> Tensor;
    typedef std::vector<Tensor> Tensors;

    struct Network
    {
        struct Options
        {
            String outputDirectory;
            size_t workThreads;
            bool consoleSilence;
            int batchSize;
            int debugPrint;
            float regionThreshold;
            Options(String od, size_t wt, bool cs, int bs, int dp, float rt)
                : outputDirectory(od)
                , workThreads(wt)
                , consoleSilence(cs)
                , batchSize(bs)
                , debugPrint(dp)
                , regionThreshold(rt)
            {}
        };

        Network() {}
        virtual ~Network() {}
        virtual String Name() const { return String(); }
        virtual String Type() const { return String(); }
        virtual size_t SrcCount() const { return 0; }
        virtual Shape SrcShape(size_t index) const { return Shape(); }
        virtual size_t SrcSize(size_t index) const { return 0; }
        virtual bool Init(const String & model, const String & weight, const Options & options, const TestParam & param) { return false; }
        virtual void Free() { _output.clear(); }
        virtual const Tensors& Predict(const Tensors& src) { return _output; }
        virtual void DebugPrint(const Tensors& src, std::ostream & os, int flag, int first, int last, int precision) { }
        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const { return Regions(); }
        virtual size_t MemoryUsage() const { return 0; }

    protected:
        Tensors _output;
        float _regionThreshold;
    };
    typedef std::shared_ptr<Network> NetworkPtr;
    typedef std::vector<NetworkPtr> NetworkPtrs;
}


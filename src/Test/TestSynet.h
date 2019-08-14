/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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

#include "Test/TestCommon.h"
#include "Test/TestPerformance.h"

#define SYNET_PERF_FUNC() TEST_PERF_FUNC()
#define SYNET_PERF_BLOCK(name) TEST_PERF_BLOCK(name)
#define SYNET_PERF_BLOCK_END(name) TEST_PERF_BLOCK_END(name)
#include "Synet/Synet.h"

namespace Test
{
    typedef Synet::Shape Shape;
    typedef Synet::Shapes Shapes;

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

    struct TestParam
    {
        SYNET_PARAM_VALUE(String, origin, String());
        SYNET_PARAM_VALUE(float, lower, 0.0f);
        SYNET_PARAM_VALUE(float, upper, 1.0f);
        SYNET_PARAM_VECTOR(ShapeParam, input);
        SYNET_PARAM_VECTOR(ShapeParam, output);
    };

    SYNET_PARAM_HOLDER(TestParamHolder, TestParam, test);
}

namespace Test
{
    typedef Synet::Region<float> Region;
    typedef std::vector<Region> Regions;

    struct Network
    {
        Network() {}
        virtual ~Network() {}
        virtual String Name() const = 0;
        virtual size_t SrcCount() const = 0;
        virtual Shape SrcShape(size_t index) const = 0;
        virtual size_t SrcSize(size_t index) const = 0;
        virtual bool Init(const String & model, const String & weight, size_t threadNumber, size_t batchSize, const TestParam & param) = 0;
        virtual const Vectors & Predict(const Vectors & src) = 0;
#ifdef SYNET_DEBUG_PRINT_ENABLE
        virtual void DebugPrint(std::ostream & os) = 0;
#endif
        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const { return Regions(); }
        virtual size_t MemoryUsage() const { return 0; }
    protected:
        Vectors _output;
    };
    typedef std::shared_ptr<Network> NetworkPtr;
}

namespace Test
{
    struct SynetNetwork : public Network
    {
        virtual String Name() const
        {
            return "Synet";
        }

        virtual size_t SrcCount() const
        {
            return _net.Src().size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _net.Src()[index]->Shape();
            Synet::TensorFormat format = _net.Src()[index]->Format();
            return shape.size() == 4 && format == Synet::TensorFormatNhwc ? Shape({shape[0], shape[3], shape[1], shape[2]}) : shape;
        }

        virtual size_t SrcSize(size_t index) const
        {
            return _net.Src()[index]->Size();
        }

        virtual bool Init(const String & model, const String & weight, size_t threadNumber, size_t batchSize, const TestParam & param)
        {
            TEST_PERF_FUNC();
            Synet::SetThreadNumber(threadNumber);
            if (_net.Load(model, weight))
            {
                _trans = _net.Format() == Synet::TensorFormatNhwc;
                if (param.input().size() || param.output().size())
                {
                    if (!Reshape(param, batchSize))
                        return false;
                }
                else if (_net.Src().size() == 1)
                {
                    const Shape & shape = _net.NchwShape();
                    if (shape.size() == 4 && shape[0] != batchSize)
                    {
                        if (!_net.Reshape(shape[3], shape[2], batchSize))
                            return false;
                    }
                }
                _net.CompactWeight();
                return true;
            }
            return false;
        }

        virtual const Vectors & Predict(const Vectors & x)
        {
            SetInput(x);
            {
                TEST_PERF_FUNC();
                _net.Forward();
            }
            SetOutput();
            return _output;
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        virtual void DebugPrint(std::ostream & os)
        {
            _net.DebugPrint(os, true);
        };
#endif

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            return _net.GetRegions(size.x, size.y, threshold, overlap);
        }

        virtual size_t MemoryUsage() const
        {
            return _net.MemoryUsage();
        }

    private:
        typedef Synet::Network<float> Net;
        Net _net;
        bool _trans;

        void SetInput(const Vectors & x)
        {
            assert(x.size() == _net.Src().size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                Net::Tensor & src = *_net.Src()[i];
                assert(x[i].size() == src.Size());
                if (src.Format() == Synet::TensorFormatNhwc && src.Count() == 4)
                {
                    const float * pX = x[i].data();
                    for (size_t n = 0; n < src.Axis(0); ++n)
                        for (size_t c = 0; c < src.Axis(3); ++c)
                            for (size_t y = 0; y < src.Axis(1); ++y)
                                for (size_t x = 0; x < src.Axis(2); ++x)
                                    src.CpuData(Shape({ n, y, x, c }))[0] = *pX++;
                }
                else
                    memcpy(src.CpuData(), x[i].data(), x[i].size() * sizeof(float));
            }
        }

        void SetOutput()
        {
            Net::TensorPtrs dst = _net.Dst();
            _output.resize(dst.size());
            for (size_t i = 0; i < dst.size(); ++i)
            {
                _output[i].clear();
                if (dst[i]->Count() == 4 && dst[i]->Axis(3) == 7 && _net.Back()[i]->Param().type() == Synet::LayerTypeDetectionOutput)
                {
                    const float * pDst = dst[i]->CpuData();
                    for (size_t j = 0; j < dst[i]->Axis(2); ++j, pDst += 7)
                    {
                        if (pDst[1] == -1)
                            break;
                        size_t offset = _output[i].size();
                        _output[i].resize(offset + 7);
                        _output[i][offset + 0] = pDst[0];
                        _output[i][offset + 1] = pDst[1];
                        _output[i][offset + 2] = pDst[2];
                        _output[i][offset + 3] = pDst[3];
                        _output[i][offset + 4] = pDst[4];
                        _output[i][offset + 5] = pDst[5];
                        _output[i][offset + 6] = pDst[6];
                    }
                    SortDetectionOutput(_output[i].data(), _output[i].size());
                }
                else
                {
                    bool trans = dst[i]->Format() == Synet::TensorFormatNhwc;
                    _output[i].resize(dst[i]->Size());
                    if (trans && dst[i]->Count() == 4)
                    {
                        float * out = _output[i].data();
                        for (size_t n = 0; n < dst[i]->Axis(0); ++n)
                            for (size_t c = 0; c < dst[i]->Axis(3); ++c)
                                for (size_t y = 0; y < dst[i]->Axis(1); ++y)
                                    for (size_t x = 0; x < dst[i]->Axis(2); ++x)
                                        *out++ = dst[i]->CpuData(Shape({ n, y, x, c }))[0];
                    }
                    else if (trans && dst[i]->Count() == 3)
                    {
                        float * out = _output[i].data();
                        for (size_t c = 0; c < dst[i]->Axis(2); ++c)
                            for (size_t y = 0; y < dst[i]->Axis(0); ++y)
                                for (size_t x = 0; x < dst[i]->Axis(1); ++x)
                                    *out++ = dst[i]->CpuData(Shape({ y, x, c }))[0];
                    }
                    else if (trans && dst[i]->Count() == 2 && dst[i]->Axis(0) == 1)
                    {
                        float * out = _output[i].data();
                        for (size_t c = 0; c < dst[i]->Axis(1); ++c)
                            for (size_t s = 0; s < dst[i]->Axis(0); ++s)
                                *out++ = dst[i]->CpuData(Shape({ s, c }))[0];
                    }
                    else
                        memcpy(_output[i].data(), dst[i]->CpuData(), dst[i]->Size() * sizeof(float));
                }
            }
        }

        bool Reshape(const TestParam & param, size_t batchSize)
        {
            Strings srcNames;
            Shapes srcShapes;
            for (size_t i = 0; i < param.input().size(); ++i)
            {
                const ShapeParam & shape = param.input()[i];
                srcNames.push_back(shape.name());
                Shape srcShape;
                if (shape.size() > 0)
                    srcShape.push_back(shape.size());
                else
                {
                    for (size_t j = 0; j < shape.shape().size(); ++j)
                    {
                        const SizeParam & size = shape.shape()[j];
                        if (size.size() > 0)
                            srcShape.push_back(size.size());
                        else if (size.name().size() > 0)
                        {
                            Net::Tensor tensor;
                            if (_net.GetMetaConst(size.name(), tensor) && tensor.GetType() == Synet::TensorType32i && tensor.Size())
                            {
                                int32_t value = tensor.As32i().CpuData()[0];
                                if (value > 0)
                                    srcShape.push_back(value);
                                else
                                    return false;
                            }
                            else
                                return false;
                        }
                        else
                            return false;
                    }
                }
                if (srcShape.size() == 4)   
                {
                    srcShape[0] = batchSize;
                    if (_trans)
                        srcShape = Shape({ srcShape[0], srcShape[2], srcShape[3], srcShape[1] });
                }
                if (srcShape.empty())
                    return false;
                srcShapes.push_back(srcShape);
            }

            Strings dstNames;
            for (size_t i = 0; i < param.output().size(); ++i)
                dstNames.push_back(param.output()[i].name());

            bool equal = false;
            if (srcShapes.size() == _net.Src().size() && _net.Back().size() == dstNames.size())
            {
                equal = true;
                for (size_t i = 0; i < srcShapes.size() && equal; ++i)
                    if (srcShapes[i] != _net.Src()[i]->Shape())
                        equal = false;
                for (size_t i = 0; i < dstNames.size() && equal; ++i)
                    if (dstNames[i] != _net.Back()[i]->Param().name())
                        equal = false;
            }

            return equal || _net.Reshape(srcNames, srcShapes, dstNames);
        }
    };

}


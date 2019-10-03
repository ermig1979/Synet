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

#ifdef SYNET_LAYER_STATISTIC
#define SYNET_PERF_FUNC() TEST_PERF_FUNC()
#define SYNET_PERF_BLOCK(name) TEST_PERF_BLOCK(name)
#define SYNET_PERF_BLOCK_END(name) TEST_PERF_BLOCK_END(name)
#endif //SYNET_LAYER_STATISTIC
#include "Synet/Synet.h"

namespace Test
{
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

    struct TestParam
    {
        SYNET_PARAM_VALUE(String, origin, String());
        SYNET_PARAM_VALUE(Floats, lower, Floats(1, 0.0f));
        SYNET_PARAM_VALUE(Floats, upper, Floats(1, 1.0f));
        SYNET_PARAM_VECTOR(ShapeParam, input);
        SYNET_PARAM_VECTOR(ShapeParam, output);
    };

    SYNET_PARAM_HOLDER(TestParamHolder, TestParam, test);
}

namespace Test
{
    typedef Synet::Region<float> Region;
    typedef std::vector<Region> Regions;
    typedef Synet::Floats Floats;

    struct Network
    {
        Network() {}
        virtual ~Network() {}
        virtual String Name() const { return String(); }
        virtual size_t SrcCount() const { return 0; }
        virtual Shape SrcShape(size_t index) const { return Shape(); }
        virtual size_t SrcSize(size_t index) const { return 0; }
        virtual bool Init(const String & model, const String & weight, const Options & options, const TestParam & param) { return false; }
        virtual const Vectors & Predict(const Vectors & src) { return _output; }
        virtual void DebugPrint(std::ostream & os, int flag, int first, int last) { }
        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const { return Regions(); }
        virtual size_t MemoryUsage() const { return 0; }
    protected:
        Vectors _output;
        float _regionThreshold;
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

        virtual bool Init(const String & model, const String & weight, const Options & options, const TestParam & param)
        {
            TEST_PERF_FUNC();
            _regionThreshold = options.regionThreshold;
            Synet::SetThreadNumber(options.workThreads);
            if (Load(model, weight))
            {
                _trans = _net.Format() == Synet::TensorFormatNhwc;
                if (param.input().size() || param.output().size())
                {
                    if (!Reshape(param, options.batchSize))
                        return false;
                }
                else if (_net.Src().size() == 1)
                {
                    const Shape & shape = _net.NchwShape();
                    if (shape.size() == 4 && shape[0] != options.batchSize)
                    {
                        if (!_net.Reshape(shape[3], shape[2], options.batchSize))
                            return false;
                    }
                }
                _net.CompactWeight();
                _lower = param.lower();
                _upper = param.upper();
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

        virtual void DebugPrint(std::ostream & os, int flag, int first, int last)
        {
            if (flag)
                _net.DebugPrint(os, (flag&DEBUG_PRINT_WEIGHT) != 0, (flag&DEBUG_PRINT_INTERIM) != 0, first, last);
        };

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
        Floats _lower, _upper;

        bool Load(const String & model, const String & weight)
        {
#ifdef SYNET_TEST_MEMORY_LOAD
            std::ifstream mifs(model, std::ios::binary);
            if (!mifs)
                return false;
            mifs.unsetf(std::ios::skipws);
            mifs.seekg(0, std::ios::end);
            size_t msize = mifs.tellg();
            mifs.seekg(0);
            std::vector<char> mdata(msize + 1, 0);
            mifs.read(mdata.data(), (std::streamsize)msize);
            mifs.close();

            std::ifstream wifs(weight, std::ios::binary);
            if (!wifs)
                return false;
            wifs.unsetf(std::ios::skipws);
            wifs.seekg(0, std::ios::end);
            size_t wsize = wifs.tellg();
            wifs.seekg(0);
            std::vector<char> wdata(wsize + 1, 0);
            wifs.read(wdata.data(), (std::streamsize)wsize);
            wifs.close();

            return _net.Load(mdata.data(), msize, wdata.data(), wsize);
#else
            return _net.Load(model, weight);
#endif
        }

        void SetInput(const Vectors & x)
        {
            assert(x.size() == _net.Src().size());
#ifdef SYNET_TEST_SET_INPUT
            if (_net.Src().size() == 1 && _net.Src()[0]->Count() == 4)
            {
                Views views;
                InputToViews(x[0], views);
                _net.SetInput(views, _lower, _upper);
                return;
            }
#endif
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

#ifdef SYNET_TEST_SET_INPUT
        void InputToViews(const Vector & src, Views & dst)
        {
            Shape shape = _net.NchwShape();
            assert(shape[1] == 1 || shape[1] == 3);
            if (_lower.size() == 1)
                _lower.resize(shape[1], _lower[0]);
            if (_upper.size() == 1)
                _upper.resize(shape[1], _upper[0]);
            dst.resize(shape[0]);
            for (size_t b = 0; b < shape[0]; ++b)
            {
                dst[b].Recreate(Size(shape[3], shape[2]), shape[1] == 1 ? View::Gray8 : View::Bgr24);
                for (size_t c = 0; c < shape[1]; ++c)
                    for (size_t y = 0; y < shape[2]; ++y)
                        for (size_t x = 0; x < shape[3]; ++x)
                            dst[b].data[dst[b].stride*y + x * shape[1] + c] = Float32ToUint8(src[((b*shape[1] + c)*shape[2] + y)*shape[3] + x], _lower[c], _upper[c]);
            }
        }

        uint8_t Float32ToUint8(const float value, float lower, float upper)
        {
            return (uint8_t)Synet::Round((std::min(std::max(value, lower), upper) - lower) * 255.0f / (upper - lower));
        }
#endif

        void SetOutput()
        {
            typedef std::map<String, Net::Tensor*> Dst;
            Dst dst;
            for (size_t i = 0; i < _net.Dst().size(); ++i)
                dst[_net.Dst()[i]->Name()] = _net.Dst()[i];
            _output.resize(dst.size());
            size_t i = 0;
            for (Dst::const_iterator it = dst.begin(); it != dst.end(); ++it, ++i)
                SetOutput(*it->second, *_net.Back()[i], _output[i]);
        }

        void SetOutput(const Net::Tensor & src, const Net::Layer & back, Vector & dst)
        {
            dst.clear();
            if (src.Count() == 4 && src.Axis(3) == 7 && back.Param().type() == Synet::LayerTypeDetectionOutput)
            {
                const float * pSrc = src.CpuData();
                for (size_t j = 0; j < src.Axis(2); ++j, pSrc += 7)
                {
                    if (pSrc[1] == -1 || pSrc[2] < _regionThreshold)
                        break;
                    size_t offset = dst.size();
                    dst.resize(offset + 7);
                    dst[offset + 0] = pSrc[0];
                    dst[offset + 1] = pSrc[1];
                    dst[offset + 2] = pSrc[2];
                    dst[offset + 3] = pSrc[3];
                    dst[offset + 4] = pSrc[4];
                    dst[offset + 5] = pSrc[5];
                    dst[offset + 6] = pSrc[6];
                }
                SortDetectionOutput(dst.data(), dst.size());
            }
            else
            {
                bool trans = src.Format() == Synet::TensorFormatNhwc;
                dst.resize(src.Size());
                if (trans && src.Count() == 4)
                {
                    float * pDst = dst.data();
                    for (size_t n = 0; n < src.Axis(0); ++n)
                        for (size_t c = 0; c < src.Axis(3); ++c)
                            for (size_t y = 0; y < src.Axis(1); ++y)
                                for (size_t x = 0; x < src.Axis(2); ++x)
                                    *pDst++ = src.CpuData(Shape({ n, y, x, c }))[0];
                }
                else if (trans && src.Count() == 3)
                {
                    float * pDst = dst.data();
                    for (size_t c = 0; c < src.Axis(2); ++c)
                        for (size_t y = 0; y < src.Axis(0); ++y)
                            for (size_t x = 0; x < src.Axis(1); ++x)
                                *pDst++ = src.CpuData(Shape({ y, x, c }))[0];
                }
                else if (trans && src.Count() == 2 && src.Axis(0) == 1)
                {
                    float * pDst = dst.data();
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t s = 0; s < src.Axis(0); ++s)
                            *pDst++ = src.CpuData(Shape({ s, c }))[0];
                }
                else
                    memcpy(dst.data(), src.CpuData(), src.Size() * sizeof(float));
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
#ifdef SYNET_TEST_NET_RESHAPE
            if (_trans)
                return _net.Reshape(srcShapes[0][2], srcShapes[0][1], srcShapes[0][0]);
            else
                return _net.Reshape(srcShapes[0][3], srcShapes[0][2], srcShapes[0][0]);
#endif
            return equal || _net.Reshape(srcNames, srcShapes, dstNames);
        }
    };

}


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
#include "TestPerformance.h"

#if defined(SYNET_LAYER_STATISTIC) && !defined(SYNET_PERF_FUNC)
#define SYNET_PERF_FUNC() TEST_PERF_FUNC()
#define SYNET_PERF_BLOCK(name) TEST_PERF_BLOCK(name)
#define SYNET_PERF_BLOCK_END(name) TEST_PERF_BLOCK_END(name)
#define SYNET_PERF_DECL(name) Test::PerformanceMeasurer * name;
#define SYNET_PERF_SET(name, value) name = value;
#define SYNET_PERF_INIT(name, desc, flop) name = Test::PerformanceMeasurerStorage::s_storage.Get(desc, flop);
#define SYNET_PERF_TEST(name) Test::PerformanceMeasurerHolder SYNET_CAT(__pmh,__LINE__)(name);
#endif
#include "Synet/Synet.h"

#include "TestNetwork.h"

namespace Test
{
    struct SynetNetwork : public Network
    {
        virtual String Name() const
        {
            return "Synet";
        }

        virtual String Type() const
        {
            return _net.Is8i() ? "int8" : "fp32";
        }

        virtual size_t SrcCount() const
        {
            return _net.Src().size();
        }

        virtual Shape SrcShape(size_t index) const
        {
            Shape shape = _net.Src()[index]->Shape();
            Synet::TensorFormat format = _net.Src()[index]->Format();
            return shape.size() == 4 && format == Synet::TensorFormatNhwc ? Shp(shape[0], shape[3], shape[1], shape[2]) : shape;
        }

        virtual size_t SrcSize(size_t index) const
        {
            return _net.Src()[index]->Size();
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
            TEST_PERF_BLOCK(Type());
            _regionThreshold = options.regionThreshold;
            Synet::SetThreadNumber(options.workThreads);
            if (Load(model, weight))
            {
                _trans = _net.Format() == Synet::TensorFormatNhwc;
                _sort = param.output().empty();
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
                _synetMemoryUsage = _net.MemoryUsage();
                return true;
            }
            return false;
        }

        virtual void Free()
        {
            Network::Free();
            _net.Clear();
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            SetInput(src);
            {
                TEST_PERF_BLOCK_FLOP(Type(), _net.Flop());
                _net.Forward();
            }
            SetOutput();
            return _output;
        }

        virtual void DebugPrint(const Tensors& src, std::ostream & os, int flag, int first, int last, int precision)
        {
            if (flag)
            {
                SetInput(src);
                _net.DebugPrint(os, flag, first, last, precision);
            }
        };

        virtual Regions GetRegions(const Size & size, float threshold, float overlap) const
        {
            return _net.GetRegions(size.x, size.y, threshold, overlap);
        }

        virtual size_t MemoryUsage() const
        {
            return _synetMemoryUsage;
        }

    private:
        typedef Synet::Network<float> Net;
        Net _net;
        bool _trans, _sort;
        Floats _lower, _upper;
        size_t _synetMemoryUsage;

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

        void SetInput(const Tensors& x)
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
                assert(x[i].Size() == src.Size());
                if (src.Format() == Synet::TensorFormatNhwc && src.Count() == 4)
                {
                    const float * pX = x[i].CpuData();
                    for (size_t n = 0; n < src.Axis(0); ++n)
                        for (size_t c = 0; c < src.Axis(3); ++c)
                            for (size_t y = 0; y < src.Axis(1); ++y)
                                for (size_t x = 0; x < src.Axis(2); ++x)
                                    src.CpuData(Shape({ n, y, x, c }))[0] = *pX++;
                }
                else
                    memcpy(src.CpuData(), x[i].CpuData(), x[i].Size() * sizeof(float));
            }
        }

#ifdef SYNET_TEST_SET_INPUT
        void InputToViews(const Tensor & src, Views & dst)
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
                dst[b].Recreate(Size(shape[3], shape[2]), shape[1] == 1 ? View::Gray8 : View::Bgra32);
                for (size_t c = 0; c < shape[1]; ++c)
                    for (size_t y = 0; y < shape[2]; ++y)
                        for (size_t x = 0; x < shape[3]; ++x)
                            dst[b].data[dst[b].stride*y + x * (shape[1] == 1 ? 1 : 4) + c] = 
                            Float32ToUint8(src.CpuData(Shp( b, c, y, x ))[0], _lower[c], _upper[c]);
            }
        }

        uint8_t Float32ToUint8(const float value, float lower, float upper)
        {
            return (uint8_t)Synet::Round((std::min(std::max(value, lower), upper) - lower) * 255.0f / (upper - lower));
        }
#endif

        void SetOutput()
        {
            if (_sort)
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
            else
            {
                _output.resize(_net.Dst().size());
                for (size_t i = 0; i < _net.Dst().size(); ++i)
                    SetOutput(*_net.Dst()[i], *_net.Back()[i], _output[i]);
            }
        }

        void SetOutput(const Net::Tensor & src, const Net::Layer & back, Tensor & dst)
        {
            if (src.Count() == 4 && src.Axis(3) == 7 && back.Param().type() == Synet::LayerTypeDetectionOutput)
            {
                assert(src.Axis(0) == 1);
                Vector tmp;
                const float * pSrc = src.CpuData();
                for (size_t j = 0; j < src.Axis(2); ++j, pSrc += 7)
                {
                    if (pSrc[1] == -1 || pSrc[2] < _regionThreshold)
                        break;
                    size_t offset = tmp.size();
                    tmp.resize(offset + 7);
                    tmp[offset + 0] = pSrc[0];
                    tmp[offset + 1] = pSrc[1];
                    tmp[offset + 2] = pSrc[2];
                    tmp[offset + 3] = pSrc[3];
                    tmp[offset + 4] = pSrc[4];
                    tmp[offset + 5] = pSrc[5];
                    tmp[offset + 6] = pSrc[6];
                }
                SortDetectionOutput(tmp.data(), tmp.size());
                dst.Reshape(Shp(1, 1, tmp.size()/7, 7));
                memcpy(dst.CpuData(), tmp.data(), dst.Size() * sizeof(float));
            }
            else
            {
                bool trans = src.Format() == Synet::TensorFormatNhwc;
                bool batch = _net.Src()[0]->Axis(0) != 1;
                if (trans && src.Count() == 4)
                {
                    dst.Reshape(Shp(src.Axis(0), src.Axis(3), src.Axis(1), src.Axis(2)), Synet::TensorFormatNchw);
                    for (size_t n = 0; n < src.Axis(0); ++n)
                        for (size_t c = 0; c < src.Axis(3); ++c)
                            for (size_t y = 0; y < src.Axis(1); ++y)
                                for (size_t x = 0; x < src.Axis(2); ++x)
                                    dst.CpuData(Shp(n, c, y, x))[0] = src.CpuData(Shp(n, y, x, c))[0];
                }
                else if (trans && src.Count() == 3)
                {
                    if (batch)
                    {
                        dst.Reshape(Shp(src.Axis(0), src.Axis(2), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t n = 0; n < src.Axis(0); ++n)
                            for (size_t c = 0; c < src.Axis(2); ++c)
                                for (size_t s = 0; s < src.Axis(1); ++s)
                                    dst.CpuData(Shp(n, c, s))[0] = src.CpuData(Shp(n, s, c))[0];
                    }
                    else
                    {
                        dst.Reshape(Shp(src.Axis(2), src.Axis(0), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t c = 0; c < src.Axis(2); ++c)
                            for (size_t y = 0; y < src.Axis(0); ++y)
                                for (size_t x = 0; x < src.Axis(1); ++x)
                                    dst.CpuData(Shp(c, y, x))[0] = src.CpuData(Shp(y, x, c))[0];
                    }
                }
                else if (trans && src.Count() == 2 && src.Axis(0) == 1)
                {
                    dst.Reshape(Shp(src.Axis(1), src.Axis(0)), Synet::TensorFormatNchw);
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t s = 0; s < src.Axis(0); ++s)
                            dst.CpuData(Shp(c, s))[0] = src.CpuData(Shp(s, c))[0];
                }
                else
                {
                    dst.Reshape(src.Shape(), Synet::TensorFormatNchw);
                    memcpy(dst.CpuData(), src.CpuData(), src.Size() * sizeof(float));
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


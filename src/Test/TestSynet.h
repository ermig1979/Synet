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
#include "TestPerformance.h"
#include "TestNetwork.h"

#include "Synet/Network.h"

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

        virtual Synet::TensorType SrcType(size_t index) const
        {
            return _net.Src()[index]->GetType();
        }

        virtual size_t SrcSize(size_t index) const
        {
            return _net.Src()[index]->Size();
        }

        virtual bool Init(const String & model, const String & weight, const Options& options, const TestParam & param)
        {
            CPL_PERF_BEG(Type());
            _regionThreshold = options.regionThreshold;
            _decoderName = param.detection().decoder();
            Synet::SetThreadNumber(options.workThreads);
            if (Load(model, weight, options))
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
                if(param.detection().decoder() == "epsilon")
                    _anchor.Init(_net, param.detection().epsilon());
                if (param.detection().decoder() == "retina")
                    _anchor.Init(_net, param.detection().retina());
                if (param.detection().decoder() == "ultraface")
                    _ultraface.Init(param.detection().ultraface());
                if (param.detection().decoder() == "yoloV5")
                    _yoloV5.Init(param.detection().yoloV5());
                if (param.detection().decoder() == "yoloV7")
                    _yoloV7.Init();
                if (param.detection().decoder() == "yoloV8")
                    _yoloV8.Init();
                if (param.detection().decoder() == "iim")
                    _iim.Init(param.detection().iim());
                if (param.detection().decoder() == "rtdetr")
                    _rtdetr.Init();
                return true;
            }
            return false;
        }

        virtual void Free()
        {
            Network::Free();
            _net.Clear();
#ifdef __linux__
            malloc_trim(0);
#endif
        }

        virtual const Tensors & Predict(const Tensors& src)
        {
            SetInput(src);
            {
                CPL_PERF_BEGF(Type(), _net.Flop());
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
            if (_anchor.Enable())
                return _anchor.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else if (_ultraface.Enable())
                return _ultraface.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else if (_iim.Enable())
                return _iim.GetRegions(_net, size.x, size.y)[0];
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(_net, size.x, size.y, threshold, overlap)[0];
            else
                return _net.GetRegions(size.x, size.y, threshold, overlap);
        }

        virtual size_t MemoryUsage() const
        {
            return _synetMemoryUsage;
        }

    private:
        typedef Synet::Network Net;
        Net _net;
        bool _trans, _sort;
        Floats _lower, _upper;
        size_t _synetMemoryUsage;
        Synet::AnchorDecoder _anchor;
        Synet::UltrafaceDecoder _ultraface;
        Synet::YoloV5Decoder _yoloV5;
        Synet::YoloV7Decoder _yoloV7;
        Synet::YoloV8Decoder _yoloV8;
        Synet::IimDecoder _iim;
        Synet::RtdetrDecoder _rtdetr;

        bool Load(const String & model, const String & weight, const Options& options)
        {
            Synet::Options synOpt;
            synOpt.performanceLog = (Synet::Options::PerfomanceLog)options.performanceLog;
            synOpt.bf16RoundTest = options.bf16Test;

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

            return _net.Load(mdata.data(), msize, wdata.data(), wsize, synOpt);
#else
            return _net.Load(model, weight, synOpt);
#endif
        }

        template<class T> static void SetInput(const Tensor & src, Tensor & dst)
        {
            assert(src.Size() == dst.Size() && src.GetType() == dst.GetType());
            if (dst.Format() == Synet::TensorFormatNhwc && dst.Count() == 4)
            {
                for (size_t n = 0; n < src.Axis(0); ++n)
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t y = 0; y < src.Axis(2); ++y)
                            for (size_t x = 0; x < src.Axis(3); ++x)
                                dst.Data<T>(Shape({ n, y, x, c }))[0] = src.Data<T>(Shape({ n, c, y, x }))[0];
            }
            else
                memcpy(dst.RawData(), src.RawData(), src.RawSize());
        }

        void SetInput(const Tensors& src)
        {
            assert(src.size() == _net.Src().size());
#ifdef SYNET_TEST_SET_INPUT
            if (_net.Src().size() == 1 && _net.Src()[0]->Count() == 4 && _netSrc().GetType() == Synet::TensorType32f)
            {
                Views views;
                InputToViews(x[0], views);
                _net.SetInput(views, _lower, _upper);
                return;
            }
#endif
            for (size_t i = 0; i < src.size(); ++i)
            {
                switch (src[i].GetType())
                {
                case Synet::TensorType32f: SetInput<float>(src[i], *_net.Src()[i]); break;
                case Synet::TensorType32i: SetInput<int32_t>(src[i], *_net.Src()[i]); break;
                case Synet::TensorType64i: SetInput<int64_t>(src[i], *_net.Src()[i]); break;
                default:
                    assert(0);
                }
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
                            Float32ToUint8(src.Data<float>(Shp( b, c, y, x ))[0], _lower[c], _upper[c]);
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

        void SetOutput(const Net::Tensor& src, const Net::Layer& back, Tensor& dst)
        {
            switch (src.GetType())
            {
            case Synet::TensorType32f:
                SetOutputT(src.As32f(), back, dst);
                break;
            case Synet::TensorType32i:
                SetOutputT(src.As32i(), back, dst);
                break;
            case Synet::TensorType64i:
                SetOutputT(src.As64i(), back, dst);
                break;
            default:
                assert(0);
            }
        }

        template<class T> void SetOutputT(const Synet::Tensor<T> & src, const Net::Layer & back, Tensor & dst)
        {
            if (src.Count() == 4 && src.Axis(3) == 7 && back.Param().type() == Synet::LayerTypeDetectionOutput)
            {
                assert(src.Axis(0) == 1);
                Vector tmp;
                const T * pSrc = src.CpuData();
                for (size_t j = 0; j < src.Axis(2); ++j, pSrc += 7)
                {
                    if (pSrc[0] == -1)
                        break;
                    if (pSrc[2] <= _regionThreshold)
                        continue;
                    size_t offset = tmp.size();
                    tmp.resize(offset + 7);
                    tmp[offset + 0] = (float)pSrc[0];
                    tmp[offset + 1] = (float)pSrc[1];
                    tmp[offset + 2] = (float)pSrc[2];
                    tmp[offset + 3] = (float)pSrc[3];
                    tmp[offset + 4] = (float)pSrc[4];
                    tmp[offset + 5] = (float)pSrc[5];
                    tmp[offset + 6] = (float)pSrc[6];
                }
                SortDetectionOutput(tmp.data(), tmp.size());
                dst.Reshape(Shp(1, 1, tmp.size()/7, 7));
                memcpy(dst.CpuData(), tmp.data(), dst.Size() * sizeof(float));
            }
            else if (_decoderName == "rtdetr")
            {
                assert(src.Axis(0) == 1 && src.Axis(2) == 6);
                Vector tmp;
                const T* pSrc = src.CpuData();
                for (size_t i = 0, n = src.Axis(1); i < n; ++i, pSrc += 6)
                {
                    if (pSrc[4] <= _regionThreshold)
                        continue;
                    size_t offset = tmp.size();
                    tmp.resize(offset + 6);
                    tmp[offset + 0] = (float)pSrc[0];
                    tmp[offset + 1] = (float)pSrc[1];
                    tmp[offset + 2] = (float)pSrc[2];
                    tmp[offset + 3] = (float)pSrc[3];
                    tmp[offset + 4] = (float)pSrc[4];
                    tmp[offset + 5] = (float)pSrc[5];
                }
                SortRtdetr(tmp.data(), tmp.size());
                dst.Reshape(Shp(1, tmp.size() / 6, 6));
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
                                    dst.CpuData(Shp(n, c, y, x))[0] = (float)src.CpuData(Shp(n, y, x, c))[0];
                }
                else if (trans && src.Count() == 3)
                {
                    if (batch)
                    {
                        dst.Reshape(Shp(src.Axis(0), src.Axis(2), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t n = 0; n < src.Axis(0); ++n)
                            for (size_t c = 0; c < src.Axis(2); ++c)
                                for (size_t s = 0; s < src.Axis(1); ++s)
                                    dst.CpuData(Shp(n, c, s))[0] = (float)src.CpuData(Shp(n, s, c))[0];
                    }
                    else
                    {
                        dst.Reshape(Shp(src.Axis(2), src.Axis(0), src.Axis(1)), Synet::TensorFormatNchw);
                        for (size_t c = 0; c < src.Axis(2); ++c)
                            for (size_t y = 0; y < src.Axis(0); ++y)
                                for (size_t x = 0; x < src.Axis(1); ++x)
                                    dst.CpuData(Shp(c, y, x))[0] = (float)src.CpuData(Shp(y, x, c))[0];
                    }
                }
                else if (trans && src.Count() == 2 && (src.Axis(0) == 1 || src.Format() == Synet::TensorFormatNhwc))
                {
                    dst.Reshape(Shp(src.Axis(1), src.Axis(0)), Synet::TensorFormatNchw);
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t s = 0; s < src.Axis(0); ++s)
                            dst.CpuData(Shp(c, s))[0] = (float)src.CpuData(Shp(s, c))[0];
                }
                else
                {
                    dst.Reshape(src.Shape(), Synet::TensorFormatNchw);
                    for (size_t i = 0; i < src.Size(); ++i)
                        dst.CpuData()[i] = (float)src.CpuData()[i];
                }
            }
        }

        bool Reshape(const TestParam & param, size_t batchSize)
        {
            Strings srcNames;
            Shapes srcShapes;
            for (size_t i = 0; i < param.input().size(); ++i)
            {
                const InputParam & shape = param.input()[i];
                srcNames.push_back(shape.name());
                Shape srcShape;
                if (shape.dims().size())
                    srcShape = shape.dims();
                else
                {
                    for (size_t j = 0; j < shape.shape().size(); ++j)
                    {
                        const SizeParam& size = shape.shape()[j];
                        if (size.size() > 0)
                            srcShape.push_back(size.size());
                        else
                            SYNET_ERROR("Test parameter input.shape.size must be > 0!");
                    }
                }
                if (srcShape.size() > 1)   
                {
                    if(batchSize != 1 && srcShape[0] == 1)
                        srcShape[0] = batchSize;
                    if (_trans && srcShape.size() == 4)
                        srcShape = Shape({ srcShape[0], srcShape[2], srcShape[3], srcShape[1] });
                }
                if (srcShape.empty())
                    SYNET_ERROR("Test parameter input.shape is empty!");
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


/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void ScaleLayerForwardCpu(const T * src, const T * scale, const T * bias, size_t channels, size_t height, size_t width, T * dst, TensorFormat format, int compatibility)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    const T s = scale[c];
                    const T b = bias ? bias[c] : 0;
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            dst[w] = src[w] * s + b;
                        }
                        src += width;
                        dst += width;
                    }
                }
            }
            else if (format == TensorFormatNhwc)
            {
                if (bias)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c] + bias[c];
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c];
                            src += channels;
                            dst += channels;
                        }
                    }
                }
            }
            else
                assert(0);
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void ScaleLayerForwardCpu<float>(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float* dst, TensorFormat format, int compatibility)
        {
            ::SimdSynetScaleLayerForward(src, scale, bias, channels, height, width, dst, (::SimdTensorFormatType)format, (::SimdSynetCompatibilityType)compatibility);
        }
#endif
    }

    template <class T> class ScaleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        ScaleLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ScaleParam & param = this->Param().scale();
            _axis = param.axis();
            _biasTerm = param.biasTerm();
            assert(this->Weight().size());
            if (_biasTerm)
            {
                assert(this->Weight().size() > 1);
                assert(this->Weight()[0].Shape() == this->Weight()[1].Shape());
            }
            const Tensor & scale = this->Weight()[0];
            _channels = scale.Size();
            _type = src[0]->GetType();
            _format = src[0]->Format();
            if (scale.Size() == src[0]->Size())
            {
                _batch = 1;
                _height = 1;
                _width = 1;
            }
            else
            {
                _batch = src[0]->Size(0, _axis);
                if (src[0]->Count() < 4)
                {
                    _height = 1;
                    _width = src[0]->Size() / _batch / _channels;
                }
                else
                {
                    _height = _format == TensorFormatNhwc ? src[0]->Axis(1) : src[0]->Axis(2);
                    _width = _format == TensorFormatNhwc ? src[0]->Axis(2) : src[0]->Axis(3);
                }
            }
            assert(src[0]->Size() == _batch*_channels*_height*_width);
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
            _compatibility = 1;
        }

        virtual int64_t Flop() const
        {
            return _batch * _channels * _height * _width * 2;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * scale = this->Weight()[0].CpuData();
            const Type * bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            if (_type == TensorType32f)
            {
                ForwardCpu(src[0]->CpuData(), scale, bias, dst[0]->CpuData());
            }
            else 
                assert(0);
        }

        void ForwardCpu(const float * src, const float* scale, const float * bias, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ScaleLayerForwardCpu(src, scale, bias, _channels, _height, _width, dst, _format, _compatibility);
                src += _channels * _height * _width;
                dst += _channels * _height * _width;
            }
        }

    private:
        TensorFormat _format;
        TensorType _type;
        size_t _axis, _batch, _channels, _height, _width;
        int _compatibility;
        bool _biasTerm;
        Tensor _scale, _shift;
    };
}
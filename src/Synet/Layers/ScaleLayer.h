/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void ScaleLayerForwardCpu(const T * src, const T * scale, const T * bias, size_t channels, size_t height, size_t width, T * dst, int trans, int compatible)
        {
            if (trans)
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
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void ScaleLayerForwardCpu<float>(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float* dst, int trans, int compatible)
        {
            ::SimdSynetScaleLayerForward(src, scale, bias, channels, height, width, dst, (::SimdTensorFormatType)trans, (::SimdBool)compatible);
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
            _trans = src[0]->Format() == TensorFormatNhwc;
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
                    _height = _trans ? src[0]->Axis(1) : src[0]->Axis(2);
                    _width = _trans ? src[0]->Axis(2) : src[0]->Axis(3);
                }
            }
            assert(src[0]->Size() == _batch*_channels*_height*_width);
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
            _compatible = 1;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type* pSrc = src[0]->CpuData();
            const Type * pScale = this->Weight()[0].CpuData();
            const Type * pBias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            Type * pDst = dst[0]->CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ScaleLayerForwardCpu(pSrc, pScale, pBias, _channels, _height, _width, pDst, _trans, _compatible);
                pSrc += _channels * _height * _width;
                pDst += _channels * _height * _width;
            }
        }

    private:
        size_t _axis, _batch, _channels, _height, _width;
        int _trans, _compatible;
        bool _biasTerm;
    };
}
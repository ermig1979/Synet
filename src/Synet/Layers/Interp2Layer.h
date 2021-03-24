/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
#include "Synet/Layers/Interp2Layer.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void Interp2LayerForwardCpuBilinear(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, T * dst, size_t dstH, size_t dstW, int alignCorners, int trans)
        {
            if (trans)
            {
                float rh, rw;
                if (alignCorners)
                {
                    rh = (dstH > 1) ? static_cast<float>(sizeH - 1) / (dstH - 1) : 0.f;
                    rw = (dstW > 1) ? static_cast<float>(sizeW - 1) / (dstW - 1) : 0.f;
                }
                else
                {
                    rh = static_cast<float>(sizeH) / dstH;
                    rw = static_cast<float>(sizeW) / dstW;
                }
                for (int h2 = 0; h2 < dstH; ++h2)
                {
                    const float h1r = rh * h2;
                    const int h1 = (int)h1r;
                    const int h1p = (h1 < sizeH - 1) ? channels : 0;
                    const T h1lambda = h1r - h1;
                    const T h0lambda = T(1.) - h1lambda;
                    for (int w2 = 0; w2 < dstW; ++w2)
                    {
                        const float w1r = rw * w2;
                        const int w1 = (int)w1r;
                        const int w1p = (w1 < sizeW - 1) ? channels : 0;
                        const T w1lambda = w1r - w1;
                        const T w0lambda = T(1.) - w1lambda;
                        const T* pos1 = &src[(h1 * srcW + w1) * channels];
                        T* pos2 = &dst[(h2 * dstW + w2) * channels];
                        for (int c = 0; c < channels; ++c)
                        {
                            pos2[0] =
                                h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                                h1lambda * (w0lambda * pos1[h1p * srcW] + w1lambda * pos1[h1p * srcW + w1p]);
                            pos1 += 1;
                            pos2 += 1;
                        }
                    }
                }
            }
            else
            {
                float rh, rw;
                if (alignCorners)
                {
                    rh = (dstH > 1) ? static_cast<float>(sizeH - 1) / (dstH - 1) : 0.f;
                    rw = (dstW > 1) ? static_cast<float>(sizeW - 1) / (dstW - 1) : 0.f;
                }
                else
                {
                    rh = static_cast<float>(sizeH) / dstH;
                    rw = static_cast<float>(sizeW) / dstW;
                }
                for (int h2 = 0; h2 < dstH; ++h2)
                {
                    const float h1r = rh * h2;
                    const int h1 = (int)h1r;
                    const int h1p = (h1 < sizeH - 1) ? 1 : 0;
                    const T h1lambda = h1r - h1;
                    const T h0lambda = T(1.) - h1lambda;
                    for (int w2 = 0; w2 < dstW; ++w2)
                    {
                        const float w1r = rw * w2;
                        const int w1 = (int)w1r;
                        const int w1p = (w1 < sizeW - 1) ? 1 : 0;
                        const T w1lambda = w1r - w1;
                        const T w0lambda = T(1.) - w1lambda;
                        const T * pos1 = &src[h1 * srcW + w1];
                        T * pos2 = &dst[h2 * dstW + w2];
                        for (int c = 0; c < channels; ++c)
                        {
                            pos2[0] =
                                h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                                h1lambda * (w0lambda * pos1[h1p * srcW] + w1lambda * pos1[h1p * srcW + w1p]);
                            pos1 += srcH * srcW;
                            pos2 += dstH * dstW;
                        }
                    }
                }
            }
        }


#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        template <> inline void Interp2LayerForwardCpuBilinear<float>(size_t channels, const float * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, float * dst, size_t dstH, size_t dstW, int alignCorners, int trans)
        {
            SimdResizeMethodType method = alignCorners ? ::SimdResizeMethodCaffeInterp : ::SimdResizeMethodInferenceEngineInterp;
            if (trans)
            {
                void * resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, channels, ::SimdResizeChannelFloat, method);
                ::SimdResizerRun(resizer, (uint8_t*)src, channels * srcW * sizeof(float), (uint8_t*)dst, channels * dstW * sizeof(float));
                ::SimdRelease(resizer);
            }
            else
            {
                void * resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, 1, ::SimdResizeChannelFloat, method);
                for (size_t c = 0; c < channels; ++c)
                {
                    ::SimdResizerRun(resizer, (uint8_t*)src, srcW * sizeof(float), (uint8_t*)dst, dstW * sizeof(float));
                    src += srcH * srcW;
                    dst += dstH * dstW;
                }
                ::SimdRelease(resizer);            
            }
        }
#endif

        template <typename T> void Interp2LayerForwardCpu(size_t channels, const T * src, size_t srcH, size_t srcW, size_t padY, size_t padX, size_t padH, size_t padW, T * dst, size_t dstH, size_t dstW, int alignCorners, int trans)
        {
            size_t sizeH = srcH - padY - padH;
            size_t sizeW = srcW - padX - padW;
            src += (padY * srcW + padX)*(trans ? channels : 1);
            if (sizeH == dstH && sizeW == dstW)
                InterpLayerForwardCpuCopy(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, trans);
            else
                Interp2LayerForwardCpuBilinear(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, alignCorners, trans);
        }
    }

    template <class T> class Interp2Layer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        Interp2Layer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Interp2Param & param = this->Param().interp2();
            if (param.pad().empty())
            {
                _padY = 0;
                _padX = 0;
                _padH = 0;
                _padW = 0;
            }
            else
            {
                assert(param.pad().size() == 4);
                _padY = param.pad()[0];
                _padX = param.pad()[1];
                _padH = param.pad()[2];
                _padW = param.pad()[3];
            }
            _alignCorners = param.alignCorners() ? 1 : 0;

            _trans = src[0]->Format() == TensorFormatNhwc;
            _num = src[0]->Axis(0);
            _channels = _trans ? src[0]->Axis(3) : src[0]->Axis(1);
            _srcH = _trans ? src[0]->Axis(1) : src[0]->Axis(2);
            _srcW = _trans ? src[0]->Axis(2) : src[0]->Axis(3);

            size_t srcH = _srcH - _padY - _padH;
            size_t srcW = _srcW - _padX - _padW;
            if (param.factor() != 1.0f)
            {
                float factor = param.factor();
                if (_alignCorners)
                {
                    assert(0);
                }
                else
                {
                    _dstH = size_t(srcH * factor);
                    _dstW = size_t(srcW * factor);
                }
            }
            else if (param.height() && param.width())
            {
                _dstH = param.height();
                _dstW = param.width();
            }
            else
                assert(0);
            if(_trans)
                dst[0]->Reshape({ _num, _dstH, _dstW, _channels }, TensorFormatNhwc);
            else
                dst[0]->Reshape({ _num, _channels, _dstH, _dstW }, TensorFormatNchw);
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            for(size_t i = 0; i < _num; ++i)
            {
                Detail::Interp2LayerForwardCpu(_channels, pSrc, _srcH, _srcW, _padY, _padX, _padH, _padW, pDst, _dstH, _dstW, _alignCorners, _trans);
                pSrc += _channels*_srcH*_srcW;
                pDst += _channels*_dstH*_dstW;
            }
        }

    private:
        size_t _num, _channels, _srcH, _srcW, _dstH, _dstW, _padY, _padX, _padH, _padW;
        int _trans, _alignCorners;
    };
}
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

namespace Synet
{
    namespace Detail
    {
        template <typename T> void InterpLayerForwardCpuCopy(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, T * dst, size_t dstH, size_t dstW, int trans)
        {
            if (trans)
            {
                for (size_t h = 0; h < dstH; ++h)
                    memcpy(dst + h*dstW*channels, src + h*srcW*channels, sizeW*channels * sizeof(T));
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < dstH; ++h)
                        memcpy(dst + h*dstW, src + h*srcW, sizeW * sizeof(T));
                    src += srcH * srcW;
                    dst += dstH * dstW;
                }
            }
        }

        template <typename T> void InterpLayerForwardCpuBilinear(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, T * dst, size_t dstH, size_t dstW, int trans)
        {
            if (trans)
            {
                assert(0);
            }
            else
            {
                const float rheight = (dstH > 1) ? static_cast<float>(sizeH - 1) / (dstH - 1) : 0.f;
                const float rwidth = (dstW > 1) ? static_cast<float>(sizeW - 1) / (dstW - 1) : 0.f;
                for (int h2 = 0; h2 < dstH; ++h2)
                {
                    const float h1r = rheight * h2;
                    const int h1 = (int)h1r;
                    const int h1p = (h1 < sizeH - 1) ? 1 : 0;
                    const T h1lambda = h1r - h1;
                    const T h0lambda = T(1.) - h1lambda;
                    for (int w2 = 0; w2 < dstW; ++w2)
                    {
                        const float w1r = rwidth * w2;
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
                            pos1 += srcH * srcH;
                            pos2 += dstH * dstW;
                        }
                    }
                }
            }
        }

        template <typename T> void InterpLayerForwardCpuNearest(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, T * dst, size_t dstH, size_t dstW, int trans)
        {
            float ky = float(sizeH) / float(dstH);
            float kx = float(sizeW) / float(dstW);
            if (trans)
            {
                for (int dy = 0; dy < dstH; ++dy)
                {
                    //size_t sy = (size_t)(ky*(dy + 0.5f));
                    size_t sy = Round(dy * ky + kx / 2.0f - 0.5f);
                    for (int dx = 0; dx < dstW; ++dx)
                    {
                        //size_t sx = (size_t)(kx*(dx + 0.5f));
                        size_t sx = Round(dx * kx + ky / 2.0f - 0.5f);
                        const T * s = src + (sy * srcW + sx)*channels;
                        T * d = dst + (dy * dstW + dx)*channels;
                        memcpy(d, s, channels * sizeof(T));
                    }
                }
            }
            else
            {
                for (int dy = 0; dy < dstH; ++dy)
                {
                    //size_t sy = (size_t)(ky*(dy + 0.5f));
                    size_t sy = Round(dy * ky + kx / 2.0f - 0.5f);
                    for (int dx = 0; dx < dstW; ++dx)
                    {
                        //size_t sx = (size_t)(kx*(dx + 0.5f));
                        size_t sx = Round(dx * kx + ky / 2.0f - 0.5f);
                        const T * s = src + sy * srcW + sx;
                        T * d = dst + dy * dstW + dx;
                        for (int c = 0; c < channels; ++c)
                        {
                            d[0] = s[0];
                            s += srcH * srcW;
                            d += dstH * dstW;
                        }
                    }
                }
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        template <> inline void InterpLayerForwardCpuBilinear<float>(size_t channels, const float * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, float * dst, size_t dstH, size_t dstW, int trans)
        {
            if (trans)
            {
                void * resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, channels, ::SimdResizeChannelFloat, ::SimdResizeMethodCaffeInterp);
                ::SimdResizerRun(resizer, (uint8_t*)src, channels * srcW * sizeof(float), (uint8_t*)dst, channels * dstW * sizeof(float));
                ::SimdRelease(resizer);
            }
            else
            {
                void * resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, 1, ::SimdResizeChannelFloat, ::SimdResizeMethodCaffeInterp);
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

        template <typename T> void InterpLayerForwardCpu(size_t channels, const T * src, size_t srcH, size_t srcW, size_t cropB, size_t cropE, T * dst, size_t dstH, size_t dstW, InterpolationType type, int trans)
        {
            size_t sizeH = srcH - cropB - cropE;
            size_t sizeW = srcW - cropB - cropE;
            src += (cropB * srcW + cropB)*(trans ? channels : 1);
            if (sizeH == dstH && sizeW == dstW)
                InterpLayerForwardCpuCopy(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, trans);
            else if (type == InterpolationTypeBilinear)
                InterpLayerForwardCpuBilinear(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, trans);
            else if (type == InterpolationTypeNearest)
                InterpLayerForwardCpuNearest(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, trans);
            else
                assert(0);
        }
    }

    template <class T> class InterpLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        InterpLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual bool Resizable() const
        {
            const InterpParam& param = this->Param().interp();
            return !(param.height() && param.width());
        }

        virtual bool Can8i() const
        {
            const InterpParam& param = this->Param().interp();
            return param.interpolationType() == InterpolationTypeNearest;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const InterpParam & param = this->Param().interp();
            _cropBeg = param.cropBeg();
            _cropEnd = param.cropEnd();
            _type = param.interpolationType();
            _trans = src[0]->Format() == TensorFormatNhwc;
            _num = src[0]->Axis(0);
            _channels = _trans ? src[0]->Axis(3) : src[0]->Axis(1);
            _srcH = _trans ? src[0]->Axis(1) : src[0]->Axis(2);
            _srcW = _trans ? src[0]->Axis(2) : src[0]->Axis(3);
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
            assert(_src8u == _dst8u);
            size_t srcH = _srcH - _cropBeg - _cropEnd;
            size_t srcW = _srcW - _cropBeg - _cropEnd;
            if (param.useTensorSize())
            {
                assert(src.size() > 1 && _trans == 0);
                _dstH = _trans ? src[1]->Axis(1) : src[1]->Axis(2);
                _dstW = _trans ? src[1]->Axis(2) : src[1]->Axis(3);
            }
            else if (param.shrinkFactor() != 1 && param.zoomFactor() == 1)
            {
                size_t shrinkFactor = param.shrinkFactor();
                _dstH = (srcH - 1) / shrinkFactor + 1;
                _dstW = (srcW - 1) / shrinkFactor + 1;
            }
            else if (param.shrinkFactor() == 1 && param.zoomFactor() != 1)
            {
                size_t zoomFactor = param.zoomFactor();
                _dstH = srcH + (srcH - 1) * (zoomFactor - 1);
                _dstW = srcW + (srcW - 1) * (zoomFactor - 1);
            }
            else if (param.height() && param.width())
            {
                _dstH = param.height();
                _dstW = param.width();
            }
            else if (param.shrinkFactor() != 1 && param.zoomFactor() != 1)
            {
                size_t shrinkFactor = param.shrinkFactor();
                size_t zoomFactor = param.zoomFactor();
                _dstH = (srcH - 1) / shrinkFactor + 1;
                _dstW = (srcW - 1) / shrinkFactor + 1;
                _dstH = _dstH + (_dstH - 1) * (zoomFactor - 1);
                _dstW = _dstW + (_dstW - 1) * (zoomFactor - 1);
            }
            else
                assert(0);
            Shape dstShape = _trans ? Shp(_num, _dstH, _dstW, _channels) : Shp(_num, _channels, _dstH, _dstW);
            if (_src8u && _dst8u)
            {
                assert(param.interpolationType() == InterpolationTypeNearest);
                dst[0]->As8u().Reshape(dstShape, src[0]->Format());
            }
            else
                dst[0]->As32f().Reshape(dstShape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_src8u && _dst8u)
                ForwardCpu(src[0]->As8u().CpuData(), dst[0]->As8u().CpuData());
            else
                ForwardCpu(src[0]->As32f().CpuData(), dst[0]->As32f().CpuData());
        }

        template<class TT> void ForwardCpu(const TT * src, TT * dst)
        {
            for (size_t i = 0; i < _num; ++i)
            {
                Detail::InterpLayerForwardCpu(_channels, src, _srcH, _srcW, _cropBeg, _cropEnd, dst, _dstH, _dstW, _type, _trans);
                src += _channels * _srcH * _srcW;
                dst += _channels * _dstH * _dstW;
            }
        }

    private:
        size_t _num, _channels, _srcH, _srcW, _dstH, _dstW, _cropBeg, _cropEnd;
        InterpolationType _type;
        int _trans;
        bool _src8u, _dst8u;
    };
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Layers/Resize/InterpLayer.h"

namespace Synet
{
    template <typename T> void InterpLayerForwardCpuCopy(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, T * dst, size_t dstH, size_t dstW, TensorFormat format)
    {
        if (format == TensorFormatNhwc)
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

    template <typename T> void InterpLayerForwardCpuBilinear(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, 
        T * dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        if (format == TensorFormatNhwc)
        {
            assert(0);
        }
        else
        {
            float rheight, rwidth;
            if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypeCaffe)
            {
                rheight = (dstH > 1) ? static_cast<float>(sizeH - 1) / (dstH - 1) : 0.f;
                rwidth = (dstW > 1) ? static_cast<float>(sizeW - 1) / (dstW - 1) : 0.f;
            }
            else
                assert(0);
            for (int h2 = 0; h2 < dstH; ++h2)
            {
                const float h1r = rheight * h2;
                const int h1 = (int)h1r;
                const int h1p = (h1 < sizeH - 1) ? 1 : 0;
                const T h1lambda = T(h1r - h1);
                const T h0lambda = T(1.) - h1lambda;
                for (int w2 = 0; w2 < dstW; ++w2)
                {
                    const float w1r = rwidth * w2;
                    const int w1 = (int)w1r;
                    const int w1p = (w1 < sizeW - 1) ? 1 : 0;
                    const T w1lambda = T(w1r - w1);
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

    template <typename T> void InterpLayerForwardCpuNearest(size_t channels, const T * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, 
        T * dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        float ky, kx;
        if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypePytorch)
        {
            ky = float(sizeH) / float(dstH);
            kx = float(sizeW) / float(dstW);
        }
        else
            assert(0);
        if (format == TensorFormatNhwc)
        {
            for (int dy = 0; dy < dstH; ++dy)
            {
                size_t sy = dy * sizeH / dstH;
                for (int dx = 0; dx < dstW; ++dx)
                {
                    size_t sx = dx * sizeW / dstW;
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
                size_t sy = dy * sizeH / dstH;
                for (int dx = 0; dx < dstW; ++dx)
                {
                    size_t sx = dx * sizeW / dstW;
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
    template <> inline void InterpLayerForwardCpuBilinear<float>(size_t channels, const float * src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, 
        float * dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        SimdResizeMethodType method;
        if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypeCaffe)
            method = ::SimdResizeMethodBilinearCaffe;
        else if (coordTransfType == CoordinateTransformTypePytorch)
            method = ::SimdResizeMethodBilinearPytorch;
        else if (coordTransfType == CoordinateTransformTypeHalfPixel)
            method = ::SimdResizeMethodBilinear;
        if (format == TensorFormatNhwc)
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

    template <> inline void InterpLayerForwardCpuNearest<float>(size_t channels, const float* src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW, 
        float* dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        SimdResizeMethodType method;
        if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypePytorch)
            method = ::SimdResizeMethodNearestPytorch;
        else if (coordTransfType == CoordinateTransformTypeHalfPixel)
            method = ::SimdResizeMethodNearest;
        else
            assert(0);
        if (format == TensorFormatNhwc)
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, channels, ::SimdResizeChannelFloat, method);
            ::SimdResizerRun(resizer, (uint8_t*)src, channels * srcW * sizeof(float), (uint8_t*)dst, channels * dstW * sizeof(float));
            ::SimdRelease(resizer);
        }
        else
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, 1, ::SimdResizeChannelFloat, method);
            for (size_t c = 0; c < channels; ++c)
            {
                ::SimdResizerRun(resizer, (uint8_t*)src, srcW * sizeof(float), (uint8_t*)dst, dstW * sizeof(float));
                src += srcH * srcW;
                dst += dstH * dstW;
            }
            ::SimdRelease(resizer);
        }
    }

    template <> inline void InterpLayerForwardCpuBilinear<uint16_t>(size_t channels, const uint16_t* src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW,
        uint16_t* dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        SimdResizeMethodType method;
        if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypeCaffe)
            method = ::SimdResizeMethodBilinearCaffe;
        else if (coordTransfType == CoordinateTransformTypePytorch)
            method = ::SimdResizeMethodBilinearPytorch;
        else if (coordTransfType == CoordinateTransformTypeHalfPixel)
            method = ::SimdResizeMethodBilinear;
        if (format == TensorFormatNhwc)
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, channels, ::SimdResizeChannelBf16, method);
            ::SimdResizerRun(resizer, (uint8_t*)src, channels * srcW * sizeof(uint16_t), (uint8_t*)dst, channels * dstW * sizeof(uint16_t));
            ::SimdRelease(resizer);
        }
        else
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, 1, ::SimdResizeChannelBf16, method);
            for (size_t c = 0; c < channels; ++c)
            {
                ::SimdResizerRun(resizer, (uint8_t*)src, srcW * sizeof(uint16_t), (uint8_t*)dst, dstW * sizeof(uint16_t));
                src += srcH * srcW;
                dst += dstH * dstW;
            }
            ::SimdRelease(resizer);
        }
    }

    template <> inline void InterpLayerForwardCpuNearest<uint16_t>(size_t channels, const uint16_t* src, size_t srcH, size_t srcW, size_t sizeH, size_t sizeW,
        uint16_t* dst, size_t dstH, size_t dstW, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        SimdResizeMethodType method;
        if (coordTransfType == CoordinateTransformTypeLegacy || coordTransfType == CoordinateTransformTypePytorch)
            method = ::SimdResizeMethodNearestPytorch;
        else if (coordTransfType == CoordinateTransformTypeHalfPixel)
            method = ::SimdResizeMethodNearest;
        else
            assert(0);
        if (format == TensorFormatNhwc)
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, channels, ::SimdResizeChannelBf16, method);
            ::SimdResizerRun(resizer, (uint8_t*)src, channels * srcW * sizeof(uint16_t), (uint8_t*)dst, channels * dstW * sizeof(uint16_t));
            ::SimdRelease(resizer);
        }
        else
        {
            void* resizer = ::SimdResizerInit(sizeW, sizeH, dstW, dstH, 1, ::SimdResizeChannelBf16, method);
            for (size_t c = 0; c < channels; ++c)
            {
                ::SimdResizerRun(resizer, (uint8_t*)src, srcW * sizeof(uint16_t), (uint8_t*)dst, dstW * sizeof(uint16_t));
                src += srcH * srcW;
                dst += dstH * dstW;
            }
            ::SimdRelease(resizer);
        }
    }
#endif

    template <typename T> void InterpLayerForwardCpu(size_t channels, const uint8_t * src8, size_t srcH, size_t srcW, size_t cropB, size_t cropE, uint8_t* dst8, size_t dstH, size_t dstW,
        InterpolationType interpType, CoordinateTransformType coordTransfType, TensorFormat format)
    {
        const T* src = (T*)src8;
        T* dst = (T*)dst8;
        size_t sizeH = srcH - cropB - cropE;
        size_t sizeW = srcW - cropB - cropE;
        src += (cropB * srcW + cropB)*(format == TensorFormatNhwc ? channels : 1);
        if (sizeH == dstH && sizeW == dstW)
            InterpLayerForwardCpuCopy(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, format);
        else if (interpType == InterpolationTypeBilinear)
            InterpLayerForwardCpuBilinear(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, coordTransfType, format);
        else if (interpType == InterpolationTypeNearest)
            InterpLayerForwardCpuNearest(channels, src, srcH, srcW, sizeH, sizeW, dst, dstH, dstW, coordTransfType, format);
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    InterpLayer::InterpLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool InterpLayer::Resizable() const
    {
        const InterpParam& param = this->Param().interp();
        return !(param.height() && param.width());
    }

    LowPrecisionType InterpLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType8u && p.interp().interpolationType() == InterpolationTypeNearest)
            return LowPrecisionTypePassive;
        if (type == TensorType16b && p.interp().interpolationType() == InterpolationTypeNearest)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    int64_t InterpLayer::Flop() const
    {
        int64_t size = _batch * _channels * _dstH * _dstW;
        switch (_interpType)
        {
        case InterpolationTypeNearest: return size * 1;
        case InterpolationTypeBilinear: return size * 9;
        default:
            return size * 0;
        }
    }

    bool InterpLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("InterpLayer supports 1 or 2 inputs and 1 output!");
        if (src[0]->GetType() != dst[0]->GetType())
            SYNET_ERROR("InterpLayer src[0] and dst[0] must have the same type!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("InterpLayer supports only 4D input tensor!");
        if (!(src[0]->GetType() == TensorType32f || src[0]->GetType() == TensorType8u || src[0]->GetType() == TensorType16b))
            SYNET_ERROR("InterpLayer supports only FP32, INT8 or BF16 input format!");

        const InterpParam & param = this->Param().interp();
        _cropBeg = param.cropBeg();
        _cropEnd = param.cropEnd();
        _interpType = param.interpolationType();
        _coordTransfType = param.coordinateTransformType();
        _format = src[0]->Format();
        _type = src[0]->GetType();
        _batch = src[0]->Axis(0);
        _elem = src[0]->TypeSize();
        if (_format == TensorFormatNhwc)
        {
            _channels = src[0]->Axis(3);
            _srcH = src[0]->Axis(1);
            _srcW = src[0]->Axis(2);
        }
        else
        {
            _channels = src[0]->Axis(1);
            _srcH = src[0]->Axis(2);
            _srcW = src[0]->Axis(3);
        }
        size_t srcH = _srcH - _cropBeg - _cropEnd;
        size_t srcW = _srcW - _cropBeg - _cropEnd;
        if (src.size() == 2 || Weight().size())
        {
            if (param.useTensorSize())
            {
                if (src[1]->Count() != 4)
                    SYNET_ERROR("InterpLayer src[2] must be 4D tensor if interp().useTensorSize() == true!");
                if (_format == TensorFormatNhwc)
                {
                    _dstH = src[1]->Axis(1);
                    _dstW = src[1]->Axis(2);
                }
                else
                {
                    _dstH = src[1]->Axis(2);
                    _dstW = src[1]->Axis(3);
                }
            } 
            else
            {
                const Tensor& src1 = src.size() > 1 ? *src[1] : Weight()[0];
                if (src1.GetType() == TensorType32f)
                {
                    const float * factor = src1.Data<float>();
                    _dstH = size_t(srcH * factor[2]);
                    _dstW = size_t(srcW * factor[3]);
                }
                else if (src1.GetType() == TensorType64i)
                {
                    const int64_t * sizes = src1.Data<int64_t>();
                    _dstH = size_t(sizes[2]);
                    _dstW = size_t(sizes[3]);
                }
                else
                    SYNET_ERROR("InterpLayer src[1] has wrong type!");
            }
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
            SYNET_ERROR("InterpLayer: unknown output shape!");
        if(_interpType == InterpolationTypeBilinear && _type != TensorType32f)
            SYNET_ERROR("InterpLayer: Bilinear interpolation is supported only for FP32 type!");

        Shape dstShape = _format == TensorFormatNhwc ? 
            Shp(_batch, _dstH, _dstW, _channels) : Shp(_batch, _channels, _dstH, _dstW);
        dst[0]->Reshape(_type, dstShape, src[0]->Format());
        _const = false;
        if(Options().BFloat16Enable())
            this->UsePerfStat(Cpl::ToStr(_type));
        else
            this->UsePerfStat(Cpl::ToStr(_type));
        return true;
    }

    void InterpLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const uint8_t* src8 = src[0]->RawData();
         uint8_t* dst8 = dst[0]->RawData();
        for (size_t b = 0; b < _batch; ++b)
        {
            switch (_type)
            {
            case TensorType32f: InterpLayerForwardCpu<float>(_channels, src8, _srcH, _srcW, _cropBeg, _cropEnd, dst8, _dstH, _dstW, _interpType, _coordTransfType, _format); break;
            case TensorType8u: InterpLayerForwardCpu<uint8_t>(_channels, src8, _srcH, _srcW, _cropBeg, _cropEnd, dst8, _dstH, _dstW, _interpType, _coordTransfType, _format); break;
            case TensorType16b: InterpLayerForwardCpu<uint16_t>(_channels, src8, _srcH, _srcW, _cropBeg, _cropEnd, dst8, _dstH, _dstW, _interpType, _coordTransfType, _format); break;
            }
            src8 += _channels * _srcH * _srcW * _elem;
            dst8 += _channels * _dstH * _dstW * _elem;
        }
    }
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/PoolingLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T> void PoolingMax2D(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, T * dst, size_t dstH, size_t dstW, TensorFormat format)
    {
        if (format == TensorFormatNhwc)
        {
            for (size_t ph = 0; ph < dstH; ++ph)
            {
                size_t hStart = ph * strideY - padY;
                size_t hEnd = Min(hStart + kernelY, srcH);
                hStart = Max<ptrdiff_t>(0, hStart);
                for (size_t pw = 0; pw < dstW; ++pw)
                {
                    size_t wStart = pw * strideX - padX;
                    size_t wEnd = Min(wStart + kernelX, srcW);
                    wStart = Max<ptrdiff_t>(0, wStart);
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = std::numeric_limits<T>::lowest();
                    for (size_t h = hStart; h < hEnd; ++h)
                    {
                        for (size_t w = wStart; w < wEnd; ++w)
                        {
                            const T* pc = src + (h * srcW + w) * channels;
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = Max(dst[c], pc[c]);
                        }
                    }
                    dst += channels;
                }
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t ph = 0; ph < dstH; ++ph)
                {
                    size_t hStart = ph * strideY - padY;
                    size_t hEnd = Min(hStart + kernelY, srcH);
                    hStart = Max<ptrdiff_t>(0, hStart);
                    for (size_t pw = 0; pw < dstW; ++pw)
                    {
                        size_t wStart = pw * strideX - padX;
                        size_t wEnd = Min(wStart + kernelX, srcW);
                        wStart = Max<ptrdiff_t>(0, wStart);
                        T max = std::numeric_limits<T>::lowest();
                        for (size_t h = hStart; h < hEnd; ++h)
                            for (size_t w = wStart; w < wEnd; ++w)
                                max = Max(max, src[h * srcW + w]);
                        dst[ph * dstW + pw] = max;
                    }
                }
                src += srcW * srcH;
                dst += dstW * dstH;
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> void PoolingAverage2D(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, T * dst, size_t dstH, size_t dstW, int excludePad, TensorFormat format)
    {
        typedef T Type;
        assert((std::is_same<Type, float>::value));
        if (format == TensorFormatNhwc)
        {
            for (size_t ph = 0; ph < dstH; ++ph)
            {
                size_t hStart = ph * strideY - padY;
                size_t hEnd = Min(hStart + kernelY, srcH);
                hStart = Max<ptrdiff_t>(0, hStart);
                for (size_t pw = 0; pw < dstW; ++pw)
                {
                    size_t wStart = pw * strideX - padX;
                    size_t wEnd = Min(wStart + kernelX, srcW);
                    wStart = Max<ptrdiff_t>(0, wStart);
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = Type(0);
                    for (size_t h = hStart; h < hEnd; ++h)
                    {
                        for (size_t w = wStart; w < wEnd; ++w)
                        {
                            const Type * pc = src + (h * srcW + w)*channels;
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] += pc[c];
                        }
                    }
                    if (excludePad)
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = dst[c] / Type((hEnd - hStart) * (wEnd - wStart));
                    else
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = dst[c] / Type(kernelY * kernelX);
                    dst += channels;
                }
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t ph = 0; ph < dstH; ++ph)
                {
                    size_t hStart = ph * strideY - padY;
                    size_t hEnd = Min(hStart + kernelY, srcH);
                    hStart = Max<ptrdiff_t>(0, hStart);
                    for (size_t pw = 0; pw < dstW; ++pw)
                    {
                        size_t wStart = pw * strideX - padX;
                        size_t wEnd = Min(wStart + kernelX, srcW);
                        wStart = Max<ptrdiff_t>(0, wStart);
                        Type sum = Type(0);
                        for (size_t h = hStart; h < hEnd; ++h)
                            for (size_t w = wStart; w < wEnd; ++w)
                                sum += src[h * srcW + w];
                        if (excludePad)
                            dst[ph*dstW + pw] = sum / Type((hEnd - hStart) * (wEnd - wStart));
                        else
                            dst[ph*dstW + pw] = sum / Type(kernelY * kernelX);
                    }
                }
                src += srcW * srcH;
                dst += dstW * dstH;
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> void PoolingMax3D(const T * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelC, size_t kernelY, size_t kernelX,
        size_t strideC, size_t strideY, size_t strideX, size_t padC, size_t padY, size_t padX, T * dst, size_t dstC, size_t dstH, size_t dstW, TensorFormat format)
    {
        if (format == TensorFormatNhwc)
        {
            for (size_t dh = 0; dh < dstH; ++dh)
            {
                size_t hBeg = dh * strideY - padY;
                size_t hEnd = Min(hBeg + kernelY, srcH);
                hBeg = Max<ptrdiff_t>(0, hBeg);
                for (size_t dw = 0; dw < dstW; ++dw)
                {
                    size_t wBeg = dw * strideX - padX;
                    size_t wEnd = Min(wBeg + kernelX, srcW);
                    wBeg = Max<ptrdiff_t>(0, wBeg);
                    for (size_t dc = 0; dc < dstC; ++dc)
                    {
                        size_t cBeg = dc * strideC - padC;
                        size_t cEnd = Min(cBeg + kernelC, srcC);
                        cBeg = Max<ptrdiff_t>(0, cBeg);
                        T max = std::numeric_limits<T>::lowest();
                        for (size_t sh = hBeg; sh < hEnd; ++sh)
                        {
                            for (size_t sw = wBeg; sw < wEnd; ++sw)
                            {
                                const T* ps = src + (sh * srcW + sw) * srcC;
                                for (size_t c = cBeg; c < cEnd; ++c)
                                    max = Max(max, ps[c]);
                            }
                        }
                        dst[(dh * dstW + dw) * dstC + dc] = max;
                    }
                }
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                size_t cBeg = dc * strideC - padC;
                size_t cEnd = Min(cBeg + kernelC, srcC);
                cBeg = Max<ptrdiff_t>(0, cBeg);
                for (size_t dh = 0; dh < dstH; ++dh)
                {
                    size_t hBeg = dh * strideY - padY;
                    size_t hEnd = Min(hBeg + kernelY, srcH);
                    hBeg = Max<ptrdiff_t>(0, hBeg);
                    for (size_t dw = 0; dw < dstW; ++dw)
                    {
                        size_t wBeg = dw * strideX - padX;
                        size_t wEnd = Min(wBeg + kernelX, srcW);
                        wBeg = Max<ptrdiff_t>(0, wBeg);
                        T max = std::numeric_limits<T>::lowest();
                        for (size_t sc = cBeg; sc < cEnd; ++sc)
                            for (size_t sh = hBeg; sh < hEnd; ++sh)
                                for (size_t sw = wBeg; sw < wEnd; ++sw)
                                    max = Max(max, src[(sc * srcH + sh) * srcW + sw]);
                        dst[(dc * dstH + dh) * dstW + dw] = max;
                    }
                }
            }
        }
        else
            assert(0);
    }

    //-----------------------------------------------------------------------------------------

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> SYNET_INLINE void PoolingMax2D<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, TensorFormat format)
    {
        SimdSynetPoolingMax32f(src, channels, srcH, srcW, 1, kernelY, kernelX, 1, strideY, strideX, 0, padY, padX, dst, channels, dstH, dstW, (SimdTensorFormatType)format);
    }

    template <> SYNET_INLINE void PoolingMax3D<float>(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelC, size_t kernelY, size_t kernelX,
        size_t strideC, size_t strideY, size_t strideX, size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, TensorFormat format)
    {
        SimdSynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX, strideC, strideY, strideX, 
            padC, padY, padX, dst, dstC, dstH, dstW, (SimdTensorFormatType)format);
    }

    template <> SYNET_INLINE void PoolingMax2D<uint8_t>(const uint8_t* src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, TensorFormat format)
    {
        SimdSynetPoolingMax8u(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (SimdTensorFormatType)format);
    }

    template <> SYNET_INLINE void PoolingAverage2D<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, int excludePad, TensorFormat format)
    {
        SimdSynetPoolingAverage(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdBool)excludePad, (SimdTensorFormatType)format);
    }
#endif

    //-------------------------------------------------------------------------------------------------

    PoolingLayer::PoolingLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        _method = this->Param().pooling().method();
    }

    int64_t PoolingLayer::Flop() const
    {
        return _const ? int64_t(0) : _batch * _kernelC * _kernelY * _kernelX * _dstC * _dstH * _dstW;
    }

    LowPrecisionType PoolingLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u && _method == PoolingMethodTypeMax)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }
        
    bool PoolingLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("PoolingLayer supports only 1-2 inputs and 1 output!");

        const PoolingParam & param = this->Param().pooling();

        _yoloCompatible = param.yoloCompatible();
        _roundingType = param.roundingType();
        _excludePad = param.excludePad();
        if (src[0]->Count() < 4)
            SYNET_ERROR("PoolingLayer input must have at least 4 dimensions!");

        _type = src[0]->GetType();
        _format = src[0]->Format();
        _3d = false;

        _batch = src[0]->Axis(0);
        _srcC = _format == TensorFormatNhwc ? src[0]->Axis(-1) : src[0]->Axis(-3);
        _srcH = _format == TensorFormatNhwc ? src[0]->Axis(-3) : src[0]->Axis(-2);
        _srcW = _format == TensorFormatNhwc ? src[0]->Axis(-2) : src[0]->Axis(-1);

        if (param.globalPooling())
        {
            _kernelC = 1;
            _kernelY = _srcH;
            _kernelX = _srcW;
        }
        else
        {
            const Shape & kernel = param.kernel();
            if(kernel.size() < 1 && kernel.size() > 3)
                SYNET_ERROR("PoolingLayer parameter kernel size must be in range [1..3]!");
            if (kernel.size() == 3)
            {
                _kernelC = kernel[0];
                _kernelY = kernel[1];
                _kernelX = kernel[2];
                _3d = true;
                if(param.stride().size() != 3 || param.pad().size() != 6)
                    SYNET_ERROR("PoolingLayer parameter stride (or pad) has wrong size!");
            }
            else
            {
                _kernelC = 1;
                _kernelY = kernel[0];
                _kernelX = kernel.size() > 1 ? kernel[1] : kernel[0];
            }
            if (_kernelC <= 0 || _kernelY <= 0 || _kernelX <= 0)
                SYNET_ERROR("PoolingLayer parameter kernel must be positive!");
        }

        const Shape & stride = param.stride();
        if (stride.empty())
        {
            _strideC = 1;
            _strideY = 1;
            _strideX = 1;
        }
        else
        {
            if(stride.size() < 1 && stride.size() > 3)
                SYNET_ERROR("PoolingLayer parameter stride has wrong size!");
            if (stride.size() == 3)
            {
                _strideC = stride[0];
                _strideY = stride[1];
                _strideX = stride[2];
            }
            else
            {
                _strideC = 1;
                _strideY = stride[0];
                _strideX = stride.size() > 1 ? stride[1] : stride[0];
            }
        }

        const Shape & pad = param.pad();
        if (param.padType() == PoolingPadTypeTensorflowSame)
        {
            if(_3d != false)
                SYNET_ERROR("PoolingLayer does not support 3D pooling for PoolingPadTypeTensorflowSame!");
            if (_strideX == 2 && _strideY == 2)
            {
                if (_kernelX == 3 && _kernelY == 3)
                {
                    _padC = 0;
                    _padY = _srcH%_strideY;
                    _padX = _srcW%_strideX;
                    _padD = 0;
                    _padH = 1;
                    _padW = 1;
                }
                else if (_kernelX == 2 && _kernelY == 2)
                {
                    _padC = 0;
                    _padY = 0;
                    _padX = 0;
                    _padD = 0;
                    _padH = _srcH%_strideY;
                    _padW = _srcW%_strideX;
                }
                else
                    SYNET_ERROR("PoolingLayer wrong kernel parameter for PoolingPadTypeTensorflowSame!");
            }
            else
                SYNET_ERROR("PoolingLayer wrong stride parameter for PoolingPadTypeTensorflowSame!");
        }
        else if (pad.empty())
        {
            _padC = 0;
            _padY = 0;
            _padX = 0;
            _padD = 0;
            _padH = 0;
            _padW = 0;
        }
        else if (pad.size() == 1)
        {
            _padC = 0;
            _padY = pad[0];
            _padX = pad[0];
            _padD = 0;
            _padH = pad[0];
            _padW = pad[0];
        }
        else if (pad.size() == 2)
        {
            _padC = 0;
            _padY = pad[0];
            _padX = pad[1];
            _padD = 0;
            _padH = pad[0];
            _padW = pad[1];
        }
        else if (pad.size() == 4)
        {
            _padC = 0;
            _padY = pad[0];
            _padX = pad[1];
            _padD = 0;
            _padH = pad[2];
            _padW = pad[3];
        }
        else if (pad.size() == 6)
        {
            _padC = pad[0];
            _padY = pad[1];
            _padX = pad[2];
            _padD = pad[3];
            _padH = pad[4];
            _padW = pad[5];
        }
        else
            SYNET_ERROR("PoolingLayer wrong pad parameter!");

        if (_yoloCompatible == 2)
        {
            _dstC = (_srcC + _padD - _kernelC) / _strideC + 1;
            _dstH = (_srcH + _padH - _kernelY) / _strideY + 1;
            _dstW = (_srcW + _padW - _kernelX) / _strideX + 1;
        }
        else if (_yoloCompatible == 1)
        {
            _dstC = (_srcC + _padC + _padD) / _strideC;
            _dstH = (_srcH + _padY + _padH) / _strideY;
            _dstW = (_srcW + _padX + _padW) / _strideX;
        }
        else
        {
            if (_roundingType == RoundingTypeCeil)
            {
                _dstC = (size_t)(::ceil((float)(_srcC + _padC + _padD - _kernelC) / _strideC)) + 1;
                _dstH = (size_t)(::ceil((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
                _dstW = (size_t)(::ceil((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
            }
            else if (_roundingType == RoundingTypeFloor)
            {
                _dstC = (size_t)(::floor((float)(_srcC + _padC + _padD - _kernelC) / _strideC)) + 1;
                _dstH = (size_t)(::floor((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
                _dstW = (size_t)(::floor((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
            }
            else
                SYNET_ERROR("PoolingLayer wrong roundingTupe parameter!");
            if (_padC || _padX || _padY)
            {
                if ((_dstC - 1) * _strideC >= _srcC + _padC)
                    --_dstC;
                if ((_dstH - 1) * _strideY >= _srcH + _padY)
                    --_dstH;
                if ((_dstW - 1) * _strideX >= _srcW + _padX)
                    --_dstW;
                if((_dstC - 1) * _strideC >= _srcC + _padC || (_dstH - 1) * _strideY >= _srcH + _padY || (_dstW - 1) * _strideX >= _srcW + _padX)
                    SYNET_ERROR("PoolingLayer check stride and pad parameters!");
            }
        }

        bool skip = _kernelC == 1 && _kernelX == 1 && _kernelY == 1 && 
            _strideC == 1 && _strideY == 1 && _strideX == 1 && 
            _padC == 0 && _padY == 0 && _padX == 0 && _padD == 0 && _padH == 0 && _padW == 0;

        if (skip)
        {
            _const = true;
            dst[0]->Share(*src[0]);
        }
        else
        {
            if(_type != TensorType32f && _method == PoolingMethodTypeAverage)
                SYNET_ERROR("PoolingLayer does not support 3D pooling for PoolingMethodTypeAverage!");
            if(_type != TensorType32f && _type != TensorType8u && _type != TensorType8i)
                SYNET_ERROR("PoolingLayer: unsupported input type!");

            TensorFormat format = src[0]->Format();
            Shape shape = src[0]->Shape();
            if (_format == TensorFormatNhwc)
            {
                shape[shape.size() - 3] = _dstH;
                shape[shape.size() - 2] = _dstW;
                shape[shape.size() - 1] = _dstC;
            }
            else
            {
                shape[shape.size() - 3] = _dstC;
                shape[shape.size() - 2] = _dstH;
                shape[shape.size() - 1] = _dstW;
            }
            dst[0]->Reshape(_type, shape, format);

            std::stringstream desc;
            desc << _batch << "x" << _srcC << "x" << _srcH << "x" << _srcW;
            if(_3d)
                desc << "-" << _kernelC << "x" << _kernelY << "x" << _kernelX << "-" << _strideC << "x" << _strideY << "x" << _strideX;
            else
                desc << "-" << _kernelY << "x" << _kernelX << "-" << _strideY << "x" << _strideX;
            desc << (_method ? " avg" : " max") << "-" << Cpl::ToStr(_type);
            this->UsePerfStat(desc.str(), Flop());
            _const = false;
        }
        return true;
    }

    void PoolingLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_type)
        {
        case TensorType32f: ForwardCpu(src[0]->Data<float>(), dst[0]->Data<float>()); break;
        case TensorType8u: ForwardCpu(src[0]->Data<uint8_t>(), dst[0]->Data<uint8_t>()); break;
        case TensorType8i: ForwardCpu(src[0]->Data<int8_t>(), dst[0]->Data<int8_t>()); break;
        }
    }

    template <class T> void PoolingLayer::ForwardCpu(const T * src, T * dst)
    {
        switch (_method)
        {
        case PoolingMethodTypeMax:
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_3d)
                {
                    PoolingMax3D(src, _srcC, _srcH, _srcW, _kernelC, _kernelY, _kernelX,
                        _strideC, _strideY, _strideX, _padC, _padY, _padX, dst, _dstC, _dstH, _dstW, _format);
                }
                else
                {
                    PoolingMax2D(src, _srcC, _srcH, _srcW, _kernelY, _kernelX,
                        _strideY, _strideX, _padY, _padX, dst, _dstH, _dstW, _format);
                }
                src += _srcC *_srcW * _srcH;
                dst += _dstC *_dstW * _dstH;
            }
            break;
        case PoolingMethodTypeAverage:
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_3d)
                {
                    assert(0);
                }
                else
                {
                    PoolingAverage2D(src, _srcC, _srcH, _srcW, _kernelY, _kernelX,
                        _strideY, _strideX, _padY, _padX, dst, _dstH, _dstW, _excludePad, _format);
                }
                src += _srcC *_srcW * _srcH;
                dst += _dstC *_dstW * _dstH;
            }
            break;
        default:
            assert(0);
        }
    }
}
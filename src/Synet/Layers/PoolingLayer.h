/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
        template <class T> void PoolingForwardCpuMax2D(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
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

        //-----------------------------------------------------------------------------------------

        template <class T> void PoolingForwardCpuAverage2D(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
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

        //-----------------------------------------------------------------------------------------

        template <class T> void PoolingForwardCpuMax3D(const T * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelC, size_t kernelY, size_t kernelX,
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
        template <> SYNET_INLINE void PoolingForwardCpuMax2D<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, TensorFormat format)
        {
            ::SimdSynetPoolingMax32f(src, channels, srcH, srcW, 1, kernelY, kernelX, 1, strideY, strideX, 0, padY, padX, dst, channels, dstH, dstW, (::SimdTensorFormatType)format);
        }

        template <> SYNET_INLINE void PoolingForwardCpuMax3D<float>(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelC, size_t kernelY, size_t kernelX,
            size_t strideC, size_t strideY, size_t strideX, size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, TensorFormat format)
        {
            ::SimdSynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX, strideC, strideY, strideX, 
                padC, padY, padX, dst, dstC, dstH, dstW, (::SimdTensorFormatType)format);
        }

        template <> SYNET_INLINE void PoolingForwardCpuMax2D<uint8_t>(const uint8_t* src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, TensorFormat format)
        {
            ::SimdSynetPoolingMax8u(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdTensorFormatType)format);
        }

        template <> SYNET_INLINE void PoolingForwardCpuAverage2D<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, int excludePad, TensorFormat format)
        {
            ::SimdSynetPoolingAverage(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdBool)excludePad, (::SimdTensorFormatType)format);
        }
#endif
    }

    template <class T> class PoolingLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PoolingLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
            _method = this->Param().pooling().method();
        }

        virtual int64_t Flop() const
        {
            return _skip ? int64_t(0) : _batch * _kernelC * _kernelY * _kernelX * _dstC * _dstH * _dstW;
        }

        virtual bool Can8i() const
        {
            return _method == PoolingMethodTypeMax;
        }

        virtual bool HasZero() const
        {
            return _padC || _padY || _padH || _padD || _padH || _padW;
        }
        
        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PoolingParam & param = this->Param().pooling();

            _yoloCompatible = param.yoloCompatible();
            _roundingType = param.roundingType();
            _excludePad = param.excludePad();
            assert(src[0]->Count() >= 4);

            _type = src[0]->GetType();
            _format = src[0]->Format();
            _trans = _format == TensorFormatNhwc;
            _3d = false;

            _batch = src[0]->Axis(0);
            _srcC = _trans ? src[0]->Axis(-1) : src[0]->Axis(-3);
            _srcH = _trans ? src[0]->Axis(-3) : src[0]->Axis(-2);
            _srcW = _trans ? src[0]->Axis(-2) : src[0]->Axis(-1);

            if (param.globalPooling())
            {
                _kernelC = 1;
                _kernelY = _srcH;
                _kernelX = _srcW;
            }
            else
            {
                const Shape & kernel = param.kernel();
                assert(kernel.size() >= 1 && kernel.size() <= 3);
                if (kernel.size() == 3)
                {
                    _kernelC = kernel[0];
                    _kernelY = kernel[1];
                    _kernelX = kernel[2];
                    _3d = true;
                    assert(param.stride().size() == 3 && param.pad().size() == 6);
                }
                else
                {
                    _kernelC = 1;
                    _kernelY = kernel[0];
                    _kernelX = kernel.size() > 1 ? kernel[1] : kernel[0];
                }
                assert(_kernelC > 0 && _kernelY > 0 && _kernelX > 0);
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
                assert(stride.size() >= 1 && stride.size() <= 3);
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
                assert(_3d == false);
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
                        assert(0);
                }
                else
                    assert(0);
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
                assert(0);

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
                    assert(0);
                if (_padC || _padX || _padY)
                {
                    if ((_dstC - 1) * _strideC >= _srcC + _padC)
                        --_dstC;
                    if ((_dstH - 1) * _strideY >= _srcH + _padY)
                        --_dstH;
                    if ((_dstW - 1) * _strideX >= _srcW + _padX)
                        --_dstW;
                    assert((_dstC - 1) * _strideC < _srcC + _padC);
                    assert((_dstH - 1) * _strideY < _srcH + _padY);
                    assert((_dstW - 1) * _strideX < _srcW + _padX);
                }
            }

            _skip = _kernelC == 1 && _kernelX == 1 && _kernelY == 1 && 
                _strideC == 1 && _strideY == 1 && _strideX == 1 && 
                _padC == 0 && _padY == 0 && _padX == 0 && _padD == 0 && _padH == 0 && _padW == 0;

            if (_skip)
                dst[0]->Share(*src[0]);
            else
            {
                assert(_type == TensorType32f || _method == PoolingMethodTypeMax);

                TensorFormat format = src[0]->Format();
                Shape shape = src[0]->Shape();
                if (_trans)
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
                switch (_type)
                {
                case TensorType32f: dst[0]->As32f().Reshape(shape, format); break;
                case TensorType8u: dst[0]->As8u().Reshape(shape, format); break;
                case TensorType8i: dst[0]->As8i().Reshape(shape, format); break;
                default:
                    assert(0);
                }
                std::stringstream desc;
                desc << _batch << "x" << _srcC << "x" << _srcH << "x" << _srcW;
                if(_3d)
                    desc << "-" << _kernelC << "x" << _kernelY << "x" << _kernelX << "-" << _strideC << "x" << _strideY << "x" << _strideX;
                else
                    desc << "-" << _kernelY << "x" << _kernelX << "-" << _strideY << "x" << _strideX;
                desc << (_method ? " avg" : " max");
                this->UsePerfStat(desc.str(), Flop());
            }
        }

    protected:

        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_skip)
                return;

            switch (_type)
            {
            case TensorType32f: ForwardCpu(src[0]->As32f().CpuData(), dst[0]->As32f().CpuData()); break;
            case TensorType8u: ForwardCpu(src[0]->As8u().CpuData(), dst[0]->As8u().CpuData()); break;
            case TensorType8i: ForwardCpu(src[0]->As8i().CpuData(), dst[0]->As8i().CpuData()); break;
            }
        }

        template <class TT> void ForwardCpu(const TT * src, TT * dst)
        {
            switch (_method)
            {
            case PoolingMethodTypeMax:
                for (size_t b = 0; b < _batch; ++b)
                {
                    if (_3d)
                    {
                        Detail::PoolingForwardCpuMax3D(src, _srcC, _srcH, _srcW, _kernelC, _kernelY, _kernelX,
                            _strideC, _strideY, _strideX, _padC, _padY, _padX, dst, _dstC, _dstH, _dstW, _format);
                    }
                    else
                    {
                        Detail::PoolingForwardCpuMax2D(src, _srcC, _srcH, _srcW, _kernelY, _kernelX,
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
                        Detail::PoolingForwardCpuAverage2D(src, _srcC, _srcH, _srcW, _kernelY, _kernelX,
                            _strideY, _strideX, _padY, _padX, dst, _dstH, _dstW, _excludePad, _format);
                    }
                    src += _srcC *_srcW * _srcH;
                    dst += _dstC *_dstW * _dstH;
                }
                break;
            case PoolingMethodTypeStochastic:
                assert(0);
                break;
            default:
                assert(0);
            }
        }

    private:
        TensorFormat _format;
        TensorType _type;
        PoolingMethodType _method;
        RoundingType _roundingType;
        bool _skip, _3d;
        int _yoloCompatible, _trans, _excludePad;
        size_t _batch, _srcC, _srcH, _srcW, _kernelC, _kernelY, _kernelX, _strideC, _strideY, _strideX, 
            _padC, _padY, _padX, _padD, _padH, _padW, _dstC, _dstH, _dstW;
    };
}
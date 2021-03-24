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
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void PoolingForwardCpuMax(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
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

        template <class T> void PoolingForwardCpuAverage(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
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

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        template <> SYNET_INLINE void PoolingForwardCpuMax<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, TensorFormat format)
        {
            ::SimdSynetPoolingForwardMax32f(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdTensorFormatType)format);
        }

        template <> SYNET_INLINE void PoolingForwardCpuMax<uint8_t>(const uint8_t* src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, TensorFormat format)
        {
            ::SimdSynetPoolingForwardMax8u(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdTensorFormatType)format);
        }

        template <> SYNET_INLINE void PoolingForwardCpuAverage<float>(const float * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, int excludePad, TensorFormat format)
        {
            ::SimdSynetPoolingForwardAverage(src, channels, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, (::SimdBool)excludePad, (::SimdTensorFormatType)format);
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
            return _skip ? int64_t(0) : _num * _kernelY * _kernelX * _channels * _dstH * _dstW;
        }

        virtual bool Can8i() const
        {
            return _method == PoolingMethodTypeMax;
        }

        virtual bool HasZero() const
        {
            return _padY || _padH || _padH || _padW;
        }
        
        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PoolingParam & param = this->Param().pooling();

            _yoloCompatible = param.yoloCompatible();
            _roundingType = param.roundingType();
            _excludePad = param.excludePad();
            assert(src[0]->Count() == 4);

            _type = src[0]->GetType();
            _format = src[0]->Format();
            _trans = _format == TensorFormatNhwc;

            _num = src[0]->Axis(0);
            _channels = _trans ? src[0]->Axis(3) : src[0]->Axis(1);
            _srcH = _trans ? src[0]->Axis(1) : src[0]->Axis(2);
            _srcW = _trans ? src[0]->Axis(2) : src[0]->Axis(3);

            if (param.globalPooling())
            {
                _kernelY = _srcH;
                _kernelX = _srcW;
            }
            else
            {
                const Shape & kernel = param.kernel();
                assert(kernel.size() == 1 || kernel.size() == 2);
                _kernelY = kernel[0];
                _kernelX = kernel.size() > 1 ? kernel[1] : kernel[0];
                assert(_kernelY > 0 && _kernelX > 0);
            }

            const Shape & stride = param.stride();
            if (stride.empty())
            {
                _strideY = 1;
                _strideX = 1;
            }
            else
            {
                assert(stride.size() == 1 || stride.size() == 2);
                _strideY = stride[0];
                _strideX = stride.size() > 1 ? stride[1] : stride[0];
            }

            const Shape & pad = param.pad();
            if (param.padType() == PoolingPadTypeTensorflowSame)
            {
                if (_strideX == 2 && _strideY == 2)
                {
                    if (_kernelX == 3 && _kernelY == 3)
                    {
                        _padY = _srcH%_strideY;
                        _padX = _srcW%_strideX;
                        _padH = 1;
                        _padW = 1;
                    }
                    else if (_kernelX == 2 && _kernelY == 2)
                    {
                        _padY = 0;
                        _padX = 0;
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
                _padY = 0;
                _padX = 0;
                _padH = 0;
                _padW = 0;
            }
            else if (pad.size() == 1)
            {
                _padY = pad[0];
                _padX = pad[0];
                _padH = pad[0];
                _padW = pad[0];
            }
            else if (pad.size() == 2)
            {
                _padY = pad[0];
                _padX = pad[1];
                _padH = pad[0];
                _padW = pad[1];
            }
            else if (pad.size() == 4)
            {
                _padY = pad[0];
                _padX = pad[1];
                _padH = pad[2];
                _padW = pad[3];
            }
            else
                assert(0);
            //assert(_padY + _padH < _kernelY && _padX + _padW < _kernelX);

            if (_yoloCompatible == 2)
            {
                _dstH = (_srcH + _padH - _kernelY) / _strideY + 1;
                _dstW = (_srcW + _padW - _kernelX) / _strideX + 1;
            }
            else if (_yoloCompatible == 1)
            {
                _dstH = (_srcH + _padY + _padH) / _strideY;
                _dstW = (_srcW + _padX + _padW) / _strideX;
            }
            else
            {
                if (_roundingType == RoundingTypeCeil)
                {
                    _dstH = (size_t)(::ceil((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
                    _dstW = (size_t)(::ceil((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
                }
                else if (_roundingType == RoundingTypeFloor)
                {
                    _dstH = (size_t)(::floor((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
                    _dstW = (size_t)(::floor((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
                }
                else
                    assert(0);
                if (_padX || _padY)
                {
                    if ((_dstH - 1) * _strideY >= _srcH + _padY)
                        --_dstH;
                    if ((_dstW - 1) * _strideX >= _srcW + _padX)
                        --_dstW;
                    assert((_dstH - 1) * _strideY < _srcH + _padY);
                    assert((_dstW - 1) * _strideX < _srcW + _padX);
                }
            }

            _skip = _kernelX == 1 && _kernelY == 1 && _strideY == 1 && _strideX == 1 && _padY == 0 && _padX == 0 && _padH == 0 && _padW == 0;

            if (_skip)
                dst[0]->Share(*src[0]);
            else
            {
                assert(_type == TensorType32f || _method == PoolingMethodTypeMax);

                TensorFormat format = src[0]->Format();
                Shape shape = _trans ? Shape({ _num, _dstH, _dstW , _channels }) : Shape({ _num, _channels, _dstH, _dstW });
                switch (_type)
                {
                case TensorType32f: dst[0]->As32f().Reshape(shape, format); break;
                case TensorType8u: dst[0]->As8u().Reshape(shape, format); break;
                case TensorType8i: dst[0]->As8i().Reshape(shape, format); break;
                default:
                    assert(0);
                }
                std::stringstream desc;
                desc << "i=" << _num << "x" << _channels << "x" << _srcH << "x" << _srcW;
                desc << " k=" << _kernelY << "x" << _kernelX << " s=" << _strideY << "x" << _strideX << (_method ? " avg" : " max");
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
                for (size_t n = 0; n < _num; ++n)
                {
                    Detail::PoolingForwardCpuMax(src, _channels, _srcH, _srcW, _kernelY, _kernelX, 
                        _strideY, _strideX, _padY, _padX, dst, _dstH, _dstW, _format);
                    src += _channels*_srcW * _srcH;
                    dst += _channels*_dstW * _dstH;
                }
                break;
            case PoolingMethodTypeAverage:
                for (size_t n = 0; n < _num; ++n)
                {
                    Detail::PoolingForwardCpuAverage(src, _channels, _srcH, _srcW, _kernelY, _kernelX, 
                        _strideY, _strideX, _padY, _padX, dst, _dstH, _dstW, _excludePad, _format);
                    src += _channels*_srcW * _srcH;
                    dst += _channels*_dstW * _dstH;
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
        bool _skip;
        int _yoloCompatible, _trans, _excludePad;
        size_t _num, _channels, _srcH, _srcW, _kernelY, _kernelX, _strideX, _strideY, _padX, _padY, _padW, _padH, _dstH, _dstW;
    };
}
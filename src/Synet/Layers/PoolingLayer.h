/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
            size_t strideY, size_t strideX, size_t padY, size_t padX, T * dst, size_t dstH, size_t dstW, int trans)
        {
            if (trans)
            {
                for (size_t ph = 0; ph < dstH; ++ph)
                {
                    size_t hStart = ph * strideY - padY;
                    size_t hEnd = std::min(hStart + kernelY, srcH);
                    hStart = std::max<ptrdiff_t>(0, hStart);
                    for (size_t pw = 0; pw < dstW; ++pw)
                    {
                        size_t wStart = pw * strideX - padX;
                        size_t wEnd = std::min(wStart + kernelX, srcW);
                        wStart = std::max<ptrdiff_t>(0, wStart);
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = T(-FLT_MAX);
                        for (size_t h = hStart; h < hEnd; ++h)
                        {
                            for (size_t w = wStart; w < wEnd; ++w)
                            {
                                const T * pc = src + (h * srcW + w)*channels;
                                for (size_t c = 0; c < channels; ++c)
                                    dst[c] = std::max(dst[c], pc[c]);
                            }
                        }
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = std::min(hStart + kernelY, srcH);
                        hStart = std::max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = std::min(wStart + kernelX, srcW);
                            wStart = std::max<ptrdiff_t>(0, wStart);
                            T max = T(-FLT_MAX);
                            for (size_t h = hStart; h < hEnd; ++h)
                                for (size_t w = wStart; w < wEnd; ++w)
                                    max = std::max(max, src[h * srcW + w]);
                            dst[ph*dstW + pw] = max;
                        }
                    }
                    src += srcW * srcH;
                    dst += dstW * dstH;
                }            
            }
        }

        template <class T> void PoolingForwardCpuAverage(const T * src, size_t channels, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, T * dst, size_t dstH, size_t dstW, int trans)
        {
            if (trans)
            {
                for (size_t ph = 0; ph < dstH; ++ph)
                {
                    size_t hStart = ph * strideY - padY;
                    size_t hEnd = std::min(hStart + kernelY, srcH);
                    hStart = std::max<ptrdiff_t>(0, hStart);
                    for (size_t pw = 0; pw < dstW; ++pw)
                    {
                        size_t wStart = pw * strideX - padX;
                        size_t wEnd = std::min(wStart + kernelX, srcW);
                        wStart = std::max<ptrdiff_t>(0, wStart);
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = T(0);
                        for (size_t h = hStart; h < hEnd; ++h)
                        {
                            for (size_t w = wStart; w < wEnd; ++w)
                            {
                                const T * pc = src + (h * srcW + w)*channels;
                                for (size_t c = 0; c < channels; ++c)
                                    dst[c] += pc[c];
                            }
                        }
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = dst[c] / (hEnd - hStart) / (wEnd - wStart);
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = std::min(hStart + kernelY, srcH);
                        hStart = std::max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = std::min(wStart + kernelX, srcW);
                            wStart = std::max<ptrdiff_t>(0, wStart);
                            T sum = T(0);
                            for (size_t h = hStart; h < hEnd; ++h)
                                for (size_t w = wStart; w < wEnd; ++w)
                                    sum += src[h * srcW + w];
                            dst[ph*dstW + pw] = sum/(hEnd - hStart)/(wEnd - wStart);
                        }
                    }
                    src += srcW * srcH;
                    dst += dstW * dstH;
                }
            }
        }

        template <class T> void PoolingForwardMaxCpu(const T * src, size_t srcStride, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX, 
            size_t padY, size_t padX, size_t strideY, size_t strideX, T * dst, size_t dstX, size_t dstY)
        {
            for (size_t dy = 0; dy < dstY; ++dy)
            {
                size_t yStart = dy * strideY - padY;
                size_t yEnd = std::min(yStart + kernelY, srcY);
                yStart = std::max<ptrdiff_t>(0, yStart);
                for (size_t dx = 0; dx < dstX; ++dx)
                {
                    size_t xStart = dx * strideX - padX;
                    size_t xEnd = std::min(xStart + kernelX, srcX);
                    xStart = std::max<ptrdiff_t>(0, xStart);
                    T max = -std::numeric_limits<T>::max();
                    for (size_t sy = yStart; sy < yEnd; ++sy)
                        for (size_t sx = xStart; sx < xEnd; ++sx)
                            max = std::max(max, src[sy * srcStride + sx]);
                    dst[dy*dstX + dx] = max;
                }
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void PoolingForwardMaxCpu<float>(const float * src, size_t srcStride, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX,
            size_t padY, size_t padX, size_t strideY, size_t strideX, float * dst, size_t dstX, size_t dstY)
        {
            if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && padY == 1 && padX == 1)
            {
                ::SimdNeuralPooling1x1Max3x3(src, srcStride, srcX, srcY, dst, dstX);
                return;
            }
            if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0)
            {
                ::SimdNeuralPooling2x2Max3x3(src, srcStride, srcX, srcY, dst, dstX);
                return;
            }
            if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0)
            {
                ::SimdNeuralPooling2x2Max2x2(src, srcStride, srcX, srcY, dst, dstX);
                return;
            }
            for (size_t dy = 0; dy < dstY; ++dy)
            {
                size_t yStart = dy * strideY - padY;
                size_t yEnd = std::min(yStart + kernelY, srcY);
                yStart = std::max<ptrdiff_t>(0, yStart);
                for (size_t dx = 0; dx < dstX; ++dx)
                {
                    size_t xStart = dx * strideX - padX;
                    size_t xEnd = std::min(xStart + kernelX, srcX);
                    xStart = std::max<ptrdiff_t>(0, xStart);
                    float max = -std::numeric_limits<float>::max();
                    for (size_t sy = yStart; sy < yEnd; ++sy)
                        for (size_t sx = xStart; sx < xEnd; ++sx)
                            max = std::max(max, src[sy * srcStride + sx]);
                    dst[dy*dstX + dx] = max;
                }
            }
        }
#endif
    }

    template <class T> class PoolingLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PoolingLayer(const LayerParam & param)
            : Base(param)
        {
        }
        
        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PoolingParam & param = this->Param().pooling();
            _method = param.method();
            _yoloCompatible = param.yoloCompatible();
            _roundingType = param.roundingType();
            assert(src[0]->Count() == 4);

            _trans = src[0]->Format() == TensorFormatNhwc;

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
            assert(_padY + _padH < _kernelY && _padX + _padW < _kernelX);

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

            if(_trans)
                dst[0]->Reshape(Shape({ _num, _dstH, _dstW , _channels}), Type(), TensorFormatNhwc);
            else
                dst[0]->Reshape(Shape({ _num, _channels, _dstH, _dstW }), Type(), TensorFormatNchw);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            size_t dstSize = dst[0]->Size();
            switch (_method)
            {
            case PoolingMethodTypeMax:
                for (size_t n = 0; n < _num; ++n)
                {
                    if (_trans)
                    {
                        Detail::PoolingForwardCpuMax(pSrc, _channels, _srcH, _srcW, _kernelY, _kernelX, _strideY, _strideX, _padY, _padX, pDst, _dstH, _dstW, _trans);
                        pSrc += _channels*_srcW * _srcH;
                        pDst += _channels*_srcW * _srcH;
                    }
                    else
                    {
                        for (size_t c = 0; c < _channels; ++c)
                        {
                            size_t srcW = _srcW, srcH = _srcH;
                            if (_yoloCompatible == 1)
                            {
                                srcH = _dstH*_strideY - _padY - _padH;
                                srcW = _dstW*_strideX - _padX - _padW;
                            }
                            Detail::PoolingForwardMaxCpu(pSrc, _srcW, srcW, srcH, _kernelY, _kernelX, _padY, _padX, _strideY, _strideX, pDst, _dstW, _dstH);
                            pSrc += _srcW * _srcH;
                            pDst += _dstW * _dstH;
                        }
                    }
                }
                break;
            case PoolingMethodTypeAverage:
                for (size_t n = 0; n < _num; ++n)
                {
                    Detail::PoolingForwardCpuAverage(pSrc, _channels, _srcH, _srcW, _kernelY, _kernelX, _strideY, _strideX, _padY, _padX, pDst, _dstH, _dstW, _trans);
                    pSrc += _channels*_srcW * _srcH;
                    pDst += _channels*_srcW * _srcH;
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
        PoolingMethodType _method;
        RoundingType _roundingType;
        int _yoloCompatible, _trans;
        size_t _num, _channels, _srcH, _srcW, _kernelY, _kernelX, _strideX, _strideY, _padX, _padY, _padW, _padH, _dstH, _dstW;
    };
}
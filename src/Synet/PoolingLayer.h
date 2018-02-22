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
#include "Synet/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void PoolingForwardMaxCpu(const T * src, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX, 
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
                            max = std::max(max, src[sy * srcX + sx]);
                    dst[dy*dstX + dx] = max;
                }
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void PoolingForwardMaxCpu<float>(const float * src, size_t srcX, size_t srcY, size_t kernelY, size_t kernelX,
            size_t padY, size_t padX, size_t strideY, size_t strideX, float * dst, size_t dstX, size_t dstY)
        {
            if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && padY == 1 && padX == 1)
            {
                ::SimdNeuralPooling1x1Max3x3(src, srcX, srcX, srcY, dst, dstX);
                return;
            }
            if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0)
            {
                ::SimdNeuralPooling2x2Max3x3(src, srcX, srcX, srcY, dst, dstX);
                return;
            }
            if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0)
            {
                ::SimdNeuralPooling2x2Max2x2(src, srcX, srcX, srcY, dst, dstX);
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
                            max = std::max(max, src[sy * srcX + sx]);
                    dst[dy*dstX + dx] = max;
                }
            }
        }
#endif
    }

    template <class T, template<class> class A> class PoolingLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PoolingLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst)
        {
            _method = this->Param().pooling().method();
        }        
        
        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src[0]->Count() == 4);
            _channels = src[0]->Axis(1);
            _srcY = src[0]->Axis(2);
            _srcX = src[0]->Axis(3);

            if (this->Param().pooling().globalPooling())
            {
                _kernelY = src[0]->Axis(2);
                _kernelX = src[0]->Axis(3);
            }
            else
            {
                const Shape & kernel = this->Param().pooling().kernel();
                assert(kernel.size() == 1 || kernel.size() == 2);
                _kernelY = kernel[0];
                _kernelX = kernel.size() > 1 ? kernel[1] : kernel[0];
                assert(_kernelY > 0 && _kernelX > 0);
            }

            const Shape & pad = this->Param().pooling().pad();
            if (pad.empty())
            {
                _padY = 0;
                _padX = 0;
            }
            else
            {
                assert(pad.size() == 1 || pad.size() == 2);
                _padY = pad[0];
                _padX = pad.size() > 1 ? pad[1] : pad[0];
                assert(_padY < _kernelY && _padX < _kernelX);
            }

            const Shape & stride = this->Param().pooling().stride();
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

            _dstX = (size_t)(::ceil((float)(_srcX + 2 * _padX - _kernelX) / _strideX)) + 1;
            _dstY = (size_t)(::ceil((float)(_srcY + 2 * _padY - _kernelY) / _strideY)) + 1;
            if (_padX || _padY)
            {
                if ((_dstX - 1) * _strideX >= _srcX + _padX)
                    --_dstX;
                if ((_dstY - 1) * _strideY >= _srcY + _padY)
                    --_dstY;
                assert((_dstX - 1) * _strideX < _srcX + _padX);
                assert((_dstY - 1) * _strideY < _srcY + _padY);
            }
            dst[0]->Reshape(Shape({ src[0]->Axis(0), _channels, _dstY, _dstX }));
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pSrc = src[0]->Data();
            Type * pDst = dst[0]->Data();
            size_t num = dst[0]->Axis(0);
            size_t dstSize = dst[0]->Size();
            switch (_method)
            {
            case PoolingMethodTypeMax:
                CpuSet(dstSize, Type(-FLT_MAX), pDst);
                for (size_t n = 0; n < num; ++n)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        Detail::PoolingForwardMaxCpu(pSrc, _srcX, _srcY, _kernelY, _kernelX, _padY, _padX, _strideY, _strideX, pDst, _dstX, _dstY);
                        pSrc += _srcX * _srcY;
                        pDst += _dstX * _dstY;
                    }
                }
                break;
            case PoolingMethodTypeAverage:
                CpuSet(dstSize, Type(0), pDst);
                for (size_t n = 0; n < num; ++n)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        for (size_t ph = 0; ph < _dstY; ++ph)
                        {
                            size_t hStart = ph * _strideY - _padY;
                            size_t hEnd = std::min(hStart + _kernelY, _srcY);
                            hStart = std::max<ptrdiff_t>(0, hStart);
                            for (size_t pw = 0; pw < _dstX; ++pw)
                            {
                                size_t wStart = pw * _strideX - _padX;
                                size_t wEnd = std::min(wStart + _kernelX, _srcX);
                                wStart = std::max<ptrdiff_t>(0, wStart);
                                size_t poolSize = (hEnd - hStart) * (wEnd - wStart);
                                Type sum = 0;
                                for (size_t h = hStart; h < hEnd; ++h)
                                    for (size_t w = wStart; w < wEnd; ++w)
                                        sum += pSrc[h * _srcX + w];
                                pDst[ph*_dstX + pw] = sum / poolSize;
                            }
                        }
                        pSrc += _srcX * _srcY;
                        pDst += _dstX * _dstY;
                    }
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
        size_t _channels, _srcX, _srcY, _kernelX, _kernelY, _dstX, _dstY, _strideX, _strideY, _padX, _padY;
    };
}
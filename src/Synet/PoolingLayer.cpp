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

#include "Synet/PoolingLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> void PoolingLayer<T, A>::Setup(const PoolingLayer::TensorPtrs & src, const PoolingLayer::TensorPtrs & dst)
    {
    }

    template <class T, template<class> class A> void PoolingLayer<T, A>::Reshape(const PoolingLayer::TensorPtrs & src, const PoolingLayer::TensorPtrs & dst)
    {
        assert(src[0]->Count() == 4);
        _channels = src[0]->Axis(1);
        _srcX = src[0]->Axis(3);
        _srcY = src[0]->Axis(2);
        if (_param.globalPooling) 
        {
            _kernelX = src[0]->Axis(3);
            _kernelY = src[0]->Axis(2);
        }
        else
        {
            _kernelX = _param.kernelX;
            _kernelY = _param.kernelY;
        }
        _dstX = (size_t)(::ceil((float)(_srcX + 2 * _param.padX - _kernelX) / _param.strideX)) + 1;
        _dstY = (size_t)(::ceil((float)(_srcY + 2 * _param.padY - _kernelY) / _param.strideY)) + 1;
        if (_param.padX || _param.padY) 
        {
            if ((_dstX - 1) * _param.strideX >= _srcX + _param.padX)
                --_dstX;
            if ((_dstY - 1) * _param.strideY >= _srcY + _param.padY)
                --_dstY;
            assert((_dstX - 1) * _param.strideX < _srcX + _param.padX);
            assert((_dstY - 1) * _param.strideY < _srcY + _param.padY);
        }
        dst[0]->Reshape(Shape({ src[0]->Axis(0), _channels, _dstY, _dstX }));
    }

    template <class T, template<class> class A> void PoolingLayer<T, A>::ForwardCpu(const PoolingLayer::TensorPtrs & src, const PoolingLayer::TensorPtrs & dst)
    {
        const Type * pSrc = src[0]->Data();
        Type * pDst = dst[0]->Data();
        size_t num = dst[0]->Axis(0);
        size_t dstSize = dst[0]->Size();
        switch (_param.method) 
        {
        case PoolingLayerParam::MethodMax:
            CpuSet(dstSize, Type(-FLT_MAX), pDst);
            for (size_t n = 0; n < num; ++n)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t ph = 0; ph < _dstY; ++ph) 
                    {
                        size_t hStart = std::max(size_t(0), ph * _param.strideY - _param.padY);
                        size_t hEnd = std::min(hStart + _kernelY, _srcY);
                        for (size_t pw = 0; pw < _dstX; ++pw) 
                        {
                            size_t wStart = std::max(size_t(0), pw * _param.strideX - _param.padX);
                            size_t wEnd = std::min(wStart + _kernelX, _srcX);
                            Type max = -FLT_MAX;
                            for (size_t h = hStart; h < hEnd; ++h)
                                for (size_t w = wStart; w < wEnd; ++w)
                                    max = std::max(max, pSrc[h * _srcX + w]);
                            pDst[ph*_dstX + pw] = max;
                        }
                    }
                    pSrc += _srcX * _srcY;
                    pDst += _dstX * _dstY;
                }
            }
            break;
        case PoolingLayerParam::MethodAverage:
            CpuSet(dstSize, Type(0), pDst);
            for (size_t n = 0; n < num; ++n)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t ph = 0; ph < _dstY; ++ph)
                    {
                        size_t hStart = std::max(size_t(0), ph * _param.strideY - _param.padY);
                        size_t hEnd = std::min(hStart + _kernelY, _srcY);
                        for (size_t pw = 0; pw < _dstX; ++pw)
                        {
                            size_t wStart = std::max(size_t(0), pw * _param.strideX - _param.padX);
                            size_t wEnd = std::min(wStart + _kernelX, _srcX);
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
        case PoolingLayerParam::MethodStochastic:
            assert(0);
            break;
        default:
            assert(0);
        }
    }

    SYNET_CLASS_INSTANCE(PoolingLayer);
}
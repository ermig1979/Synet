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
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
    }

    template <class T> class PermuteLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PermuteLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PermuteParam & param = this->Param().permute();
            _permute = false;
            _order = param.order();
            _count = _order.size();
            assert(_count >= 2 && _count <= 5);
            size_t is = 0, os = 0;
            Shape permute;
            for (size_t i = 0; i < _order.size(); ++i)
            {
                if (_order[i] != i)
                    permute.push_back(i);
                is += i;
                os += _order[i];
            }
            assert(is == os);
            _permute = permute.size() > 1;
            if (permute.size() == 2 && permute[0] + 1 == permute[1])
            {
                if (src[0]->Axis(permute[0]) == 1 || src[0]->Axis(permute[1]) == 1)
                    _permute = false;
            }

            if (_permute)
            {
                _srcShape = src[0]->Shape();
                assert(_srcShape.size() == _count);
                _dstShape.clear();
                for (size_t i = 0; i < _count; ++i)
                    _dstShape.push_back(_srcShape[_order[i]]);
                _srcStride.resize(_count, 1);
                _dstStride.resize(_count, 1);
                Shape dstStride(_count, 1);
                for (ptrdiff_t i = _count - 2; i >= 0; i--)
                {
                    _srcStride[i] = _srcStride[i + 1] * _srcShape[i + 1];
                    dstStride[i] = dstStride[i + 1] * _dstShape[i + 1];
                }
                for (size_t i = 0; i < _count; ++i)
                    _dstStride[_order[i]] = dstStride[i];
                dst[0]->Reshape(_dstShape, param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
                this->UsePerfStat();
            }
            else
                dst[0]->ShareAs(*src[0], src[0]->Shape(), param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_permute)
            {
                const Type * pSrc = src[0]->CpuData();
                Type * pDst = dst[0]->CpuData();
                switch (_count)
                {
                case 2:
                    for (size_t i = 0; i < _srcShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _srcShape[1]; ++j)
                        {
                            size_t srcOffset = i*_srcStride[0] + j*_srcStride[1];
                            size_t dstOffset = i*_dstStride[0] + j*_dstStride[1];
                            pDst[dstOffset] = pSrc[srcOffset];
                        }
                    }
                    break;
                case 3:
                    for (size_t i = 0; i < _srcShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _srcShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _srcShape[2]; ++k)
                            {
                                size_t srcOffset = i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2];
                                size_t dstOffset = i*_dstStride[0] + j*_dstStride[1] + k*_dstStride[2];
                                pDst[dstOffset] = pSrc[srcOffset];
                            }
                        }
                    }
                    break;
                case 4:
                    for (size_t i = 0; i < _srcShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _srcShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _srcShape[2]; ++k)
                            {
                                for (size_t l = 0; l < _srcShape[3]; ++l)
                                {
                                    size_t srcOffset = i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2] + l*_srcStride[3];
                                    size_t dstOffset = i*_dstStride[0] + j*_dstStride[1] + k*_dstStride[2] + l*_dstStride[3];
                                    pDst[dstOffset] = pSrc[srcOffset];
                                }
                            }
                        }
                    }
                    break;
                case 5:
                    for (size_t i = 0; i < _srcShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _srcShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _srcShape[2]; ++k)
                            {
                                for (size_t l = 0; l < _srcShape[3]; ++l)
                                {
                                    for (size_t m = 0; m < _srcShape[4]; ++m)
                                    {
                                        size_t srcOffset = i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2] + l*_srcStride[3] + m*_srcStride[4];
                                        size_t dstOffset = i*_dstStride[0] + j*_dstStride[1] + k*_dstStride[2] + l*_dstStride[3] + m*_dstStride[4];
                                        pDst[dstOffset] = pSrc[srcOffset];
                                    }
                                }
                            }
                        }
                    }
                    break;
                default:
                    assert(0);
                }
            }
        }

    private:
        bool _permute;
        size_t _count;
        Shape _order, _srcShape, _dstShape, _srcStride, _dstStride;
    };
}
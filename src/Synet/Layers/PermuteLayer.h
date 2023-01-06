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
    }

    template <class T> class PermuteLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PermuteLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PermuteParam & param = this->Param().permute();
            _permute = false;
            _dstOrder = param.order();
            _count = _dstOrder.size();
            assert(_count >= 2 && _count <= 5);
            _srcOrder.resize(_count);
            size_t is = 0, os = 0;
            Shape permute;
            for (size_t i = 0; i < _dstOrder.size(); ++i)
            {
                if (_dstOrder[i] != i)
                    permute.push_back(i);
                is += i;
                os += _dstOrder[i];
                _srcOrder[_dstOrder[i]] = i;
            }
            assert(is == os);
            _permute = permute.size() > 1;
            if (permute.size() == 2 && permute[0] + 1 == permute[1])
            {
                if (src[0]->Axis(permute[0]) == 1 || src[0]->Axis(permute[1]) == 1)
                    _permute = false;
            }

            _srcShape = src[0]->Shape();
            assert(_srcShape.size() == _count);
            _dstShape.clear();
            for (size_t i = 0; i < _count; ++i)
                _dstShape.push_back(_srcShape[_dstOrder[i]]);
            if (_permute)
            {
                dst[0]->Reshape(_dstShape, param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
                CompactShapes();
                _srcStride = Stride(_srcShape, _srcOrder);
                _dstStride = Stride(_dstShape, _dstOrder);
                this->UsePerfStat();
            }
            else
                dst[0]->ShareAs(*src[0], _dstShape, param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
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
                    for (size_t i = 0; i < _dstShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _dstShape[1]; ++j)
                        {
                            *pDst++ = pSrc[i * _srcStride[0] + j * _srcStride[1]];
                        }
                    }
                    break;
                case 3:
                    for (size_t i = 0; i < _dstShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _dstShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _dstShape[2]; ++k)
                            {
                                *pDst++= pSrc[i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2]];
                            }
                        }
                    }
                    break;
                case 4:
                    for (size_t i = 0; i < _dstShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _dstShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _dstShape[2]; ++k)
                            {
                                for (size_t l = 0; l < _dstShape[3]; ++l)
                                {
                                    *pDst++ = pSrc[i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2] + l*_srcStride[3]];
                                }
                            }
                        }
                    }
                    break;
                case 5:
                    for (size_t i = 0; i < _dstShape[0]; ++i)
                    {
                        for (size_t j = 0; j < _dstShape[1]; ++j)
                        {
                            for (size_t k = 0; k < _dstShape[2]; ++k)
                            {
                                for (size_t l = 0; l < _dstShape[3]; ++l)
                                {
                                    for (size_t m = 0; m < _dstShape[4]; ++m)
                                    {
                                        *pDst++ = pSrc[i*_srcStride[0] + j*_srcStride[1] + k*_srcStride[2] + l*_srcStride[3] + m*_srcStride[4]];
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
        Shape _srcOrder, _dstOrder, _srcShape, _dstShape, _srcStride, _dstStride;

        static Shape Stride(const Shape & shape, const Shape& order)
        {
            Shape buf(shape.size(), 1), out(shape.size(), 1);
            for (ptrdiff_t i = shape.size() - 2; i >= 0; i--)
                buf[i] = buf[i + 1] * shape[i + 1];
            for (size_t i = 0; i < shape.size(); ++i)
               out[order[i]] = buf[i];
            return out;
        }

        static Shape CompactOrder(Shape order)
        {
            for (size_t i = 1; i < order.size(); ++i)
            {
                if (order[i] == order[i - 1] + 1)
                {
                    order.erase(order.begin() + i);
                    for (size_t j = 0; j < order.size(); ++j)
                        if (order[j] > order[i - 1])
                            order[j]--;
                }
            }
            return order;
        }

        static Shape CompactShape(const Shape& shape, const Shape& order)
        {
            Shape compact;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                size_t dim = shape[order[i]];
                if (i && order[i] == order[i - 1] + 1)
                    compact.back() *= dim;
                else
                    compact.push_back(dim);
            }
            return compact;
        }

        void CompactShapes()
        {
            size_t count = 0;
            for (size_t i = 0; i < _count; ++i)
                if (i == 0 || _dstOrder[i] != _dstOrder[i - 1] + 1)
                    count++;
            if (count == _count)
                return;
            _count = count;
            Shape dstShape = _dstShape;
            _dstShape = CompactShape(_srcShape, _dstOrder);
            _srcShape = CompactShape(dstShape, _srcOrder);
            _dstOrder = CompactOrder(_dstOrder);
            _srcOrder = CompactOrder(_srcOrder);
        }
    };
}
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
        template <typename T> void BiasLayerForwardCpu(const T * src, const T * bias, size_t count, size_t size, T * dst, int trans)
        {
            if (trans)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    for (size_t i = 0; i < count; ++i)
                        dst[i] = src[i] + bias[i];
                    src += count;
                    dst += count;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    const T b = bias[i];
                    for (size_t j = 0; j < size; ++j)
                        dst[j] = src[j]  + b;
                    src += size;
                    dst += size;
                }
            }
        }
    }

    template <class T> class BiasLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        BiasLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ScaleParam & param = this->Param().scale();
            _axis = param.axis();
            const Tensor & bias = (src.size() > 1 ? *src[1] : this->Weight()[0]);
            _trans = src[0]->Format() == TensorFormatNhwc;
            _count = bias.Size();
            if (bias.Size() == src[0]->Size())
            {
                _num = 1;
                _size = 1;
            }
            else
            {
                _num = src[0]->Size(0, _axis);
                _size = src[0]->Size() / _num / _count;
            }
            assert(src[0]->Size() == _num*_count*_size);
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pSrc = src[0]->CpuData();
            const Type * pBias = (src.size() > 1 ? *src[1] : this->Weight()[0]).CpuData();
            Type * pDst = dst[0]->CpuData();
            for (size_t n = 0; n < _num; ++n)
            {
                Detail::BiasLayerForwardCpu(pSrc, pBias, _count, _size, pDst, _trans);
                pSrc += _count*_size;
                pDst += _count*_size;
            }
        }

    private:
        size_t _axis, _num, _count, _size;
        int _trans;
    };
}
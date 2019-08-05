/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
        template <typename T> void ScaleLayerForwardCpu(const T * src, const T * scale, const T * bias, size_t count, size_t size, T * dst, int trans)
        {
            if (trans)
            {
                if (bias)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        for (size_t i = 0; i < count; ++i)
                            dst[i] = src[i] * scale[i] + bias[i];
                        src += count;
                        dst += count;
                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        for (size_t i = 0; i < count; ++i)
                            dst[i] = src[i] * scale[i];
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    const T s = scale[i];
                    const T b = bias ? bias[i] : 0;
                    for (size_t j = 0; j < size; ++j)
                        dst[j] = src[j] * s + b;
                    src += size;
                    dst += size;
                }
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void ScaleLayerForwardCpu<float>(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, int trans)
        {
            ::SimdSynetScaleLayerForward(src, scale, bias, count, size, dst, (::SimdTensorFormatType)trans);
        }
#endif
    }

    template <class T> class ScaleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        ScaleLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ScaleParam & param = this->Param().scale();
            _axis = param.axis();
            _biasTerm = param.biasTerm();
            assert(this->Weight().size());
            if (_biasTerm)
            {
                assert(this->Weight().size() > 1);
                assert(this->Weight()[0].Shape() == this->Weight()[1].Shape());
            }
            const Tensor & scale = this->Weight()[0];
            _count = scale.Size();
            _trans = src[0]->Format() == TensorFormatNhwc;
            if (scale.Size() == src[0]->Size())
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
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            const Type* pSrc = src[0]->CpuData();
            const Type * pScale = this->Weight()[0].CpuData();
            const Type * pBias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            Type * pDst = dst[0]->CpuData();
            for (size_t n = 0; n < _num; ++n)
            {
                Detail::ScaleLayerForwardCpu(pSrc, pScale, pBias, _count, _size, pDst, _trans);
                pSrc += _count*_size;
                pDst += _count*_size;
            }
        }

    private:
        size_t _axis, _num, _count, _size;
        int _trans;
        bool _biasTerm;
    };
}
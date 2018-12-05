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
        template <typename T> void ScaleLayerForwardCpu(const T * src, const T * scale, const T * bias, size_t count, size_t size, T * dst)
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

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void ScaleLayerForwardCpu<float>(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            ::SimdSynetScaleLayerForward(src, scale, bias, count, size, dst, ::SimdFalse);
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

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
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
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Tensor & scale = this->Weight()[0];
            if (scale.Size() == src[0]->Size())
            {
                _outerDim = 1;
                _scaleDim = scale.Size();
                _innerDim = 1;
            }
            else
            {
                assert(src[0]->Count() >= _axis + scale.Count());
                for (size_t i = 0; i < scale.Count(); ++i)
                    assert(src[0]->Axis(_axis + i) == scale.Axis(i));
                _outerDim = src[0]->Size(0, _axis);
                _scaleDim = scale.Size();
                _innerDim = src[0]->Size(_axis + scale.Count());
            }
            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            const Type* pSrc = src[0]->CpuData();
            const Type * pScale = this->Weight()[0].CpuData();
            const Type * pBias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            Type * pDst = dst[0]->CpuData();
            for (size_t n = 0; n < _outerDim; ++n)
            {
                Detail::ScaleLayerForwardCpu(pSrc, pScale, pBias, _scaleDim, _innerDim, pDst);
                pSrc += _scaleDim*_innerDim;
                pDst += _scaleDim*_innerDim;
            }
        }

    private:
        size_t _axis, _outerDim, _scaleDim, _innerDim;
        bool _biasTerm;
    };
}
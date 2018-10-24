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
        template <class T> void InnerProductLayerForwardCpu(const T * src, const T * weight, const T * bias, size_t count, size_t size, T * dst)
        {
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = CpuDotProduct(src, weight + size*i, size) + bias[i];
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = CpuDotProduct(src, weight + size*i, size);
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void InnerProductLayerForwardCpu<float>(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            ::SimdSynetInnerProductLayerForward(src, weight, bias, count, size, dst);
        }
#endif
    }

    template <class T> class InnerProductLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        InnerProductLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _biasTerm = this->Param().innerProduct().biasTerm();
            _transposeA = this->Param().innerProduct().transposeA();
            _transposeB = this->Param().innerProduct().transposeB();
            _axis = this->Param().innerProduct().axis();
            _K = src[0]->Size(_axis);
            if (src.size() == 2)
            {
                assert(_biasTerm == false);
                assert(_K = src[1]->Size(0, _axis));
                _N = src[1]->Axis(_axis);
            }
            else
            {
                _N = this->Param().innerProduct().outputNum();
                const typename Base::Tensors & weight = this->Weight();
                if (_biasTerm)
                    assert(weight.size() == 2);
                else
                    assert(weight.size() == 1);
                if (_transposeB)
                    assert(weight[0].Shape() == Shape({ _K, _N }));
                else
                    assert(weight[0].Shape() == Shape({ _N, _K }));
                if (_biasTerm)
                    assert(weight[1].Shape() == Shape({ _N }));
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _M = src[0]->Size(0, _axis);
            Shape dstShape = src[0]->Shape();
            dstShape.resize(_axis + 1);
            dstShape[_axis] = _N;
            dst[0]->Reshape(dstShape);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            ForwardCpu(src[0]->CpuData(), src.size() > 1 ? src[1]->CpuData() : this->Weight()[0].CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const T * a, const T * b, T * c)
        {
#ifdef SYNET_SIZE_STATISTIC
            std::stringstream ss;
            ss << " M=" << _M << " N=" << _N << " K=" << _K;
            SYNET_PERF_BLOCK(ss.str().c_str());
#else
            SYNET_PERF_FUNC();
#endif
            if (_M == 1 && !_transposeB)
            {
                Detail::InnerProductLayerForwardCpu(a, b, _biasTerm ? this->Weight()[1].CpuData() : NULL, _N, _K, c);
            }
            else
            {
                CpuGemm<Type>(_transposeA ? CblasNoTrans : CblasTrans, _transposeB ? CblasNoTrans : CblasTrans, _M, _N, _K, Type(1), a, b, Type(0), c);
                if (_biasTerm)
                    CpuAddBias(this->Weight()[1].CpuData(), _N, _M, c);
            }
        }


    private:
        typedef typename Base::Tensor Tensor;

        size_t _M, _K, _N, _axis;
        bool _biasTerm, _transposeA, _transposeB;
    };
}
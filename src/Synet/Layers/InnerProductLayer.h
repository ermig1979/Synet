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

        virtual int64_t Flop() const
        {
            return _Mdim * _Ndim * _Kdim * 2;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _biasTerm = this->Param().innerProduct().biasTerm();
            _transposeA = this->Param().innerProduct().transposeA();
            _transposeB = this->Param().innerProduct().transposeB();
            _axis = this->Param().innerProduct().axis();
            _Kdim = src[0]->Size(_axis);
            if (src.size() == 2)
            {
                assert(_biasTerm == false);
                assert(_Kdim = src[1]->Size(0, _axis));
                _Ndim = src[1]->Axis(_axis);
            }
            else
            {
                _Ndim = this->Param().innerProduct().outputNum();
                const typename Base::Tensors & weight = this->Weight();
                if (_biasTerm)
                    assert(weight.size() == 2);
                else
                    assert(weight.size() == 1);
                if (_transposeB)
                    assert(weight[0].Shape() == Shape({ _Kdim, _Ndim }));
                else
                    assert(weight[0].Shape() == Shape({ _Ndim, _Kdim }));
                if (_biasTerm)
                    assert(weight[1].Shape() == Shape({ _Ndim }));
            }

            _Mdim = src[0]->Size(0, _axis);
            Shape dstShape = src[0]->Shape();
            dstShape.resize(_axis + 1);
            dstShape[_axis] = _Ndim;
            dst[0]->Reshape(dstShape, src[0]->Format());
            std::stringstream desc;
            desc << " M=" << _Mdim << " N=" << _Ndim << " K=" << _Kdim;
            this->UsePerfStat(desc.str(), Flop());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            ForwardCpu(src[0]->CpuData(), src.size() > 1 ? src[1]->CpuData() : this->Weight()[0].CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const T * a, const T * b, T * c)
        {
            if (!_transposeB && _Mdim == 1)
            {
                for (size_t i = 0; i < _Mdim; ++i)
                    Detail::InnerProductLayerForwardCpu(a + i*_Kdim, b, _biasTerm ? this->Weight()[1].CpuData() : NULL, _Ndim, _Kdim, c + i*_Ndim);
            }
            else
            {
                CpuGemm<Type>(_transposeA ? CblasTrans : CblasNoTrans, _transposeB ? CblasNoTrans : CblasTrans, _Mdim, _Ndim, _Kdim, Type(1), a, _Kdim, b, _Kdim, Type(0), c, _Ndim);
                if (_biasTerm)
                {
                    for(size_t i = 0; i < _Mdim; ++i)
                        CpuAddBias(this->Weight()[1].CpuData(), _Ndim, 1, c + i*_Ndim);
                }
            }
        }

    private:
        typedef typename Base::Tensor Tensor;

        size_t _Mdim, _Kdim, _Ndim, _axis;
        bool _biasTerm, _transposeA, _transposeB;
    };
}

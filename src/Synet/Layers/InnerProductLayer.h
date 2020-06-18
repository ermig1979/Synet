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

#ifdef _N
#undef _N
#endif

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

        InnerProductLayer(const LayerParam & param, QuantizationMethod method)
            : Base(param)
            , _method(method)
        {
        }

        virtual int64_t Flop() const
        {
            return _M * _N * _K * 2;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const InnerProductParam& param = this->Param().innerProduct();
            _is8i = param.quantizationLevel() == TensorType8i && src.size() == 1;
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
            _biasTerm = param.biasTerm();
            _transA = param.transposeA();
            _transB = param.transposeB();
            size_t axis = param.axis();
            _K = src[0]->Size(axis);
            if (src.size() == 2)
            {
                assert(_biasTerm == false);
                assert(_K = src[1]->Size(0, axis));
                _N = src[1]->Axis(axis);
            }
            else
            {
                _N = this->Param().innerProduct().outputNum();
                const typename Base::Tensors & weight = this->Weight();
                if (_biasTerm)
                    assert(weight.size() == 2);
                else
                    assert(weight.size() == 1);
                if (_transB)
                    assert(weight[0].Shape() == Shp(_K, _N));
                else
                    assert(weight[0].Shape() == Shp(_N, _K));
                if (_biasTerm)
                    assert(weight[1].Shape() == Shp(_N));
            }
            _M = src[0]->Size(0, axis);

            Shape dstShape = src[0]->Shape();
            dstShape.resize(axis + 1);
            dstShape[axis] = _N;
            if (_dst8u)
                dst[0]->As8u().Reshape(dstShape, src[0]->Format());
            else
                dst[0]->As32f().Reshape(dstShape, src[0]->Format());
            std::stringstream desc;
            desc << "M=" << _M << " N=" << _N << " K=" << _K;
            this->UsePerfStat(desc.str(), Flop());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_is8i)
            {
                uint8_t* tmp = _src8u ? src[0]->As8u().CpuData() : Base::Buf8u(buf, 0);
                int32_t* sum = Base::Buf32i(buf, 0);
                if (!_src8u)
                    _srcCvt.Convert(src[0]->As32f().CpuData(), tmp);
                //ForwardCpu(tmp, sum);
                if (_dst8u)
                    _dstCvt.Convert(sum, dst[0]->As8u().CpuData());
                else
                    _dstCvt.Convert(sum, dst[0]->As32f().CpuData());
            }
            else
                ForwardCpu(src[0]->CpuData(), src.size() > 1 ? src[1]->CpuData() : this->Weight()[0].CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const float * src, const float* wgt, float* dst)
        {
            const float* bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            if (!_transB && _M == 1)
                Detail::InnerProductLayerForwardCpu(src, wgt, bias, _N, _K, dst);
            else
            {
                CpuGemm(_transA ? CblasTrans : CblasNoTrans, _transB ? CblasNoTrans : CblasTrans, _M, _N, _K, 1.0f, src, _K, wgt, _K, 0.0f, dst, _N);
                if (_biasTerm)
                {
                    for(size_t i = 0; i < _M; ++i)
                        CpuAddBias(bias, _N, 1, dst + i*_N);
                }
            }
        }

    private:
        QuantizationMethod _method;
        size_t _M, _N, _K;
        bool _biasTerm, _transA, _transB, _src8u, _dst8u, _is8i;
        Converter _srcCvt, _dstCvt;
    };
}

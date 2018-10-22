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

namespace Synet
{
    namespace Detail
    {
        template <class T> void FusedLayerForwardCpu0(const T * src, const T * bias, const T * scale, size_t count, size_t size, T * dst)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const T b = bias[i];
                const T s = scale[i];
                for (size_t j = 0; j < size; ++j)
                {
                    T x = src[j] + b;
                    dst[j] = (x - ::abs(x))*s + std::max(T(0), x);
                }
                src += size;
                dst += size;
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void FusedLayerForwardCpu0<float>(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst)
        {
            ::SimdSynetFusedLayerForward0(src, bias, scale, count, size, dst);
        }
#endif
    }

    template <class T> class FusedLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        FusedLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _type = this->Param().fused().type();
            const Tensors & weight = this->Weight();
            switch (_type)
            {
            case 0:
            {
                assert(weight.size() == 3);
                _t0.bias.Share(weight[0]);
                _t0.count = _t0.bias.Size();
                _t0.scale.Reshape(_t0.bias.Shape());
                assert(weight[1].Size() == _t0.count || weight[1].Size() == 1);
                if (weight[1].Size() == _t0.count)
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] = weight[1].CpuData()[i];
                }
                else
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] = weight[1].CpuData()[0];
                }
                assert(weight[2].Size() == _t0.count || weight[2].Size() == 1);
                if (weight[2].Size() == _t0.count)
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] *= weight[2].CpuData()[i];
                }
                else
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] *= weight[2].CpuData()[0];
                }
                break;
            }
            default:
                assert(0);
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            switch (_type)
            {
            case 0:
            {
                _t0.size = src[0]->Size() / _t0.count;
                assert(_t0.count == src[0]->Axis(1));
                assert(_t0.size*_t0.count == src[0]->Size());
                dst[0]->Reshape(src[0]->Shape());
                break;
            }
            default: 
                assert(0);
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            const Type* pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            switch (_type)
            {
            case 0:
                Detail::FusedLayerForwardCpu0(pSrc, _t0.bias.CpuData(), _t0.scale.CpuData(), _t0.count, _t0.size, pDst);
                break;
            default:
                assert(0);
            }
        }

    private:
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        int _type;

        struct T0
        {
            size_t count, size;
            Tensor bias, scale;
        } _t0;
    };
}
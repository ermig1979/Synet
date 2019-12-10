/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar,
*               2019-2019 Artur Voronkov.
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
#include "Synet/Layers/ScaleLayer.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void EltwiseLayerForwardCpu(T const * const * src, const T * weight, size_t count, size_t size, EltwiseOperationType type, T * dst)
        {
            assert(count >= 2);
            switch(type)
            {
            case EltwiseOperationTypeProduct:
                CpuMul(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMul(dst, src[i], size, dst);
                break;
            case EltwiseOperationTypeSum:
                CpuScale(src[0], size, weight[0], dst);
                for (size_t i = 1; i < count; ++i)
                    CpuAxpy(src[i], size, weight[i], dst);
                break;
            case EltwiseOperationTypeMax:
                CpuMax(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMax(dst, src[i], size, dst);
                break;
            case EltwiseOperationTypeMin:
                CpuMin(src[0], src[1], size, dst);
                for (size_t i = 2; i < count; ++i)
                    CpuMin(dst, src[i], size, dst);
                break;
            default:
                assert(0);
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void EltwiseLayerForwardCpu<float>(float const * const * src, const float * weight, size_t count, size_t size, EltwiseOperationType type, float * dst)
        {
            ::SimdSynetEltwiseLayerForward(src, weight, count, size, (::SimdSynetEltwiseOperationType)type, dst);
        }
#endif
    }

    template <class T> class EltwiseLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        EltwiseLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const EltwiseParam & param = this->Param().eltwise();
            assert(param.coefficients().size() == 0 || param.coefficients().size() == src.size());
            assert(!(param.operation() == EltwiseOperationTypeProduct && param.coefficients().size()));
            _operation = param.operation();
            _coefficients.resize(src.size(), Type(1));
            if (param.coefficients().size())
            {
                for (size_t i = 0; i < src.size(); ++i)
                    _coefficients[i] = param.coefficients()[i];
            } 

            if (src.size() == 2 && src[0]->Shape() != src[1]->Shape())
            {
                if (_operation == EltwiseOperationTypeProduct && src[0]->Count() == 4)
                {
                    _scale = 1;
                    _trans = src[0]->Format() == TensorFormatNhwc;
                    _batch = src[0]->Axis(0);
                    _channels = src[0]->Axis(_trans ? 3 : 1);
                    _spatial = src[0]->Size() / _batch / _channels;
                    size_t size = src[1]->Size(1);
                    if (size == _channels)
                        _scale = 1;
                    else if (size == _spatial)
                        _scale = 2;
                    else
                        assert(0);
                }
                else
                    assert(0);
            }
            else
            {
                _scale = 0;
                _src.resize(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                {
                    assert(src[i]->Shape() == src[0]->Shape());
                    _src[i] = src[i]->CpuData();
                }
            }
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            if (_scale)
            {
                const Type * pSrc = src[0]->CpuData();
                const Type * pScale = src[1]->CpuData();
                const Type * pBias = NULL;
                Type * pDst = dst[0]->CpuData();
                for (size_t b = 0; b < _batch; ++b)
                {
                    if(_scale == 1)
                        Detail::ScaleLayerForwardCpu(pSrc, pScale, pBias, _channels, _spatial, pDst, _trans);
                    else
                        Detail::ScaleLayerForwardCpu(pSrc, pScale, pBias, _spatial, _channels, pDst, 1 - _trans);
                    pSrc += _channels*_spatial;
                    pDst += _channels*_spatial;
                    pScale += (_scale == 1 ? _channels : _spatial);
                }
            }
            else
            {
                Detail::EltwiseLayerForwardCpu(_src.data(), _coefficients.data(), _src.size(), dst[0]->Size(), _operation, dst[0]->CpuData());
            }
        }

    private:
        typedef std::vector<Type> Vector;
        typedef std::vector<Type*> Pointers;

        EltwiseOperationType _operation;
        Vector _coefficients;
        Pointers _src;
        int _scale, _trans;
        size_t _batch, _channels, _spatial;
    };
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar,
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

#include "Synet/Layers/EltwiseLayer.h"

#include "Synet/Utils/Math.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/BiasLayer.h"

namespace Synet
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

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> SYNET_INLINE void EltwiseLayerForwardCpu<float>(float const * const * src, const float * weight, size_t count, size_t size, EltwiseOperationType type, float * dst)
    {
        ::SimdSynetEltwiseLayerForward(src, weight, count, size, (::SimdSynetEltwiseOperationType)type, dst);
    }
#endif

    //-------------------------------------------------------------------------------------------------

    EltwiseLayer::EltwiseLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool EltwiseLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (_src.size() < 2 || dst.size() != 1)
            SYNET_ERROR("EltwiseLayer supports 2 or more inputs and 1 output!");
        const EltwiseParam & param = this->Param().eltwise();
        if (param.coefficients().size() != 0 && param.coefficients().size() != src.size())
            SYNET_ERROR("EltwiseLayer: check coefficients size!");
        if (param.operation() != EltwiseOperationTypeSum && param.coefficients().size())
            SYNET_ERROR("EltwiseLayer: check coefficients size!");

        for (size_t i = 0; i < src.size(); ++i)
        {
            if (src[0]->Shape() != src[i]->Shape())
                SYNET_ERROR("EltwiseLayer: all inputs must have the same shape!");
            if (src[i]->GetType() != TensorType32f && src[i]->GetType() != TensorType64i)
                SYNET_ERROR("EltwiseLayer: all inputs must be FP32 or INT64 type!");
            if (src[0]->Format() != src[i]->Format())
                SYNET_ERROR("EltwiseLayer: all inputs must have the same format!");
        }
        _operation = param.operation();
        _coefficients.resize(src.size(), 1.0f);
        if (param.coefficients().size())
        {
            for (size_t i = 0; i < src.size(); ++i)
                _coefficients[i] = param.coefficients()[i];
        }
        _type = src[0]->GetType();
        _size = src[0]->Size();
        _src.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            _src[i] = src[i]->RawCpuData();

        if (dst[0] != src[0])
            dst[0]->Reshape(_type, src[0]->Shape(), src[0]->Format());

        _const = true;
        for (size_t i = 0; i < src.size(); ++i)
            _const = _const && src[i]->Const();
        if (_const)
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
        }
        else
        {
            this->UsePerfStat();
            dst[0]->SetConst(false);
        }
        return true;
    }

    int64_t EltwiseLayer::Flop() const
    {
        return _size * (_coefficients.size() - 1) * (_operation == EltwiseOperationTypeSum ? 2 : 1);
    }

    void EltwiseLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_type)
        {
        case TensorType32f:
            EltwiseLayerForwardCpu((float const* const*)_src.data(), (const float*)_coefficients.data(), _src.size(), _size, _operation, dst[0]->Data<float>());
            break;
        case TensorType64i:
        {
            int64_t coefficients[2] = { 1, 1 };
            EltwiseLayerForwardCpu((int64_t const* const*)_src.data(), coefficients, _src.size(), _size, _operation, dst[0]->Data<int64_t>());
            break;
        }
        default:
            assert(0);
        }
    }
}
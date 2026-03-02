/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/Math/UnaryOperationLayer.h"

namespace Synet
{
    void UnaryOperation32f(const float* src, size_t size, UnaryOperationType type, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        ::SimdSynetUnaryOperation32f(src, size, (::SimdSynetUnaryOperation32fType)type, dst);
#else
        switch (type)
        {
        case UnaryOperationTypeAbs:
            for (size_t i = 0; i < size; ++i)
                dst[i] = Abs(src[i]);
            break;
        case UnaryOperationTypeCeil:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::ceil(src[i]);
            break;
        case UnaryOperationTypeCos:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::cosf(src[i]);
            break;
        case UnaryOperationTypeErf:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::erff(src[i]);
            break;
        case UnaryOperationTypeExp:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::exp(src[i]);
            break;
        case UnaryOperationTypeFloor:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::floor(src[i]);
            break;
        case UnaryOperationTypeLog:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::logf(src[i]);
            break;
        case UnaryOperationTypeNeg:
            for (size_t i = 0; i < size; ++i)
                dst[i] = -src[i];
            break;
        case UnaryOperationTypeNot:
            for (size_t i = 0; i < size; ++i)
                dst[i] = Not(src[i]);
            break;
        case UnaryOperationTypeRcp:
            for (size_t i = 0; i < size; ++i)
                dst[i] = 1.0f / src[i];
            break;
        case UnaryOperationTypeRound:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::round(src[i]);
            break;
        case UnaryOperationTypeRsqrt:
            for (size_t i = 0; i < size; ++i)
                dst[i] = 1.0f / ::sqrt(src[i]);
            break;
        case UnaryOperationTypeSign:
            for (size_t i = 0; i < size; ++i)
                dst[i] = src[i] < 0.0f ? -1.0f : (src[i] == 0.0f ? 0.0f : 1.0f);
            break;
        case UnaryOperationTypeSin:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::sinf(src[i]);
            break;
        case UnaryOperationTypeSqrt:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::sqrtf(src[i]);
            break;
        case UnaryOperationTypeTanh:
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::tanhf(src[i]);
            break;
        case UnaryOperationTypeZero:
            ::memset(dst, 0, size * sizeof(float));
            break;
        default:
            assert(0);
        }
#endif
    }

    void UnaryOperation64i(const int64_t* src, size_t size, UnaryOperationType type, int64_t* dst)
    {
        switch (type)
        {
        case UnaryOperationTypeAbs:
            for (size_t i = 0; i < size; ++i)
                dst[i] = Abs(src[i]);
            break;
        case UnaryOperationTypeNeg:
            for (size_t i = 0; i < size; ++i)
                dst[i] = -src[i];
            break;
        case UnaryOperationTypeNot:
            for (size_t i = 0; i < size; ++i)
                dst[i] = Not(src[i]);
            break;
        case UnaryOperationTypeZero:
            ::memset(dst, 0, size * sizeof(int64_t));
            break;
        default:
            assert(0);
        }
    }

    void UnaryOperationBool(const bool* src, size_t size, UnaryOperationType type, bool* dst)
    {
        switch (type)
        {
        case UnaryOperationTypeNot:
            for (size_t i = 0; i < size; ++i)
                dst[i] = !src[i];
            break;
        case UnaryOperationTypeZero:
            ::memset(dst, false, size * sizeof(bool));
            break;
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    UnaryOperationLayer::UnaryOperationLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool UnaryOperationLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("UnaryOperationLayer supports only 1 input and 1 output!");

        _opType = this->Param().unaryOperation().type();
        _size = src[0]->Size();
        _srcType = src[0]->GetType();
        switch (_srcType)
        {
        case TensorType32f: 
            if (_opType < UnaryOperationTypeAbs || _opType > UnaryOperationTypeZero)
                SYNET_ERROR("Unsupported value of UnaryOperation: " << Cpl::ToStr(_opType) << " for " << Cpl::ToStr(_srcType)  << " src !");
            break;
        case TensorType64i:
            if (_opType != UnaryOperationTypeAbs && _opType != UnaryOperationTypeNeg && _opType != UnaryOperationTypeNot && _opType != UnaryOperationTypeZero)
                SYNET_ERROR("Unsupported value of UnaryOperation: " << Cpl::ToStr(_opType) << " for " << Cpl::ToStr(_srcType) << " src !");
            break;
        case TensorTypeBool:
            if (_opType != UnaryOperationTypeNot && _opType != UnaryOperationTypeZero)
                SYNET_ERROR("Unsupported value of UnaryOperation: " << Cpl::ToStr(_opType) << " for " << Cpl::ToStr(_srcType) << " src !");
            break;
        default:
            SYNET_ERROR("Unsupported input type of UnaryOperationLayer: " << Cpl::ToStr(_srcType) << " !");
        }

        dst[0]->Reshape(_srcType, src[0]->Shape(), src[0]->Format());
        if (src[0]->Const())
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void UnaryOperationLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        switch (_srcType)
        {
        case TensorType32f: UnaryOperation32f(src[0]->Data<float>(), _size, _opType, dst[0]->Data<float>()); break;
        case TensorType64i: UnaryOperation64i(src[0]->Data<int64_t>(), _size, _opType, dst[0]->Data<int64_t>()); break;
        case TensorTypeBool: UnaryOperationBool(src[0]->Data<bool>(), _size, _opType, dst[0]->Data<bool>()); break;
        default:
            assert(0);
        }
    }
}
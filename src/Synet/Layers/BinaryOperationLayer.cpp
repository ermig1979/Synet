/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
#include "Synet/Layers/BinaryOperationLayer.h"

#include "Synet/Utils/Math.h"

#include <cmath>

namespace Synet
{
    template <BinaryOperationType type, class T> struct BinaryOperation;

    template <class T> struct BinaryOperation<BinaryOperationTypeAnd, T>
    {
        static T Run(T a, T b)
        {
            return And(a, b);
        }
    };

    //-------------------------------------------------------------------------------------------------
        
    template <class T> struct BinaryOperation<BinaryOperationTypeDiv, T>
    {
        static T Run(T a, T b)
        {
            return a / b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeDiv, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeMod, T>
    {
        static T Run(T a, T b)
        {
            return a % b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeMod, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeMod, float>
    {
        static float Run(float a, float b)
        {
            return ::fmodf(a, b);
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeSub, T>
    {
        static T Run(T a, T b)
        {
            return a - b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeSub, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <BinaryOperationType type, class T> void BinaryOperationRun(const T * a, const T * b, size_t outer, size_t aSize, size_t bSize, size_t inner, T * dst)
    {
        if (aSize == bSize)
        {
            size_t size = outer*aSize*inner;
            for (size_t i = 0; i < size; ++i)
                dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
        }
        else if (aSize == 1)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t s = 0; s < bSize; ++s)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
                    b += inner;
                    dst += inner;
                }
                a += inner;
            }
        }
        else if (bSize == 1)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t s = 0; s < aSize; ++s)
                {
                    for (size_t i = 0; i < inner; ++i)
                        dst[i] = BinaryOperation<type, T>::Run(a[i], b[i]);
                    a += inner;
                    dst += inner;
                }
                b += inner;
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    template<typename T> using BinaryOperationPtr = void(*)(const T* a, const T* b, size_t outer, size_t aSize, size_t bSize, size_t inner, T* dst);

    template <class T> BinaryOperationPtr<T> GetBinaryOperation(BinaryOperationType type)
    {
        switch (type)
        {
        case BinaryOperationTypeAnd: return BinaryOperationRun<BinaryOperationTypeAnd, T>;
        case BinaryOperationTypeDiv: return std::is_same<T, bool>::value ? NULL : BinaryOperationRun<BinaryOperationTypeDiv, T>;
        case BinaryOperationTypeMod: return std::is_same<T, bool>::value ? NULL : BinaryOperationRun<BinaryOperationTypeMod, T>;
        case BinaryOperationTypeSub: return std::is_same<T, bool>::value ? NULL : BinaryOperationRun<BinaryOperationTypeSub, T>;
        default: return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    BinaryOperationLayer::BinaryOperationLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool BinaryOperationLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("BinaryOperationLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("BinaryOperation inpus have different types: " << Cpl::ToStr(src[0]->GetType()) << " != " << Cpl::ToStr(src[1]->GetType()) << " !");

        _opType = this->Param().binaryOperation().type();
        _srcType = src[0]->GetType();
        switch (_srcType)
        {
        case TensorType32f:
            _func32f = GetBinaryOperation<float>(_opType);
            if (_func32f == NULL)
                SYNET_ERROR("Unsupported BinaryOperationType: " << Cpl::ToStr(_opType) << " !");
            break;
        case TensorTypeBool:
            _funcBool = GetBinaryOperation<bool>(_opType);
            if (_funcBool == NULL)
                SYNET_ERROR("Unsupported BinaryOperationType: " << Cpl::ToStr(_opType) << " !");
            break;
        case TensorType64i:
            _func64i = GetBinaryOperation<int64_t>(_opType);
            if (_func64i == NULL)
                SYNET_ERROR("Unsupported BinaryOperationType: " << Cpl::ToStr(_opType) << " !");
            break;
        default:
            SYNET_ERROR("Unsupported input type of BinaryOperationLayer: " << Cpl::ToStr(_srcType) << " !");
        }   

        Shape shape;
        _outer = 1, _aSize = 1, _bSize = 1, _inner = 1;
        for (size_t i = 0; i < src[0]->Count(); ++i)
        {
            if (i < src[1]->Count() && src[0]->Axis(i) == src[1]->Axis(i))
            {
                (_aSize*_bSize > 1 ? _inner : _outer) *= src[0]->Axis(i);
                shape.push_back(src[0]->Axis(i));
            }
            else if (src[0]->Axis(i) == 1)
            {
                _bSize *= src[1]->Axis(i);
                shape.push_back(src[1]->Axis(i));
            }
            else if (src[1]->Size() == 1 || src[1]->Axis(i) == 1)
            {
                _aSize *= src[0]->Axis(i);
                shape.push_back(src[0]->Axis(i));
            }
            else
                SYNET_ERROR("Incompatible input shapes of BinaryOperationLayer: " << Detail::DebugPrint(src[0]->Shape()) << " and " << Detail::DebugPrint(src[1]->Shape()) << " !");
        }

        dst[0]->Reshape(_srcType, shape, src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
        {
            ForwardCpu(src, buf, dst);
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

    void BinaryOperationLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_srcType)
        {
        case TensorType32f: _func32f(src[0]->Data<float>(), src[1]->Data<float>(), _outer, _aSize, _bSize, _inner, dst[0]->Data<float>()); break;
        case TensorTypeBool: _funcBool(src[0]->Data<bool>(), src[1]->Data<bool>(), _outer, _aSize, _bSize, _inner, dst[0]->Data<bool>()); break;
        case TensorType64i: _func64i(src[0]->Data<int64_t>(), src[1]->Data<int64_t>(), _outer, _aSize, _bSize, _inner, dst[0]->Data<int64_t>()); break;
        default:
            assert(0);
        }
    }
}
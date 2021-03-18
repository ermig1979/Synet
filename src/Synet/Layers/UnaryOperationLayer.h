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

namespace Synet
{
    namespace Detail
    {
        template <class T> void UnaryOperationLayerForward(const T * src, size_t size, UnaryOperationType type, T * dst)
        {
            switch (type)
            {
            case UnaryOperationTypeAbs:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = Abs(src[i]);
                break;
            case UnaryOperationTypeExp:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = ::exp(src[i]);
                break;
            case UnaryOperationTypeLog:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = ::log(src[i]);
                break;
            case UnaryOperationTypeNeg:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = -src[i];
                break;
            case UnaryOperationTypeRsqrt:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = 1.0f / ::sqrt(src[i]);
                break;
            case UnaryOperationTypeSqrt:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = ::sqrt(src[i]);
                break;
            case UnaryOperationTypeTanh:
                for (size_t i = 0; i < size; ++i)
                    dst[i] = ::tanh(src[i]);
                break;
            case UnaryOperationTypeZero:
                ::memset(dst, 0, size * sizeof(T));
                break;
            default:
                assert(0);
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && 0
        template <> SYNET_INLINE void UnaryOperationLayerForward<float>(const float * src, size_t size, UnaryOperationType type, float * dst)
        {
            ::SimdSynetUnaryOperation32fLayerForward(src, size, (::SimdSynetUnaryOperation32fType)type, dst);
        }
#endif
    }

    template <class T> class UnaryOperationLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        UnaryOperationLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _type = this->Param().unaryOperation().type();
            assert(_type >= UnaryOperationTypeAbs && _type <= UnaryOperationTypeZero);
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::UnaryOperationLayerForward(src[0]->CpuData(), src[0]->Size(), _type, dst[0]->CpuData());
        }

    private:
        UnaryOperationType _type;
    };
}
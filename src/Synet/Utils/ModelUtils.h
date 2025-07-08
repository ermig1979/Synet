/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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
#include "Synet/Params.h"

namespace Synet
{
    namespace ModelUtils
    {
        template<class T> static bool AllEqualTo(const std::vector<T>& vector, T value)
        {
            for (size_t i = 0; i < vector.size(); ++i)
                if (vector[i] != value)
                    return false;
            return true;
        }

        bool Equal(float a, float b, float e = 0.000001f)
        {
            return abs(a - b) < e;
        }

        //-------------------------------------------------------------------------------------------------

        static SYNET_INLINE bool IsAdd(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeSum &&
                (layer.eltwise().coefficients().empty() || layer.eltwise().coefficients() == Floats({ 1.0f, 1.0f })) && layer.src().size() == 2)
                return true;
            if (layer.type() == LayerTypeAdd)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsMul(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeProduct && layer.src().size() == 2)
                return true;
            if (layer.type() == LayerTypeMul)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsSub(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeEltwise && layer.eltwise().operation() == EltwiseOperationTypeSum &&
                layer.eltwise().coefficients() == Floats({ 1.0f, -1.0f }) && layer.src().size() == 2)
                return true;
            if (layer.type() == LayerTypeBinaryOperation && layer.binaryOperation().type() == BinaryOperationTypeSub)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsMulConst(const LayerParam& layer, float value, float epsilon = 0.000001)
        {
            if (layer.type() == LayerTypePower && layer.power().power() == 1.0f && layer.power().shift() == 0.0f
                && abs(layer.power().scale() - value) < epsilon)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsAddConst(const LayerParam& layer, float value, float epsilon = 0.000001)
        {
            if (layer.type() == LayerTypePower && layer.power().power() == 1.0f && layer.power().scale() == 1.0f
                && abs(layer.power().shift() - value) < epsilon)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsMetaConst64i(const LayerParam& layer, Longs value = Longs())
        {
            if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst && 
                layer.meta().alpha().type() == TensorType64i && (value.empty() || layer.meta().alpha().i64() == value))
                return true;
            return false;
        }

        static SYNET_INLINE bool IsMetaConst(const LayerParam& layer)
        {
            if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst)
                return true;
            return false;
        }

        static SYNET_INLINE bool IsDeptwiseConvolution(const LayerParam& layer, const Shape & kernel, const Shape& stride, bool bias, ActivationFunctionType activation)
        {
            if (layer.type() == LayerTypeConvolution && layer.convolution().group() == layer.convolution().outputNum() &&
                layer.convolution().stride() == stride &&
                layer.convolution().kernel() == kernel &&
                layer.convolution().biasTerm() == bias &&
                layer.convolution().activationType() == activation)
                return true;
            return false;
        }
    };
}
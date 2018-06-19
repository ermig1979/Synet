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
    template <class T> class MetaLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        MetaLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const MetaParam & param = this->Param().meta();
            switch (param.type())
            {
            case MetaTypeAdd: ReshapeAdd(src, dst); break;
            case MetaTypeCast: ReshapeCast(src, param.alpha(), dst); break;
            case MetaTypeConst: ReshapeConst(param.alpha(), dst); break;
            case MetaTypeExpandDims: ReshapeExpandDims(src, dst); break;
            case MetaTypeFill: ReshapeFill(src, dst); break;
            case MetaTypeInput: ReshapeInput(src, dst); break;
            case MetaTypeMinimum: ReshapeMinimum(src, dst); break;
            case MetaTypeMul: ReshapeMul(src, dst); break;
            case MetaTypePack: ReshapePack(src, dst); break;
            case MetaTypeRange: ReshapeRange(src, dst); break;
            case MetaTypeRealDiv: ReshapeRealDiv(src, dst); break;
            case MetaTypeReshape: ReshapeReshape(src, dst); break;
            case MetaTypeRsqrt: ReshapeRsqrt(src, dst); break;
            case MetaTypeShape: ReshapeShape(src, dst); break;
            case MetaTypeSlice: ReshapeSlice(src, dst); break;
            case MetaTypeSqrt: ReshapeSqrt(src, dst); break;
            case MetaTypeStridedSlice: ReshapeStridedSlice(src, dst); break;
            case MetaTypeStub: /*dst[0]->Reshape({});*/ break;
            case MetaTypeSub: ReshapeSub(src, dst); break;
            case MetaTypeSwitch: ReshapeSwitch(src, dst); break;
            case MetaTypeTile: ReshapeTile(src, dst); break;
            case MetaTypeUnpack: ReshapeUnpack(src, dst); break;
            default:
                assert(0);
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

    private:

        void ReshapeAdd(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType());
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = src0.CpuData()[i] + src1.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeCast(const TensorPtrs & src, const TensorParam & alpha, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            if (alpha.type() == TensorType32i)
            {
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src[0]->Shape());
                if (src[0]->GetType() == TensorType32i)
                {
                    const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = src0.CpuData()[i];

                }
                else if (src[0]->GetType() == TensorType32f)
                {
                    const Synet::Tensor<float> & src0 = src[0]->As32f();
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = (int32_t)src0.CpuData()[i];
                }
                else
                    assert(0);
            }
            else if (alpha.type() == TensorType32f)
            {
                assert(src[0]->GetType() == TensorType32i);
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = (float)src0.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeConst(const TensorParam & alpha, const TensorPtrs & dst)
        {
            if (alpha.type() == TensorType32i)
            {
                dst[0]->Import(alpha);
            }
            else
                assert(0);
        }

        void ReshapeExpandDims(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->GetType() == TensorType32i && src[1]->Size() == 1);
            const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
            ptrdiff_t axis = src1.CpuData()[0];
            if (axis < 0)
                axis += src[0]->Count();
            Shape shape;
            for (ptrdiff_t i = 0; i < axis; ++i)
                shape.push_back(src[0]->Axis(i));
            shape.push_back(1);
            for (size_t i = axis; i < src[0]->Count(); ++i)
                shape.push_back(src[0]->Axis(i));
            dst[0]->ShareAs(*src[0], shape);
        }

        void ReshapeFill(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->GetType() == TensorType32i && src[0]->Count() == 1 && src[1]->Size() == 1);
            const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
            Shape shape;
            for (size_t i = 0; i < src0.Size(); ++i)
                shape.push_back(src0.CpuData()[i]);
            if (src[1]->GetType() == TensorType32i)
            {
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(shape, src[1]->As32i().CpuData()[0]);
            }
            else
                assert(0);
        }

        void ReshapeInput(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = src0.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeMinimum(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2);
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = std::min(src0.CpuData()[i], src1.CpuData()[i]);
            }
            else
                assert(0);
        }

        void ReshapeMul(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType());
            if (src[0]->GetType() == TensorType32f)
            {
                const Synet::Tensor<float> & src0 = src[0]->As32f();
                const Synet::Tensor<float> & src1 = src[1]->As32f();
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = src0.CpuData()[i] * src1.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapePack(const TensorPtrs & src, const TensorPtrs & dst)
        {
            if (src[0]->GetType() == TensorType32i)
            {
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ src.size() });
                for (size_t i = 0; i < src.size(); ++i)
                {
                    const Synet::Tensor<int32_t> & srci = src[i]->As32i();
                    assert(srci.Size() == 1);
                    dst0.CpuData()[i] = srci.CpuData()[0];
                }
            }
            else
                assert(0);
        }

        void ReshapeRange(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 3 && src[0]->Size() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1);
            if (src[0]->GetType() == TensorType32i)
            {
                int32_t begin = src[0]->As32i().CpuData()[0];
                int32_t end = src[1]->As32i().CpuData()[0];
                int32_t step = src[1]->As32i().CpuData()[0];
                Ints result;
                if (step > 0)
                    for (int32_t i = begin; i < end; i += step)
                        result.push_back(i);
                else
                    for (int32_t i = begin; i > end; i += step)
                        result.push_back(i);
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ result.size() });
                for (size_t i = 0; i < result.size(); ++i)
                    dst0.CpuData()[i] = result[0];
            }
            else
                assert(0);
        }

        void ReshapeRealDiv(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType());
            if (src[0]->GetType() == TensorType32f)
            {
                const Synet::Tensor<float> & src0 = src[0]->As32f();
                const Synet::Tensor<float> & src1 = src[1]->As32f();
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = src0.CpuData()[i] / src1.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeReshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->GetType() == TensorType32i && src[1]->Count() == 1 && src[1]->Size() == src[0]->Count());
            const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
            Shape shape(src1.CpuData(), src1.CpuData() + src1.Size());
            size_t unknown = 0;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] == -1)
                    unknown++;
            }
            assert(unknown <= 1);
            if (unknown)
            {
                size_t known = 1, index = shape.size();
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (shape[i] != -1)
                        known *= shape[i];
                    else
                        index = i;
                }
                shape[index] = src[0]->Size() / known;
            }
            dst[0]->ShareAs(*src[0], shape);
        }

        void ReshapeRsqrt(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            if (src[0]->GetType() == TensorType32f)
            {
                const Synet::Tensor<float> & src0 = src[0]->As32f();
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = 1.0f/::sqrt(src0.CpuData()[i]);
            }
            else
                assert(0);
        }

        void ReshapeShape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            Shape shape = src[0]->Shape();
            if (shape.size() == 4)
                shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
            if (shape.size() == 2)
                shape = Shape({ shape[1], shape[0] });
            Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
            dst0.Reshape({ shape.size() });
            for (size_t i = 0; i < shape.size(); ++i)
                dst0.CpuData()[i] = (int32_t)shape[i];
        }

        void ReshapeSlice(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 3 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1);
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                size_t begin = src[1]->As32i().CpuData()[0];
                size_t size = src[2]->As32i().CpuData()[0];
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ size });
                for (size_t i = 0; i < size; ++i)
                    dst0.CpuData()[i] = src0.CpuData()[begin + i];
            }
            else
                assert(0);
        }

        void ReshapeSqrt(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            if (src[0]->GetType() == TensorType32f)
            {
                const Synet::Tensor<float> & src0 = src[0]->As32f();
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = ::sqrt(src0.CpuData()[i]);
            }
            else
                assert(0);
        }

        void ReshapeStridedSlice(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 4 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1);
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                size_t begin = src[1]->As32i().CpuData()[0];
                size_t end = src[2]->As32i().CpuData()[0];
                size_t step = src[3]->As32i().CpuData()[0];
                Shape result;
                for (size_t i = begin; i < end; i += step)
                    result.push_back(src0.CpuData()[i]);
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ result.size() });
                for (size_t i = 0; i < result.size(); ++i)
                    dst0.CpuData()[i] = (int32_t)result[i];
            }
            else
                assert(0);
        }

        void ReshapeSub(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType());
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = src0.CpuData()[i] - src1.CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeSwitch(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->Size() == 1 && src[1]->GetType() == TensorType32i);
            int32_t pred = src[1]->As32i().CpuData()[0];
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                Synet::Tensor<int32_t> & dst1 = dst[1]->As32i();
                if (pred)
                {
                    dst0.Reshape(src0.Shape());
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = src0.CpuData()[i];
                }
                else
                {
                    dst1.Reshape(src0.Shape());
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst1.CpuData()[i] = src0.CpuData()[i];
                }
            }
            else
                assert(0);
        }

        void ReshapeTile(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->GetType() == TensorType32i && src[1]->Count() == 1 && src[1]->Size() == src[0]->Count());
            const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
            Shape multiples(src1.CpuData(), src1.CpuData() + src1.Size());
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                Shape oldShape = src[0]->Shape();
                Shape newShape = oldShape;
                for (size_t i = 0; i < multiples.size(); ++i)
                    newShape[i] *= multiples[i];
                dst0.Reshape(newShape);
                if (oldShape.size() == 1)
                {
                    for (size_t i0 = 0; i0 < multiples[0]; ++i0)
                    {
                        for (size_t j0 = 0; j0 < oldShape[0]; ++j0)
                        {
                            size_t srcIdx = i0;
                            size_t dstIdx = i0*oldShape[0] + j0;
                            dst0.CpuData()[dstIdx] = src0.CpuData()[srcIdx];
                        }
                    }
                }
                else
                    assert(0);
            }
            else
                assert(0);
        }

        void ReshapeUnpack(const TensorPtrs & src, const TensorPtrs & dst)
        {
            if (src[0]->GetType() == TensorType32i)
            {
                Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                assert(src0.Size() == dst.size());
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    Synet::Tensor<int32_t> & dsti = dst[i]->As32i();
                    dsti.Reshape({ size_t(1) });
                    dsti.CpuData()[0] = src0.CpuData()[i];
                }
            }
            else
                assert(0);
        }
    };
}
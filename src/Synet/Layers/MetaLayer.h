/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

        MetaLayer(const LayerParam & param, Context* context)
            : Base(param, context)
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
            case MetaTypeDiv: ReshapeDiv(src, dst); break;
            case MetaTypeExpandDims: ReshapeExpandDims(src, dst); break;
            case MetaTypeGather: ReshapeGather(src, dst); break;
            case MetaTypeGreater: ReshapeGreater(src, dst); break;
            case MetaTypeFill: ReshapeFill(src, dst); break;
            case MetaTypeFloor: ReshapeFloor(src, dst); break;
            case MetaTypeInput: ReshapeInput(src, dst); break;
            case MetaTypeInputWithDefault: ReshapeInputWithDefault(src, dst); break;
            case MetaTypeMaximum: ReshapeMaximum(src, dst); break;
            case MetaTypeMinimum: ReshapeMinimum(src, dst); break;
            case MetaTypeMul: ReshapeMul(src, dst); break;
            case MetaTypePack: ReshapePack(src, dst); break;
            case MetaTypeRange: ReshapeRange(src, dst); break;
            case MetaTypeRealDiv: ReshapeRealDiv(src, dst); break;
            case MetaTypeReduceMin: ReshapeReduceMin(src, dst); break;
            case MetaTypeReduceProd: ReshapeReduceProd(src, dst); break;
            case MetaTypeReshape: ReshapeReshape(src, dst); break;
            case MetaTypeRsqrt: ReshapeRsqrt(src, dst); break;
            case MetaTypeShape: ReshapeShape(src, param.version(), dst); break;
            case MetaTypeSlice: ReshapeSlice(src, dst); break;
            case MetaTypeSqrt: ReshapeSqrt(src, dst); break;
            case MetaTypeSqueeze: ReshapeSqueeze(src, dst); break;
            case MetaTypeStridedSlice: ReshapeStridedSlice(src, dst); break;
            case MetaTypeStub: /*dst[0]->Reshape({});*/ break;
            case MetaTypeSub: ReshapeSub(src, dst); break;
            case MetaTypeSwitch: ReshapeSwitch(src, dst); break;
            case MetaTypeTensorArray: ReshapeTensorArray(src, param.alpha(), dst); break;
            case MetaTypeTensorArrayRead: ReshapeTensorArrayRead(src, dst); break;
            case MetaTypeTensorArraySize: ReshapeTensorArraySize(src, dst); break;
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

        template<class S, class D> static void Convert(const Synet::Tensor<S>& src, Synet::Tensor<D>& dst)
        {
            dst.Reshape(src.Shape(), src.Format());
            for (size_t i = 0, n = src.Size(); i < n; ++i)
                dst.CpuData()[i] = (D)src.CpuData()[i];
        }

        template<class D> static D GetAs(const Synet::Tensor<T> & tensor, size_t offset)
        {
            switch (tensor.GetType())
            {
            case TensorType32i: return (D)tensor.As32i().CpuData()[offset];
            case TensorType64i: return (D)tensor.As64i().CpuData()[offset];
            default: assert(0); return D(0);
            }
        }

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
            assert(src.size() == 1 && dst.size() == 1);
            switch (src[0]->GetType())
            {
            case TensorType32f:
                switch (alpha.type())
                {
                case TensorType32i: Convert(src[0]->As32f(), dst[0]->As32i()); break;
                case TensorType32f: Convert(src[0]->As32f(), dst[0]->As32f()); break;
                case TensorType64i: Convert(src[0]->As32f(), dst[0]->As64i()); break;
                default: assert(0);
                }
                break;
            case TensorType32i:
                switch (alpha.type())
                {
                case TensorType32i: Convert(src[0]->As32i(), dst[0]->As32i()); break;
                case TensorType32f: Convert(src[0]->As32i(), dst[0]->As32f()); break;
                case TensorType64i: Convert(src[0]->As32i(), dst[0]->As64i()); break;
                default: assert(0);
                }
                break;
            case TensorType64i:
                switch (alpha.type())
                {
                case TensorType32i: Convert(src[0]->As64i(), dst[0]->As32i()); break;
                case TensorType32f: Convert(src[0]->As64i(), dst[0]->As32f()); break;
                case TensorType64i: Convert(src[0]->As64i(), dst[0]->As64i()); break;
                default: assert(0);
                }
                break;
            default: assert(0);
            }
        }

        void ReshapeConst(const TensorParam & alpha, const TensorPtrs & dst)
        {
            dst[0]->Import(alpha);
        }

        void ReshapeDiv(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Size() == src[1]->Size());
            if (src[0]->GetType() == TensorType32f && src[1]->GetType() == TensorType32f)
            {
                dst[0]->As32f().Reshape(src[0]->Shape());
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    dst[0]->As32f().CpuData()[i] = src[0]->As32f().CpuData()[i] / src[1]->As32f().CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeExpandDims(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1 || src.size() == 2);
            ptrdiff_t axis;
            if (src.size() == 2)
            {
                assert(src[1]->Size() == 1);
                axis = GetAs<size_t>(*src[1], 0);
            }
            else
            {
                const TensorParam& alpha = this->Param().meta().alpha();
                assert(alpha.shape() == Shp(1) && alpha.type() == TensorType64i);
                axis = alpha.i64()[0];
            }
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

        void ReshapeFloor(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            if (src[0]->GetType() == TensorType32f)
            {
                dst[0]->As32f().Reshape(src[0]->Shape());
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    dst[0]->As32f().CpuData()[i] = ::floor(src[0]->As32f().CpuData()[i]);
            }
            else
                assert(0);
        }

        void ReshapeGather(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() >= 2 && (src[1]->GetType() == TensorType32i || src[1]->GetType() == TensorType64i));
            Shape idx(src[1]->Size());
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = src[1]->GetType() == TensorType32i ? (size_t)src[1]->As32i().CpuData()[i] : (size_t)src[1]->As64i().CpuData()[i];
            if (src[0]->GetType() == TensorType32f)
            {
                dst[0]->As32f().Reshape(src[1]->Shape());
                for (size_t i = 0; i < idx.size(); ++i)
                    dst[0]->As32f().CpuData()[i] = src[0]->As32f().CpuData()[idx[i]];
            }
            else if (src[0]->GetType() == TensorType32i)
            {
                dst[0]->As32i().Reshape(src[1]->Shape());
                for (size_t i = 0; i < idx.size(); ++i)
                    dst[0]->As32i().CpuData()[i] = src[0]->As32i().CpuData()[idx[i]];
            }
            else if (src[0]->GetType() == TensorType64i)
            {
                dst[0]->As64i().Reshape(src[1]->Shape());
                for (size_t i = 0; i < idx.size(); ++i)
                    dst[0]->As64i().CpuData()[i] = src[0]->As64i().CpuData()[idx[i]];
            }
            else
                assert(0);
        }

        void ReshapeGreater(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2);
            Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
            dst0.Reshape(src[0]->Shape());
            if (src[0]->GetType() == TensorType32f)
            {
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    dst0.CpuData()[i] = src[0]->As32f().CpuData()[i] > src[1]->As32f().CpuData()[i] ? 1 : 0;
            }
            else if (src[0]->GetType() == TensorType32i)
            {
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    dst0.CpuData()[i] = src[0]->As32i().CpuData()[i] > src[1]->As32i().CpuData()[i] ? 1 : 0;
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

        void ReshapeInputWithDefault(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2);
            Synet::Tensor<int32_t> * pSrc = NULL;
            if (src[0]->GetType() == TensorType32i)
                pSrc = &src[0]->As32i();
            else if (src[1]->GetType() == TensorType32i)
                pSrc = &src[1]->As32i();
            assert(pSrc);
            Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
            dst0.Reshape(pSrc->Shape());
            for (size_t i = 0; i < dst0.Size(); ++i)
                dst0.CpuData()[i] = pSrc->CpuData()[i];
        }

        void ReshapeMaximum(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2);
            if (src[0]->GetType() == TensorType32i)
            {
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                const Synet::Tensor<int32_t> & src1 = src[1]->As32i();
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape(src0.Shape());
                for (size_t i = 0; i < src0.Size(); ++i)
                    dst0.CpuData()[i] = Max(src0.CpuData()[i], src1.CpuData()[i]);
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
                    dst0.CpuData()[i] = Min(src0.CpuData()[i], src1.CpuData()[i]);
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
            if (src[0]->GetType() == TensorType32f)
            {
                Synet::Tensor<float> & dst0 = dst[0]->As32f();
                size_t size = 0;
                for (size_t i = 0; i < src.size(); ++i)
                {
                    assert(src[i]->Count() == 1);
                    size += src[i]->Axis(0);
                }
                dst0.Reshape({ size });
                for (size_t s = 0, d = 0; s < src.size(); ++s)
                {
                    for (size_t i = 0; i < src[s]->Axis(0); ++i, ++d)
                        dst0.CpuData()[d] = src[s]->As32f().CpuData()[i];
                }
            }
            else if (src[0]->GetType() == TensorType32i)
            {
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ src.size() });
                for (size_t i = 0; i < src.size(); ++i)
                {
                    assert(src[i]->Size() == 1);
                    if (src[i]->GetType() == TensorType32i)
                        dst0.CpuData()[i] = (int32_t)src[i]->As32i().CpuData()[0];
                    else if (src[i]->GetType() == TensorType64i)
                        dst0.CpuData()[i] = (int32_t)src[i]->As64i().CpuData()[0];
                    else
                        assert(0);

                }
            }
            else if (src[0]->GetType() == TensorType64i)
            {
                Synet::Tensor<int64_t>& dst0 = dst[0]->As64i();
                size_t size = 0;
                for (size_t i = 0; i < src.size(); ++i)
                {
                    assert(src[i]->Count() == 1);
                    size += src[i]->Size();
                }
                dst0.Reshape({ size });
                for (size_t i = 0, o = 0; i < src.size(); ++i)
                {
                    if (src[i]->GetType() == TensorType32i)
                    {
                        for (size_t j = 0; j < src[i]->Size(); ++j, ++o)
                            dst0.CpuData()[o] = (int64_t)src[i]->As32i().CpuData()[j];
                    }
                    else if (src[i]->GetType() == TensorType64i)
                    {
                        for (size_t j = 0; j < src[i]->Size(); ++j, ++o)
                            dst0.CpuData()[o] = (int64_t)src[i]->As64i().CpuData()[j];
                    }
                    else
                        assert(0);
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
                int32_t step = src[2]->As32i().CpuData()[0];
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
            else if (src[0]->GetType() == TensorType64i)
            {
                int64_t begin = src[0]->As64i().CpuData()[0];
                int64_t end = src[1]->As64i().CpuData()[0];
                int64_t step = src[2]->As64i().CpuData()[0];
                Ints result;
                if (step > 0)
                    for (int64_t i = begin; i < end; i += step)
                        result.push_back((int)i);
                else
                    for (int64_t i = begin; i > end; i += step)
                        result.push_back((int)i);
                Synet::Tensor<int64_t> & dst0 = dst[0]->As64i();
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

        void ReshapeReduceMin(const TensorPtrs& src, const TensorPtrs& dst)
        {
            bool keepDims = this->Param().meta().alpha().i32()[0] != 0;
            size_t axis = (size_t)src[1]->As64i().CpuData()[0];
            assert(src[0]->Count() == 1);
            if (src[0]->GetType() == TensorType32i)
            {
                int32_t min = INT_MAX;
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    min = Min(min, src[0]->As32i().CpuData()[i]);
                dst[0]->As64i().Reshape(Shp(1), min);
            }
            else
                assert(0);
        }

        void ReshapeReduceProd(const TensorPtrs& src, const TensorPtrs& dst)
        {
            bool keepDims = this->Param().meta().alpha().i32()[0] != 0;
            size_t axis = (size_t)src[1]->As64i().CpuData()[0];
            assert(src[0]->Count() == 1);
            int64_t prod = 1;
            for (size_t i = 0; i < src[0]->Size(); ++i)
                prod *= src[0]->As64i().CpuData()[i];
            dst[0]->As64i().Reshape(Shp(1), prod);
        }

        void ReshapeReshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[1]->Count() == 1);//&& src[1]->Size() == src[0]->Count());
            Shape shape(src[1]->Size());
            if (src[1]->GetType() == TensorType32i)
            {
                for (size_t i = 0; i < shape.size(); ++i)
                    shape[i] = (size_t)src[1]->As32i().CpuData()[i];
            }
            if (src[1]->GetType() == TensorType64i)
            {
                for (size_t i = 0; i < shape.size(); ++i)
                    shape[i] = (size_t)src[1]->As64i().CpuData()[i];
            }
            else
                assert(0);
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

        void ReshapeShape(const TensorPtrs & src, int version, const TensorPtrs & dst)
        {
            assert(src.size() == 1);
            Shape shape = src[0]->Shape();
            bool trans = src[0]->Format() == TensorFormatNhwc;
            if (version == 0)
            {
                if (!trans && shape.size() == 4)
                    shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                if (!trans && shape.size() == 2)
                    shape = Shape({ shape[1], shape[0] });
                Synet::Tensor<int32_t>& dst0 = dst[0]->As32i();
                dst0.Reshape({ shape.size() });
                for (size_t i = 0; i < shape.size(); ++i)
                    dst0.CpuData()[i] = (int32_t)shape[i];
            }
            else
            {
                if (trans && shape.size() == 4)
                    shape = Shape({ shape[0], shape[3], shape[1], shape[2] });
                Synet::Tensor<int64_t>& dst0 = dst[0]->As64i();
                dst0.Reshape({ shape.size() }, src[0]->Format());
                for (size_t i = 0; i < shape.size(); ++i)
                    dst0.CpuData()[i] = (int64_t)shape[i];
            }
        }

        void ReshapeSlice(const TensorPtrs & src, const TensorPtrs & dst)
        {
            if (src[0]->GetType() == TensorType32i)
            {
                assert(src.size() == 3 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1);
                const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                size_t begin = src[1]->As32i().CpuData()[0];
                size_t size = src[2]->As32i().CpuData()[0];
                Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                dst0.Reshape({ size });
                for (size_t i = 0; i < size; ++i)
                    dst0.CpuData()[i] = src0.CpuData()[begin + i];
            }
            else if (src[0]->GetType() == TensorType64i)
            {
                assert(src.size() == 4 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1);
                const Synet::Tensor<int64_t> & src0 = src[0]->As64i();
                size_t beg = src[1]->As64i().CpuData()[0];
                size_t end = Min<size_t>(src[2]->As64i().CpuData()[0], src[0]->Size());
                Synet::Tensor<int64_t> & dst0 = dst[0]->As64i();
                dst0.Reshape({ end - beg });
                for (size_t i = beg; i < end; ++i)
                    dst0.CpuData()[i - beg] = src0.CpuData()[i];
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

        void ReshapeSqueeze(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2);
            if (src[0]->GetType() == TensorType64i)
            {
                assert(src[0]->Size() == 1 && src[1]->Size() == 1);
                dst[0]->As64i().Reshape(src[0]->Shape(), src[0]->Format());
                for (size_t i = 0; i < src[0]->Size(); ++i)
                    dst[0]->As64i().CpuData()[i] = src[0]->As64i().CpuData()[i];
            }
            else
                assert(0);
        }

        void ReshapeStridedSlice(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() >= 3 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1);
            size_t begin = GetAs<size_t>(*src[1], 0);
            size_t end = GetAs<size_t>(*src[2], 0);
            size_t step = src.size() > 3 ? GetAs<size_t>(*src[3], 0) : size_t(1);
            Shape result;
            for (size_t i = begin; i < end; i += step)
                result.push_back(GetAs<size_t>(*src[0], i));
            if (src[0]->GetType() == TensorType32i)
            {
                dst[0]->As32i().Reshape(Shp(result.size()), src[0]->Format());
                for (size_t i = 0; i < result.size(); ++i)
                    dst[0]->As32i().CpuData()[i] = (int32_t)result[i];
            }
            else if (src[0]->GetType() == TensorType64i)
            {
                dst[0]->As64i().Reshape(Shp(result.size()), src[0]->Format());
                for (size_t i = 0; i < result.size(); ++i)
                    dst[0]->As64i().CpuData()[i] = (int64_t)result[i];
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
                    dst1.Reshape(src0.Shape());
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst1.CpuData()[i] = src0.CpuData()[i];
                }
                else
                {
                    dst0.Reshape(src0.Shape());
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = src0.CpuData()[i];
                }
            }
            else
                assert(0);
        }

        void ReshapeTensorArray(const TensorPtrs & src, const TensorParam & alpha, const TensorPtrs & dst)
        {
            assert(src.size() == 1 && src[0]->Size() == 1 && src[0]->GetType() == TensorType32i && dst.size() == 2);
            size_t size = src[1]->As32i().CpuData()[0];
            if (alpha.type() == TensorType32f)
            {
                dst[0]->As32f().Reshape({ size });
                dst[1]->As32f().Reshape({ size });
            }
            else if (alpha.type() == TensorType32i)
            {
                dst[0]->As32i().Reshape({ size });
                dst[1]->As32i().Reshape({ size });
            }
            else
                assert(0);
        }

        void ReshapeTensorArrayRead(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 3 && src[1]->Size() == 1 && src[1]->GetType() == TensorType32i && dst.size() == 1);
            size_t index = src[1]->As32i().CpuData()[0];
            if (src[0]->GetType() == TensorType32f)
            {
                dst[0]->As32f().Reshape({ size_t(1)});
                dst[0]->As32f().CpuData()[0] = src[0]->As32f().CpuData()[index];
            }
            else if (src[0]->GetType() == TensorType32i)
            {
                dst[0]->As32i().Reshape({ size_t(1) });
                dst[0]->As32i().CpuData()[0] = src[0]->As32i().CpuData()[index];
            }
            else
                assert(0);
        }

        void ReshapeTensorArraySize(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && dst.size() == 2);
            size_t size = src[0]->Size();
            dst[0]->As32i().Reshape({ size_t(1) });
            dst[0]->As32i().CpuData()[0] = (int)size;
            dst[1]->As32i().Reshape({ size_t(1) });
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
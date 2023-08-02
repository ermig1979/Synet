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

#include "Synet/Layers/MetaLayer.h"

namespace Synet
{
    MetaLayer::MetaLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool MetaLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const MetaParam & param = this->Param().meta();
        switch (param.type())
        {
        case MetaTypeAdd: return ReshapeAdd(src, dst);
        case MetaTypeCast: return ReshapeCast(src, param.alpha(), dst);
        case MetaTypeConst: return ReshapeConst(param.alpha(), dst);
        case MetaTypeDiv: return ReshapeDiv(src, dst);
        case MetaTypeEqual: return ReshapeEqual(src, dst);
        case MetaTypeExpandDims: return ReshapeExpandDims(src, param.alpha(), dst);
        case MetaTypeFloor: return ReshapeFloor(src, dst);
        case MetaTypeGather: return ReshapeGather(src, dst);
        case MetaTypeMul: return ReshapeMul(src, dst);
        case MetaTypePack: ReshapePack(src, dst); break;
        case MetaTypePermute: ReshapePermute(src, param.alpha(), dst); break;
        case MetaTypeRange: ReshapeRange(src, dst); break;
        case MetaTypeReduceMin: ReshapeReduceMin(src, dst); break;
        case MetaTypeReduceProd: ReshapeReduceProd(src, dst); break;
        case MetaTypeReshape: ReshapeReshape(src, dst); break;
        case MetaTypeSelect: ReshapeSelect(src, dst); break;
        case MetaTypeShape: ReshapeShape(src, param.version(), dst); break;
        case MetaTypeSlice: return ReshapeSlice(src, dst);
        case MetaTypeSqueeze: ReshapeSqueeze(src, dst); break;
        case MetaTypeStridedSlice: ReshapeStridedSlice(src, dst); break;
        case MetaTypeStub: ReshapeStub(src, dst); break;
        case MetaTypeSub: ReshapeSub(src, dst); break;
        default:
            SYNET_ERROR("Unsupported meta type: " << Cpl::ToStr(param.type()) << " !");
        }
        return true;
    }

    void MetaLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }

    //-------------------------------------------------------------------------------------------------

    template<class S, class D> static void Convert(const Synet::Tensor<S>& src, Synet::Tensor<D>& dst)
    {
        dst.Reshape(src.Shape(), src.Format());
        for (size_t i = 0, n = src.Size(); i < n; ++i)
            dst.CpuData()[i] = (D)src.CpuData()[i];
    }

    template<class D> static D GetAs(const Synet::Tensor<float>& tensor, size_t offset)
    {
        switch (tensor.GetType())
        {
        case TensorType32i: return (D)tensor.Data<int32_t>()[offset];
        case TensorType64i: return (D)tensor.Data<int64_t>()[offset];
        default: assert(0); return D(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> bool ReshapeAdd(const Synet::Tensor<float>& src0, const Synet::Tensor<float>& src1, Synet::Tensor<float>& dst0)
    {
        dst0.Reshape(Synet::GetTensorType<T>(), src0.Shape(), src0.Format());
        if (src0.Size() == src1.Size())
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] + src1.Data<T>()[i];
        }
        else if (src1.Size() == 1)
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] + src1.Data<T>()[0];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeAdd supported input shapes!");
        return true;
    }

    bool MetaLayer::ReshapeAdd(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeAdd supports only 2 inputs and 1 output!");
        if(src[0]->Count() != src[1]->Count() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeAdd unsupported input shape or type combination!");
        switch (src[0]->GetType())
        {
        case TensorType32f: return Synet::ReshapeAdd<float>(*src[0], *src[1], *dst[0]);
        case TensorType32i: return Synet::ReshapeAdd<int32_t>(*src[0], *src[1], *dst[0]);        
        case TensorType64i: return Synet::ReshapeAdd<int64_t>(*src[0], *src[1], *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeAdd unsupported input type!");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    template<class S, class D> void ReshapeCast(const Synet::Tensor<float>& src, Synet::Tensor<float>& dst)
    {
        for (size_t i = 0, n = src.Size(); i < n; ++i)
            dst.Data<D>()[i] = (D)src.Data<S>()[i];
    }

    template<class S> bool ReshapeCast(const Synet::Tensor<float>& src, TensorType type, Synet::Tensor<float>& dst)
    {
        dst.Reshape(type, src.Shape(), src.Format());
        switch (type)
        {
        case TensorType32i: ReshapeCast<S, int32_t>(src, dst); break;
        case TensorType32f: ReshapeCast<S, float>(src, dst); break;
        case TensorType64i: ReshapeCast<S, int64_t>(src, dst); break;
        case TensorType64u: ReshapeCast<S, uint64_t>(src, dst); break;
        default:
            SYNET_ERROR("MetaLayer::ReshapeCast can't convert " << Cpl::ToStr(src.GetType()) << " to " << Cpl::ToStr(type) << " !");
        }
        return true;
    }

    bool MetaLayer::ReshapeCast(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeCast supports only 1 inputs and 1 output!");
        switch (src[0]->GetType())
        {
        case TensorType32f: return Synet::ReshapeCast<float>(*src[0], alpha.type(), *dst[0]);
        case TensorType32i: return Synet::ReshapeCast<int32_t>(*src[0], alpha.type(), *dst[0]);
        case TensorType64i: return Synet::ReshapeCast<int64_t>(*src[0], alpha.type(), *dst[0]);
        case TensorType64u: return Synet::ReshapeCast<uint64_t>(*src[0], alpha.type(), *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeCast can't convert " << Cpl::ToStr(src[0]->GetType()) << " to " << Cpl::ToStr(alpha.type()) << " !");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeConst(const TensorParam& alpha, const TensorPtrs& dst)
    {
        return dst[0]->Import(alpha);
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> bool ReshapeDiv(const Synet::Tensor<float>& src0, const Synet::Tensor<float>& src1, Synet::Tensor<float>& dst0)
    {
        dst0.Reshape(Synet::GetTensorType<T>(), src0.Shape(), src0.Format());
        if (src0.Size() == src1.Size())
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] / src1.Data<T>()[i];
        }
        else if (src1.Size() == 1)
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] / src1.Data<T>()[0];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeDiv supported input shapes!");
        return true;
    }

    bool MetaLayer::ReshapeDiv(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeDiv supports only 2 inputs and 1 output!");
        if (src[0]->Count() != src[1]->Count() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeDiv unsupported input shape or type combination!");
        switch (src[0]->GetType())
        {
        case TensorType32f: return Synet::ReshapeDiv<float>(*src[0], *src[1], *dst[0]);
        case TensorType32i: return Synet::ReshapeDiv<int32_t>(*src[0], *src[1], *dst[0]);
        case TensorType64i: return Synet::ReshapeDiv<int64_t>(*src[0], *src[1], *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeDiv unsupported input type!");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeEqual(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeEqual supports only 2 inputs and 1 output!");
        if (src[0]->Size() != src[1]->Size() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeEqual unsupported input shape or type combination!");
        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        if (src[0]->GetType() == TensorType64i)
        {
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->Data<int64_t>()[i] = (src[0]->Data<int64_t>()[i] == src[1]->Data<int64_t>()[i]) ? 1 : 0;
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeEqual unsupported input type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeExpandDims(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeExpandDims supports only 1 or 2 inputs and 1 output!");
        ptrdiff_t axis;
        if (src.size() == 2)
        {
            if(src[1]->Size() != 1 || (src[1]->GetType() != TensorType32i && src[1]->GetType() != TensorType64i))
                SYNET_ERROR("MetaLayer::ReshapeExpandDims src[1] has wrong type or shape!");
            axis = GetAs<ptrdiff_t>(*src[1], 0);
        }
        else
        {
            if(alpha.shape() != Shp(1) || alpha.type() != TensorType64i)
                SYNET_ERROR("MetaLayer::ReshapeExpandDims has wrong alpha parameter!");
            axis = (ptrdiff_t)alpha.i64()[0];
        }
        if (axis < 0)
            axis += src[0]->Count();
        Shape shape;
        for (ptrdiff_t i = 0; i < axis; ++i)
            shape.push_back(src[0]->Axis(i));
        shape.push_back(1);
        for (size_t i = axis; i < src[0]->Count(); ++i)
            shape.push_back(src[0]->Axis(i));
        dst[0]->ShareAs(*src[0], shape, src[0]->Format());
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeFloor(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeFloor supports only 1 input and 1 output!");

        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->Data<float>()[i] = ::floor(src[0]->Data<float>()[i]);
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeFloor unsupported input type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeGather(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() < 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeGather must have 2 or more inputs and 1 output!");
        Shape idx(src[1]->Size());
        if (src[1]->GetType() == TensorType32i)
        {
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = (size_t)src[1]->Data<int32_t>()[i];
        }
        else if(src[1]->GetType() == TensorType64i)
        {
            for (size_t i = 0; i < idx.size(); ++i)
                idx[i] = (size_t)src[1]->Data<int64_t>()[i];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeFloor unsupported src[1] " << src[1]->GetType() << " type!");

        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        if (src[0]->GetType() == TensorType32f)
        {
            for (size_t i = 0; i < idx.size(); ++i)
                dst[0]->Data<float>()[i] = src[0]->Data<float>()[idx[i]];
        }
        else if (src[0]->GetType() == TensorType32i)
        {
            for (size_t i = 0; i < idx.size(); ++i)
                dst[0]->Data<int32_t>()[i] = src[0]->Data<int32_t>()[idx[i]];
        }
        else if (src[0]->GetType() == TensorType64i)
        {
            for (size_t i = 0; i < idx.size(); ++i)
                dst[0]->Data<int64_t>()[i] = src[0]->Data<int64_t>()[idx[i]];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeFloor unsupported src[0] " << src[0]->GetType() << " type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> bool ReshapeMul(const Synet::Tensor<float>& src0, const Synet::Tensor<float>& src1, Synet::Tensor<float>& dst0)
    {
        dst0.Reshape(Synet::GetTensorType<T>(), src0.Shape(), src0.Format());
        if (src0.Size() == src1.Size())
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] * src1.Data<T>()[i];
        }
        else if (src1.Size() == 1)
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] * src1.Data<T>()[0];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeMul supported input shapes!");
        return true;
    }

    bool MetaLayer::ReshapeMul(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeMul supports only 2 inputs and 1 output!");
        if (src[0]->Count() != src[1]->Count() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeMul unsupported input shape or type combination!");
        switch (src[0]->GetType())
        {
        case TensorType32f: return Synet::ReshapeMul<float>(*src[0], *src[1], *dst[0]);
        case TensorType32i: return Synet::ReshapeMul<int32_t>(*src[0], *src[1], *dst[0]);
        case TensorType64i: return Synet::ReshapeMul<int64_t>(*src[0], *src[1], *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeMul unsupported input type!");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    void MetaLayer::ReshapePack(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src[0]->GetType() == TensorType32f)
        {
            Synet::Tensor<float>& dst0 = dst[0]->As32f();
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
            Synet::Tensor<int32_t>& dst0 = dst[0]->As32i();
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
                //assert(src[i]->Count() == 1);
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

    void MetaLayer::ReshapePermute(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        assert(src[0]->Count() == alpha.i64().size());
        if (src[0]->GetType() == TensorType64i)
        {
            if (alpha.i64().size() == 2)
            {
                Shape order = Shp(alpha.i64()[0], alpha.i64()[1]), shape = src[0]->Shape();
                if (order[0] == 1 && order[1] == 0)
                {
                    dst[0]->As64i().Reshape(Shp(shape[1], shape[0]));
                    for (size_t i = 0; i < shape[0]; ++i)
                        for (size_t j = 0; j < shape[1]; ++j)
                            *dst[0]->As64i().CpuData(Shp(j, i)) = *src[0]->As64i().CpuData(Shp(i, j));
                }
                else
                {
                    dst[0]->Share(*src[0]);
                }
            }
            else
                assert(0);
        }
        else
            assert(0);
    }

    void MetaLayer::ReshapeRange(const TensorPtrs& src, const TensorPtrs& dst)
    {
        assert(src.size() == 3 && src[0]->Size() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1);
        if (src[0]->GetType() == TensorType32f)
        {
            float begin = src[0]->As32f().CpuData()[0];
            float end = src[1]->As32f().CpuData()[0];
            float step = src[2]->As32f().CpuData()[0];
            Floats result;
            if (step > 0)
                for (float i = begin; i < end; i += step)
                    result.push_back(i);
            else
                for (float i = begin; i > end; i += step)
                    result.push_back(i);
            Synet::Tensor<float>& dst0 = dst[0]->As32f();
            dst0.Reshape({ result.size() });
            for (size_t i = 0; i < result.size(); ++i)
                dst0.CpuData()[i] = result[i];
        }
        else if (src[0]->GetType() == TensorType32i)
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
            Synet::Tensor<int32_t>& dst0 = dst[0]->As32i();
            dst0.Reshape({ result.size() });
            for (size_t i = 0; i < result.size(); ++i)
                dst0.CpuData()[i] = result[i];
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
            Synet::Tensor<int64_t>& dst0 = dst[0]->As64i();
            dst0.Reshape({ result.size() });
            for (size_t i = 0; i < result.size(); ++i)
                dst0.CpuData()[i] = result[i];
        }
        else
            assert(0);
    }

    void MetaLayer::ReshapeReduceMin(const TensorPtrs& src, const TensorPtrs& dst)
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

    void MetaLayer::ReshapeReduceProd(const TensorPtrs& src, const TensorPtrs& dst)
    {
        bool keepDims = this->Param().meta().alpha().i32()[0] != 0;
        size_t axis = (size_t)src[1]->As64i().CpuData()[0];
        assert(src[0]->Count() == 1);
        int64_t prod = 1;
        for (size_t i = 0; i < src[0]->Size(); ++i)
            prod *= src[0]->As64i().CpuData()[i];
        dst[0]->As64i().Reshape(Shp(1), prod);
    }

    void MetaLayer::ReshapeReshape(const TensorPtrs& src, const TensorPtrs& dst)
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

    void MetaLayer::ReshapeSelect(const TensorPtrs& src, const TensorPtrs& dst)
    {
        assert(src.size() == 3 && src[0]->Size() == src[1]->Size() == src[2]->Size());
        if (src[0]->GetType() == TensorType64i && src[1]->GetType() == TensorType64i && src[2]->GetType() == TensorType64i)
        {
            dst[0]->As64i().Reshape(src[0]->Shape());
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->As64i().CpuData()[i] = src[0]->As64i().CpuData()[i] ?
                src[1]->As64i().CpuData()[i] : src[2]->As64i().CpuData()[i];
        }
        else
            assert(0);
    }

    void MetaLayer::ReshapeShape(const TensorPtrs& src, int version, const TensorPtrs& dst)
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

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeSlice(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src[0]->GetType() == TensorType32i)
        {
            if (!(src.size() == 3 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1))
                SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
            const Synet::Tensor<int32_t>& src0 = src[0]->As32i();
            size_t begin = src[1]->As32i().CpuData()[0];
            size_t size = src[2]->As32i().CpuData()[0];
            Synet::Tensor<int32_t>& dst0 = dst[0]->As32i();
            dst0.Reshape({ size });
            for (size_t i = 0; i < size; ++i)
                dst0.CpuData()[i] = src0.CpuData()[begin + i];
        }
        else if (src[0]->GetType() == TensorType64i)
        {
            const Synet::Tensor<int64_t>& src0 = src[0]->As64i();
            Synet::Tensor<int64_t>& dst0 = dst[0]->As64i();
            if (src[0]->Count() == 1)
            {
                if(!(src.size() == 4 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1))
                    SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
                const int64_t * src0 = src[0]->Data<int64_t>();
                int64_t size = src[0]->Size();
                int64_t beg = src[1]->Data<int64_t>()[0];
                if (beg < 0)
                    beg += size;
                int64_t end = src[2]->Data<int64_t>()[0]; 
                if (end < 0)
                    end += size;
                end = Min(end, size);
                if(beg < 0 || beg > size || end < 0 || end > size)
                    SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
                dst[0]->Reshape(src[0]->GetType(), Shp(end - beg), src[0]->Format());
                int64_t* dst0 = dst[0]->Data<int64_t>();
                for (int64_t i = beg; i < end; ++i)
                    dst0[i - beg] = src0[i];
                return true;
            }
            else if (src[0]->Count() == 2)
            {
                const Synet::Tensor<int64_t>& src0 = src[0]->As64i();
                Synet::Tensor<int64_t>& dst0 = dst[0]->As64i();
                if (!(src.size() == 5 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1 && src[4]->Size() == 1))
                    SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
                const Shape& ss = src0.Shape();
                int64_t axis = RestrictRange<int64_t>(src[3]->As64i().CpuData()[0] < 0 ? src[3]->As64i().CpuData()[0] + ss.size() : src[3]->As64i().CpuData()[0], 0, ss.size() - 1);
                int64_t beg = RestrictRange<int64_t>(src[1]->As64i().CpuData()[0] < 0 ? src[1]->As64i().CpuData()[0] + ss[axis] : src[1]->As64i().CpuData()[0], 0, ss[axis]);
                int64_t end = RestrictRange<int64_t>(src[2]->As64i().CpuData()[2] < 0 ? src[2]->As64i().CpuData()[2] + ss[axis] : src[2]->As64i().CpuData()[0], 0, ss[axis]);
                int64_t step = src[4]->As64i().CpuData()[0];
                int64_t size = 0;
                for (int64_t i = beg; step > 0 ? i < end : i >= end; i += step, size++);
                Shape ds = ss;
                ds[axis] = size;
                dst0.Reshape(ds);
                //const int64_t* s0 = src0.CpuData();
                //int64_t* d0 = dst0.CpuData();
                if (axis == 0)
                {
                    for (int64_t i = beg, o = 0; step > 0 ? i < end : i >= end; i += step, o++)
                        for (size_t j = 0; j < ss[1]; ++j)
                            *dst0.CpuData(Shp(o, j)) = *src0.CpuData(Shp(i, j));
                }
                else
                    assert(0);
            }
            else
                assert(0);
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " !");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    void MetaLayer::ReshapeSqueeze(const TensorPtrs& src, const TensorPtrs& dst)
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

    void MetaLayer::ReshapeStridedSlice(const TensorPtrs& src, const TensorPtrs& dst)
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

    void MetaLayer::ReshapeStub(const TensorPtrs& src, const TensorPtrs& dst)
    {
        assert(src.size() == 1 && dst.size() == 1);
        dst[0]->Share(*src[0]);
    }

    void MetaLayer::ReshapeSub(const TensorPtrs& src, const TensorPtrs& dst)
    {
        assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType());
        if (src[0]->GetType() == TensorType32i)
        {
            const Synet::Tensor<int32_t>& src0 = src[0]->As32i();
            const Synet::Tensor<int32_t>& src1 = src[1]->As32i();
            Synet::Tensor<int32_t>& dst0 = dst[0]->As32i();
            dst0.Reshape(src0.Shape());
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.CpuData()[i] = src0.CpuData()[i] - src1.CpuData()[i];
        }
        else
            assert(0);
    }
}
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

#include "Synet/Layers/MetaLayer.h"

namespace Synet
{
    MetaLayer::MetaLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool MetaLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (!Reshape(src, dst))
            return false;
        _const = true;
        for (size_t d = 0; d < dst.size(); ++d)
            dst[d]->SetConst(true);
        return true;
    }

    void MetaLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }

    bool MetaLayer::Reshape(const TensorPtrs& src, const TensorPtrs& dst)
    {
        const MetaParam& param = this->Param().meta();
        switch (param.type())
        {
        case MetaTypeAdd: return ReshapeAdd(src, dst);
        case MetaTypeCast: return ReshapeCast(src, param.alpha(), dst);
        case MetaTypeCeil: return ReshapeCeil(src, dst);
        case MetaTypeConst: return ReshapeConst(param.alpha(), dst);
        case MetaTypeConstantOfShape: return ReshapeConstantOfShape(src, param.alpha(), dst);
        case MetaTypeDiv: return ReshapeDiv(src, dst);
        case MetaTypeEqual: return ReshapeEqual(src, dst);
        case MetaTypeExpandDims: return ReshapeExpandDims(src, param.alpha(), dst);
        case MetaTypeFloor: return ReshapeFloor(src, dst);
        case MetaTypeGather: return ReshapeGather(src, dst);
        case MetaTypeMod: return ReshapeMod(src, dst);
        case MetaTypeMul: return ReshapeMul(src, dst);
        case MetaTypePack: return ReshapePack(src, dst);
        case MetaTypePermute: return ReshapePermute(src, param.alpha(), dst);
        case MetaTypeRange: return ReshapeRange(src, dst);
        case MetaTypeReduceMin: return ReshapeReduceMin(src, param.alpha(), dst);
        case MetaTypeReduceProd: return ReshapeReduceProd(src, param.alpha(), dst);
        case MetaTypeReshape: return ReshapeReshape(src, dst);
        case MetaTypeSelect: return ReshapeSelect(src, dst);
        case MetaTypeShape: return ReshapeShape(src, param.version(), dst);
        case MetaTypeSlice: return ReshapeSlice(src, dst);
        case MetaTypeSqueeze: return ReshapeSqueeze(src, dst);
        case MetaTypeStridedSlice: return ReshapeStridedSlice(src, dst);
        case MetaTypeStub: return ReshapeStub(src, dst);
        case MetaTypeSub: return ReshapeSub(src, dst);
        default:
            SYNET_ERROR("Unsupported meta type: " << Cpl::ToStr(param.type()) << " !");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

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

    bool MetaLayer::ReshapeCeil(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeCeil supports only 1 input and 1 output!");
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->Data<float>()[i] = ::ceil(src[0]->Data<float>()[i]);
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeCeil unsupported input type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeConst(const TensorParam& alpha, const TensorPtrs& dst)
    {
        return dst[0]->Import(alpha);
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeConstantOfShape(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeConstantOfShape supports only 1 input and 1 output!");
        if (src[0]->Count() != 1 || src[0]->GetType() != TensorType64i)
            SYNET_ERROR("MetaLayer::ReshapeConstantOfShape src[0] must be 1D 64-bit tensor!");
        if (alpha.shape() != Shp(1) || alpha.type() != TensorType64i)
            SYNET_ERROR("MetaLayer::ReshapeConstantOfShape has wrong alpha parameter!");
        Shape shape;
        for (size_t a = 0; a < src[0]->Axis(0); ++a)
            shape.push_back((size_t)(src[0]->Data<int64_t>()[a]));
        dst[0]->Reshape(TensorType64i, shape, TensorFormatUnknown, alpha.i64()[0]);
        return true;
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

        dst[0]->Reshape(src[0]->GetType(), src[1]->Shape(), src[0]->Format());
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

    template<class T> bool ReshapeMod(const Synet::Tensor<float>& src0, const Synet::Tensor<float>& src1, Synet::Tensor<float>& dst0)
    {
        dst0.Reshape(Synet::GetTensorType<T>(), src0.Shape(), src0.Format());
        if (src0.Size() == src1.Size())
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] % src1.Data<T>()[i];
        }
        else if (src1.Size() == 1)
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] % src1.Data<T>()[0];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeMod supported input shapes!");
        return true;
    }

    bool MetaLayer::ReshapeMod(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeMod supports only 2 inputs and 1 output!");
        if (src[0]->Count() != src[1]->Count() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeMod unsupported input shape or type combination!");
        switch (src[0]->GetType())
        {
        case TensorType32i: return Synet::ReshapeMod<int32_t>(*src[0], *src[1], *dst[0]);
        case TensorType64i: return Synet::ReshapeMod<int64_t>(*src[0], *src[1], *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeMod unsupported input type!");
        }
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
        if (!(src[0]->Size() == src[1]->Size() || src[1]->Size() == 1) || src[0]->GetType() != src[1]->GetType())
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

    bool MetaLayer::ReshapePack(const TensorPtrs& src, const TensorPtrs& dst)
    {
        size_t size = 0;
        for (size_t i = 0; i < src.size(); ++i)
            size += src[i]->Size();
        dst[0]->Reshape(src[0]->GetType(), Shp(size), src[0]->Format());

        if (src[0]->GetType() == TensorType32f)
        {
            for (size_t s = 0, d = 0; s < src.size(); ++s)
            {
                if(src[s]->GetType() != TensorType32f)
                    SYNET_ERROR("MetaLayer::ReshapePack has incompatible input types!");
                for (size_t i = 0; i < src[s]->Size(); ++i, ++d)
                    dst[0]->Data<float>()[d] = src[s]->Data<float>()[i];
            }
        }
        else if (src[0]->GetType() == TensorType32i)
        {
            for (size_t s = 0, d = 0; s < src.size(); ++s)
            {
                if (src[s]->GetType() == TensorType32i)
                {
                    for (size_t i = 0; i < src[s]->Size(); ++i, ++d)
                        dst[0]->Data<int32_t>()[d] = src[s]->Data<int32_t>()[i];
                }
                else if (src[s]->GetType() == TensorType64i)
                {
                    for (size_t i = 0; i < src[s]->Size(); ++i, ++d)
                        dst[0]->Data<int32_t>()[d] = (int32_t)src[s]->Data<int64_t>()[i];
                }
                else
                    SYNET_ERROR("MetaLayer::ReshapePack has incompatible input types!");
            }
        }
        else if (src[0]->GetType() == TensorType64i)
        {
            for (size_t s = 0, d = 0; s < src.size(); ++s)
            {
                if (src[s]->GetType() == TensorType32i)
                {
                    for (size_t i = 0; i < src[s]->Size(); ++i, ++d)
                        dst[0]->Data<int64_t>()[d] = (int64_t)src[s]->Data<int32_t>()[i];
                }
                else if (src[s]->GetType() == TensorType64i)
                {
                    for (size_t i = 0; i < src[s]->Size(); ++i, ++d)
                        dst[0]->Data<int64_t>()[d] = src[s]->Data<int64_t>()[i];
                }
                else
                    SYNET_ERROR("MetaLayer::ReshapePack has incompatible input types!");
            }
        }
        else
            SYNET_ERROR("MetaLayer::ReshapePack has incompatible input types!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapePermute(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapePermute supports only 1 input and 1 output!");
        if(src[0]->Count() != alpha.i64().size())
            SYNET_ERROR("MetaLayer::ReshapePermute input and alpha parameter are incompatible!");
        if (src[0]->GetType() == TensorType64i)
        {
            if (alpha.i64().size() == 2)
            {
                Shape order = Shp(alpha.i64()[0], alpha.i64()[1]), shape = src[0]->Shape();
                if (order[0] == 1 && order[1] == 0)
                {
                    dst[0]->Reshape(src[0]->GetType(), Shp(shape[1], shape[0]), TensorFormatUnknown);
                    for (size_t i = 0; i < shape[0]; ++i)
                        for (size_t j = 0; j < shape[1]; ++j)
                            *dst[0]->Data<int64_t>(Shp(j, i)) = *src[0]->Data<int64_t>(Shp(i, j));
                }
                else
                    dst[0]->Share(*src[0]);
            }
            else
                SYNET_ERROR("MetaLayer::ReshapePermute unsupported input shape!");
        }
        else
            SYNET_ERROR("MetaLayer::ReshapePermute unsupported input type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> void ReshapeRange(const std::vector<Tensor<float>*> & src, const std::vector<Tensor<float>*> dst)
    {
        T begin = src[0]->Data<T>()[0];
        T end = src[1]->Data<T>()[0];
        T step = src[2]->Data<T>()[0];
        std::vector<T> result;
        if (step > 0)
            for (T i = begin; i < end; i += step)
                result.push_back(i);
        else
            for (T i = begin; i > end; i += step)
                result.push_back(i);
        dst[0]->Reshape(GetTensorType<T>(), Shp(result.size()), TensorFormatUnknown);
        for (size_t i = 0; i < result.size(); ++i)
            dst[0]->Data<T>()[i] = result[i];
    }

    bool MetaLayer::ReshapeRange(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeRange supports only 3 inputs and 1 output!");
        if(src[0]->Size() != 1 || src[1]->Size() != 1 || src[2]->Size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeRange unsupported input shape!");
        if (src[0]->GetType() != src[1]->GetType() || src[0]->GetType() != src[2]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeRange has incompatible input types!");
        switch (src[0]->GetType())
        {
        case TensorType32f: Synet::ReshapeRange<float>(src, dst); break;
        case TensorType32i: Synet::ReshapeRange<int32_t>(src, dst); break;
        case TensorType64i: Synet::ReshapeRange<int64_t>(src, dst); break;
        default:
            SYNET_ERROR("MetaLayer::ReshapeRange unsupported input type!");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeReduceMin(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReduceMin supports only 2 inputs and 1 output!");
        if (src[1]->GetType() != TensorType64i)
            SYNET_ERROR("MetaLayer::ReshapeReduceMin has unsupported src[1] type!");
        size_t axis = (size_t)src[1]->Data<int64_t>()[0];

        if (alpha.shape() != Shp(1) || alpha.type() != TensorType32i)
            SYNET_ERROR("MetaLayer::ReshapeReduceMin has wrong alpha parameter!");
        bool keepDims = alpha.i32()[0] != 0;

        if(src[0]->Count() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReduceMin has src[0] unsupported shape!");
        if (src[0]->GetType() == TensorType32i)
        {
            int32_t min = INT_MAX;
            for (size_t i = 0; i < src[0]->Size(); ++i)
                min = Min(min, src[0]->Data<int32_t>()[i]);
            dst[0]->Reshape(TensorType64i, Shp(1), TensorFormatUnknown, min);
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeReduceMin has unsupported src[0] type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeReduceProd(const TensorPtrs& src, const TensorParam& alpha, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReduceProd supports only 2 inputs and 1 output!");
        if (src[1]->GetType() != TensorType64i)
            SYNET_ERROR("MetaLayer::ReshapeReduceprod has unsupported src[1] type!");
        size_t axis = (size_t)src[1]->Data<int64_t>()[0];

        if (alpha.shape() != Shp(1) || alpha.type() != TensorType32i)
            SYNET_ERROR("MetaLayer::ReshapeReduceProd has wrong alpha parameter!");
        bool keepDims = alpha.i32()[0] != 0;

        if (src[0]->Count() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReduceProd has src[0] unsupported shape!");
        if (src[0]->GetType() == TensorType64i)
        {
            int64_t prod = 1;
            for (size_t i = 0; i < src[0]->Size(); ++i)
                prod *= src[0]->Data<int64_t>()[i];
            dst[0]->Reshape(TensorType64i, Shp(1), TensorFormatUnknown, prod);
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeReduceProd has unsupported src[0] type!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeReshape(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReshape supports only 2 inputs and 1 output!");
        if (src[1]->Count() != 1)
            SYNET_ERROR("MetaLayer::ReshapeReduceProd has src[1] unsupported shape!");

        Shape shape(src[1]->Size());
        if (src[1]->GetType() == TensorType32i)
        {
            for (size_t i = 0; i < shape.size(); ++i)
                shape[i] = (size_t)src[1]->Data<int32_t>()[i];
        }
        else if (src[1]->GetType() == TensorType64i)
        {
            for (size_t i = 0; i < shape.size(); ++i)
                shape[i] = (size_t)src[1]->Data<int64_t>()[i];

        }
        else
            SYNET_ERROR("MetaLayer::ReshapeReshape has unsupported src[1] type!");
        size_t unknown = 0;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == -1)
                unknown++;
        }
        if(unknown > 1)
            SYNET_ERROR("MetaLayer::ReshapeReshape src[1] has more then 1 unknown dimensions!");
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
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeSelect(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeSelect supports only 3 inputs and 1 output!");
        if (src[0]->Shape() != src[1]->Shape() || src[0]->Shape() != src[2]->Shape())
            SYNET_ERROR("MetaLayer::ReshapeSelect has incompatible input shapes!");
        if (src[0]->GetType() != src[1]->GetType() || src[0]->GetType() != src[2]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeSelect has incompatible input types!");
        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        switch (src[0]->GetType())
        {
        case TensorType64i:
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->Data<int64_t>()[i] = src[0]->Data<int64_t>()[i] ?
                src[1]->Data<int64_t>()[i] : src[2]->Data<int64_t>()[i];
            break;
        default:
            SYNET_ERROR("MetaLayer::ReshapeSelect has unsupported input type!");
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeShape(const TensorPtrs& src, int version, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeShape supports only 1 input and 1 output!");
        Shape shape = src[0]->Shape();
        bool trans = src[0]->Format() == TensorFormatNhwc;
        if(version == 1)
        {
            if (trans && shape.size() == 4)
                shape = Shp(shape[0], shape[3], shape[1], shape[2]);
            dst[0]->Reshape(TensorType64i, Shp(shape.size()), src[0]->Format());
            for (size_t i = 0; i < shape.size(); ++i)
                dst[0]->Data<int64_t>()[i] = (int64_t)shape[i];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeShape unsupported version!");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeSlice(const TensorPtrs& src, const TensorPtrs& dst)
    {
        for(size_t s = 1; s < src.size(); ++s)
            if (src[0]->GetType() != src[s]->GetType())
                SYNET_ERROR("MetaLayer::ReshapeSlice has incompatible input types!");
        if (src[0]->GetType() == TensorType32i)
        {
            if (!(src.size() == 3 && src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1))
                SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
            size_t begin = src[1]->Data<int32_t>()[0];
            size_t size = src[2]->Data<int32_t>()[0];
            dst[0]->Reshape(src[0]->GetType(), Shp(size), src[0]->Format());
            for (size_t i = 0; i < size; ++i)
                dst[0]->Data<int32_t>()[i] = src[0]->Data<int32_t>()[begin + i];
        }
        else if (src[0]->GetType() == TensorType64i)
        {
            if (src[0]->Count() == 1)
            {
                if(!(src.size() == 4 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1))
                    SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
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
                for (int64_t i = beg; i < end; ++i)
                    dst[0]->Data<int64_t>()[i - beg] = src[0]->Data<int64_t>()[i];
            }
            else if (src[0]->Count() == 2)
            {
                if (!(src.size() == 5 && src[1]->Size() == 1 && src[2]->Size() == 1 && src[3]->Size() == 1 && src[4]->Size() == 1))
                    SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
                const Shape& ss = src[0]->Shape();
                int64_t axis = RestrictRange<int64_t>(src[3]->Data<int64_t>()[0] < 0 ? src[3]->Data<int64_t>()[0] + ss.size() : src[3]->Data<int64_t>()[0], 0, ss.size() - 1);
                int64_t beg = RestrictRange<int64_t>(src[1]->Data<int64_t>()[0] < 0 ? src[1]->Data<int64_t>()[0] + ss[axis] : src[1]->Data<int64_t>()[0], 0, ss[axis]);
                int64_t end = RestrictRange<int64_t>(src[2]->Data<int64_t>()[2] < 0 ? src[2]->Data<int64_t>()[2] + ss[axis] : src[2]->Data<int64_t>()[0], 0, ss[axis]);
                int64_t step = src[4]->Data<int64_t>()[0];
                int64_t size = 0;
                for (int64_t i = beg; step > 0 ? i < end : i >= end; i += step, size++);
                Shape ds = ss;
                ds[axis] = size;
                dst[0]->Reshape(src[0]->GetType(), ds, src[0]->Format());
                if (axis == 0)
                {
                    for (int64_t i = beg, o = 0; step > 0 ? i < end : i >= end; i += step, o++)
                        for (size_t j = 0; j < ss[1]; ++j)
                            *dst[0]->Data<int64_t>(Shp(o, j)) = *src[0]->Data<int64_t>(Shp(i, j));
                }
                else
                    SYNET_ERROR("MetaLayer::ReshapeSlice: wrong src[3] value!");
            }
            else
                SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " !");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeSqueeze(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeSqueeze supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeSqueeze has incompatible input types!");
        if (src[0]->GetType() == TensorType64i)
        {
            if(!(src[0]->Size() == 1 && src[1]->Size() == 1))
                SYNET_ERROR("MetaLayer::ReshapeSlice: unsupported input shapes!");
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
            for (size_t i = 0; i < src[0]->Size(); ++i)
                dst[0]->Data<int64_t>()[i] = src[0]->Data<int64_t>()[i];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeSqueeze: unsupported src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " !");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeStridedSlice(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() < 3 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeStridedSlice supports only 3 or 4 inputs and 1 output!");
        if (!(src[0]->Count() == 1 && src[1]->Size() == 1 && src[2]->Size() == 1))
            SYNET_ERROR("MetaLayer::ReshapeStridedSlice has incompatible input shapes!");
        size_t begin = GetAs<size_t>(*src[1], 0);
        size_t end = GetAs<size_t>(*src[2], 0);
        size_t step = src.size() > 3 ? GetAs<size_t>(*src[3], 0) : size_t(1);
        Shape result;
        for (size_t i = begin; i < end; i += step)
            result.push_back(GetAs<size_t>(*src[0], i));
        dst[0]->Reshape(src[0]->GetType(), Shp(result.size()), src[0]->Format());
        if (src[0]->GetType() == TensorType32i)
        {
            for (size_t i = 0; i < result.size(); ++i)
                dst[0]->Data<int32_t>()[i] = (int32_t)result[i];
        }
        else if (src[0]->GetType() == TensorType64i)
        {
            for (size_t i = 0; i < result.size(); ++i)
                dst[0]->Data<int64_t>()[i] = (int64_t)result[i];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeStridedSlice: unsupported src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " !");
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    bool MetaLayer::ReshapeStub(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeStub supports only 1 input and 1 output!");
        dst[0]->Share(*src[0]);
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> bool ReshapeSub(const Synet::Tensor<float>& src0, const Synet::Tensor<float>& src1, Synet::Tensor<float>& dst0)
    {
        dst0.Reshape(Synet::GetTensorType<T>(), src0.Shape(), src0.Format());
        if (src0.Size() == src1.Size())
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] - src1.Data<T>()[i];
        }
        else if (src1.Size() == 1)
        {
            for (size_t i = 0; i < src0.Size(); ++i)
                dst0.Data<T>()[i] = src0.Data<T>()[i] - src1.Data<T>()[0];
        }
        else
            SYNET_ERROR("MetaLayer::ReshapeSub supported input shapes!");
        return true;
    }

    bool MetaLayer::ReshapeSub(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MetaLayer::ReshapeSub supports only 2 inputs and 1 output!");
        if (src[0]->Count() != src[1]->Count() || src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("MetaLayer::ReshapeSub unsupported input shape or type combination!");
        switch (src[0]->GetType())
        {
        case TensorType32f: return Synet::ReshapeSub<float>(*src[0], *src[1], *dst[0]);
        case TensorType32i: return Synet::ReshapeSub<int32_t>(*src[0], *src[1], *dst[0]);
        case TensorType64i: return Synet::ReshapeSub<int64_t>(*src[0], *src[1], *dst[0]);
        default:
            SYNET_ERROR("MetaLayer::ReshapeSub unsupported input type!");
        }
        return true;
    }
}
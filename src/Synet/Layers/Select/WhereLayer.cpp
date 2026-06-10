/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Select/WhereLayer.h"
#include "Synet/Utils/Math.h"
#include "Synet/Utils/UniversalBinary.h"

namespace Synet
{
    template <class C, class D> void WhereUniform(const uint8_t* cnd8, const uint8_t* pos8, const uint8_t* neg8, size_t size, uint8_t* dst8)
    {
        const C* cnd = (const C*)cnd8;
        const D* pos = (const D*)pos8;
        const D* neg = (const D*)neg8;
        D* dst = (D*)dst8;
        for (size_t i = 0; i < size; ++i)
            dst[i] = cnd[i] ? pos[i] : neg[i];
    }

    //-------------------------------------------------------------------------------------------------

    template<class C> WhereLayer::WhereUniformPtr GetWhereUniform(TensorType dst)
    {
        switch (dst)
        {
        case TensorType32f: return WhereUniform<C, float>;
        case TensorType32i: return WhereUniform<C, uint32_t>;
        case TensorType64i: return WhereUniform<C, uint64_t>;
        default:
            return NULL;
        }
    }

    WhereLayer::WhereUniformPtr GetWhereUniform(TensorType cnd, TensorType dst)
    {
        switch (cnd)
        {
        case TensorType32f: return GetWhereUniform<uint32_t>(dst);
        case TensorType32i: return GetWhereUniform<uint32_t>(dst);
        case TensorType64i: return GetWhereUniform<uint64_t>(dst);
        case TensorTypeBool: return GetWhereUniform<bool>(dst);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class C, class D, size_t N> void WhereUniversal(const uint8_t* cnd8, const Shape& cndSteps, 
        const uint8_t* pos8, const Shape& posSteps, const uint8_t* neg8, const Shape& negSteps, uint8_t* dst8, const Shape& dstShape)
    {
        const C* cnd = (const C*)cnd8;
        const D* pos = (const D*)pos8;
        const D* neg = (const D*)neg8;
        D* dst = (D*)dst8;
        if (N == 1)
        {
            const C* c0 = cnd;
            const D* p0 = pos, * n0 = neg;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                *dst++ = *c0 ? *p0 : *n0;
                c0 += cndSteps[0];
                p0 += posSteps[0];
                n0 += negSteps[0];
            }
        }
        else if (N == 2)
        {
            const C* c0 = cnd;
            const D* p0 = pos, * n0 = neg;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const C* c1 = c0;
                const D* p1 = p0, * n1 = n0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    *dst++ = *c1 ? *p1 : *n1;
                    c1 += cndSteps[1];
                    p1 += posSteps[1];
                    n1 += negSteps[1];
                }
                c0 += cndSteps[0];
                p0 += posSteps[0];
                n0 += negSteps[0];
            }
        }
        else if (N == 3)
        {
            const C* c0 = cnd;
            const D* p0 = pos, * n0 = neg;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const C* c1 = c0;
                const D* p1 = p0, * n1 = n0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const C* c2 = c1;
                    const D* p2 = p1, * n2 = n1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        *dst++ = *c2 ? *p2 : *n2;
                        c2 += cndSteps[2];
                        p2 += posSteps[2];
                        n2 += negSteps[2];
                    }
                    c1 += cndSteps[1];
                    p1 += posSteps[1];
                    n1 += negSteps[1];
                }
                c0 += cndSteps[0];
                p0 += posSteps[0];
                n0 += negSteps[0];
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    template<class C, class D> WhereLayer::WhereUniversalPtr GetWhereUniversal(size_t dim)
    {
        switch (dim)
        {
        case 1: return WhereUniversal<C, D, 1>;
        case 2: return WhereUniversal<C, D, 2>;
        case 3: return WhereUniversal<C, D, 3>;
        default:
            SYNET_ERROR("WhereUniversal does not support " << dim << "D algorithm!");
        }
    }

    template<class C> WhereLayer::WhereUniversalPtr GetWhereUniversal(TensorType dst, size_t dim)
    {
        switch (dst)
        {
        case TensorType32f: return GetWhereUniversal<C, float>(dim);
        case TensorType32i: return GetWhereUniversal<C, int32_t>(dim);
        case TensorType64i: return GetWhereUniversal<C, int64_t>(dim);
        default:
            return NULL;
        }
    }

    WhereLayer::WhereUniversalPtr GetWhereUniversal(TensorType cnd, TensorType dst, size_t dim)
    {
        switch (cnd)
        {
        case TensorType32f: return GetWhereUniversal<uint32_t>(dst, dim);
        case TensorType32i: return GetWhereUniversal<uint32_t>(dst, dim);
        case TensorType64i: return GetWhereUniversal<uint64_t>(dst, dim);
        case TensorTypeBool: return GetWhereUniversal<bool>(dst, dim);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    WhereLayer::WhereLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _whereUniform(NULL)
        , _whereUniversal(NULL)
    {
    }

    SYNET_INLINE bool GetSteps(const Shape& src, const Shape& dst, Shape& steps)
    {
        steps.resize(src.size(), 0);
        size_t step = 1;
        for (ptrdiff_t i = src.size() - 1; i >= 0; --i)
        {
            if (src[i] != dst[i] && src[i] != 1)
                return false;
            steps[i] = src[i] == 1 ? 0 : step;
            step *= src[i];
        }
        return true;
    }

    bool WhereLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("WhereLayer supports only 3 inputs and 1 output!");
        if( src[1]->GetType() != src[2]->GetType())
            SYNET_ERROR("WhereLayer src[1] type " << Cpl::ToStr(src[1]->GetType()) << " and src[2] type " << Cpl::ToStr(src[2]->GetType()) << " are different!");

        _cndType = src[0]->GetType();
        _dstType = src[1]->GetType();
        if (src[0]->Shape() == src[1]->Shape() && src[0]->Shape() == src[2]->Shape())
        {
            _size = src[0]->Size();
            dst[0]->Reshape(_dstType, src[0]->Shape(), src[0]->Format());
            _whereUniform = GetWhereUniform(_cndType, _dstType);
            if(_whereUniform == NULL)
                SYNET_ERROR("WhereLayer can't get uniform worker when src[0] " << Cpl::ToStr(_cndType) << " and src[1] " << Cpl::ToStr(_dstType) << " !");
        }
        else
        {
            Shape cndShape = src[0]->Shape(), posShape = src[1]->Shape(), negShape = src[2]->Shape();
            if(!IsCompatible(cndShape, posShape) || !IsCompatible(cndShape, negShape))
                SYNET_ERROR("WhereLayer incompatible input shapes!");

            Shape dstShape = OutputShape(OutputShape(cndShape, posShape), negShape);
            _dstShape = dstShape;
            CompactShapes(cndShape, posShape, negShape, _dstShape);
            if (!(GetSteps(cndShape, _dstShape, _cndSteps) &&
                GetSteps(posShape, _dstShape, _posSteps) &&
                GetSteps(negShape, _dstShape, _negSteps)))
                SYNET_ERROR("WhereLayer has incompatible inputs!");
            dst[0]->Reshape(_dstType, dstShape, src[0]->Format());
            _whereUniversal = GetWhereUniversal(_cndType, _dstType, _dstShape.size());
            if (_whereUniversal == NULL)
                SYNET_ERROR("WhereLayer can't work when src[0] " << Cpl::ToStr(_cndType) << " and src[1] " << Cpl::ToStr(_dstType) << " and dst[0] " << ToStr(_dstShape) << " !");
        }

        if (src[0]->Const() && src[1]->Const() && src[2]->Const())
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

    void WhereLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if (_whereUniform)
            _whereUniform(src[0]->RawData(), src[1]->RawData(), src[2]->RawData(), _size, dst[0]->RawData());
        else
            _whereUniversal(src[0]->RawData(), _cndSteps, src[1]->RawData(), _posSteps, src[2]->RawData(), _negSteps, dst[0]->RawData(), _dstShape);
    }
}
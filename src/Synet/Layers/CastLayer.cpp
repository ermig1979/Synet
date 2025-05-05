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

#include "Synet/Layers/CastLayer.h"

namespace Synet
{
    template <class S, class D> void Cast(const uint8_t* src8, size_t size, uint8_t* dst8)
    {
        const S* src = (const S*)src8;
        D* dst = (D*)dst8;
        for (size_t i = 0; i < size; ++i)
            dst[i] = (D)src[i];
    }

    //-------------------------------------------------------------------------------------------------

    template<class S> CastLayer::CastPtr GetCast(TensorType dst)
    {
        switch (dst)
        {
        case TensorType32f: return Cast<S, float>;
        case TensorType32i: return Cast<S, int32_t>;
        case TensorType64i: return Cast<S, int64_t>;
        default:
            return NULL;
        }
    }

    CastLayer::CastPtr GetCast(TensorType src, TensorType dst)
    {
        switch (src)
        {
        case TensorType32f: return GetCast<float>(dst);
        case TensorType32i: return GetCast<int32_t>(dst);
        case TensorType8u: return GetCast<uint8_t>(dst);
        case TensorType64i: return GetCast<int64_t>(dst);
        case TensorTypeBool: return GetCast<bool>(dst);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    CastLayer::CastLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool CastLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("CastLayer supports only 1 input and 1 output!");
        _srcType = src[0]->GetType();
        _dstType = this->Param().cast().type();
        _size = src[0]->Size();
        _cast = GetCast(_srcType, _dstType);
        if (_cast == NULL)
            SYNET_ERROR("CastLayer can't cast " << Cpl::ToStr(_srcType) << " to " << Cpl::ToStr(_dstType) << " !");
        if (_srcType == _dstType)
        {
            dst[0]->Share(*src[0]);
            _const = true;
        }
        else
        {
            dst[0]->Reshape(_dstType, src[0]->Shape(), src[0]->Format());
            if (src[0]->Const())
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
        }
        return true;
    }

    int64_t CastLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size;
    }

    void CastLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _cast(src[0]->RawData(), _size, dst[0]->RawData());
    }
}
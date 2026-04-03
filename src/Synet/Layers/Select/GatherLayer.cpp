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

#include "Synet/Layers/Select/GatherLayer.h"

namespace Synet
{
    template <class T, class I> void Gather(const uint8_t* src8, size_t srcOuter, size_t srcCount, size_t srcInner, const uint8_t* idx8, size_t idxOuter, size_t idxCount, uint8_t* dst8)    
    {
        const T* src = (const T*)src8;
        const I* idx = (I*)idx8;
        T* dst = (T*)dst8;
        if (srcInner == 1)
        {
            for (size_t so = 0; so < srcOuter; ++so)
            {
                const I* pi = idx;
                for (size_t io = 0; io < idxOuter; ++io)
                {
                    for (size_t ic = 0; ic < idxCount; ++ic)
                    {
                        I sc = pi[ic];
                        if (sc < 0)
                            sc += I(srcCount);
                        dst[ic] = src[sc];
                    }
                    pi += idxCount;
                    dst += idxCount;
                }
                src += srcCount * srcInner;
            }
        }
        else if(srcInner < 32)
        {
            for (size_t so = 0; so < srcOuter; ++so)
            {
                const I* pi = idx;
                for (size_t io = 0; io < idxOuter; ++io)
                {
                    for (size_t ic = 0; ic < idxCount; ++ic)
                    {
                        I sc = pi[ic];
                        if (sc < 0)
                            sc += I(srcCount);
                        const T* ps = src + sc * srcInner;
                        for (size_t si = 0; si < srcInner; ++si)
                            dst[si] = ps[si];
                        dst += srcInner;
                    }
                    pi += idxCount;
                }
                src += srcCount * srcInner;
            }
        }
        else
        {
            for (size_t so = 0; so < srcOuter; ++so)
            {
                const I* pi = idx;
                for (size_t io = 0; io < idxOuter; ++io)
                {
                    for (size_t ic = 0; ic < idxCount; ++ic)
                    {
                        I sc = pi[ic];
                        if (sc < 0)
                            sc += I(srcCount);
                        memcpy(dst, src + sc * srcInner, srcInner * sizeof(T));
                        dst += srcInner;
                    }
                    pi += idxCount;
                }
                src += srcCount * srcInner;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> GatherLayer::GatherPtr GetGather(TensorType idx)
    {
        switch (idx)
        {
        case TensorType32i: return Gather<T, int32_t>;
        case TensorType64i: return Gather<T, int64_t>;
        default:
            return NULL;
        }
    }

    GatherLayer::GatherPtr GetGather(TensorType src, TensorType idx)
    {
        switch (src)
        {
        case TensorType32f: return GetGather<float>(idx);
        case TensorType32i: return GetGather<int32_t>(idx);
        case TensorType64i: return GetGather<int64_t>(idx);
        case TensorType16b: return GetGather<uint16_t>(idx);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T, class I, int check> void GatherElementsImp(const uint8_t* src8, size_t srcOuter, size_t srcCount, size_t srcInner, const uint8_t* idx8, size_t idxCount, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        const I* idx = (I*)idx8;
        T* dst = (T*)dst8;
        if (srcInner == 1)
        {
            for (size_t o = 0; o < srcOuter; ++o)
            {
                for (size_t c = 0; c < idxCount; ++c)
                {
                    I ic = idx[c];
                    if (check)
                    {
                        if (ic < 0)
                            ic += I(srcCount);
                    }
                    dst[c] = src[ic];
                }
                src += srcCount;
                idx += idxCount;
                dst += idxCount;
            }
        }
        else
        {
            for (size_t o = 0; o < srcOuter; ++o)
            {
                for (size_t c = 0; c < idxCount; ++c)
                {
                    for (size_t i = 0; i < srcInner; ++i)
                    {
                        I ii = idx[i];
                        if (check)
                        {
                            if (ii < 0)
                                ii += I(srcCount);
                        }
                        dst[i] = src[ii * srcInner + i];
                    }
                    idx += srcInner;
                    dst += srcInner;
                }
                src += srcCount * srcInner;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> GatherLayer::GatherElementsPtr GetGatherElements(TensorType idx, bool check)
    {
        switch (idx)
        {
        case TensorType32i: return check ? GatherElementsImp<T, int32_t, 1> : GatherElementsImp<T, int32_t, 0>;
        case TensorType64i: return check ? GatherElementsImp<T, int64_t, 1> : GatherElementsImp<T, int64_t, 0>;
        default:
            return NULL;
        }
    }

    GatherLayer::GatherElementsPtr GetGatherElements(TensorType src, TensorType idx, bool check)
    {
        switch (src)
        {
        case TensorType32f: return GetGatherElements<float>(idx, check);
        case TensorType32i: return GetGatherElements<int32_t>(idx, check);
        case TensorType64i: return GetGatherElements<int64_t>(idx, check);
        case TensorType16b: return GetGatherElements<uint16_t>(idx, check);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    bool NeedCheck(const uint8_t* idx8, size_t size, TensorType type)
    {
        if (type == TensorType32i)
        {
            const int32_t * idx = (int32_t*)idx8;
            for (size_t i = 0; i < size; ++i)
                if (idx[i] < 0)
                    return true;
            return false;
        }
        else if (type == TensorType64i)
        {
            const int64_t* idx = (int64_t*)idx8;
            for (size_t i = 0; i < size; ++i)
                if (idx[i] < 0)
                    return true;
            return false;
        }
        return true;
    }

    //-------------------------------------------------------------------------------------------------

    GatherLayer::GatherLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType GatherLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    bool GatherLayer::CanIgnoreInSubGraph(size_t index, bool src) const
    {
        if (src && index)
            return true;
        return false;
    }

    size_t GatherLayer::MemoryUsage() const
    {
        return _gatherElementsSimd.InternalBufferSize();
    }

    bool GatherLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("GatherLayer supports only 2 inputs and 1 output!");
        const Tensor& data = *src[0];
        const Tensor& index = *src[1];

        _srcType = data.GetType();
        if (_srcType != TensorType32f && _srcType != TensorType32i && _srcType != TensorType64i && _srcType != TensorType16b)
            SYNET_ERROR("GatherLayer has wrong src[0] type: " << Cpl::ToStr(_srcType) << " !");

        _idxType = index.GetType();
        if (_idxType != TensorType32i && _idxType != TensorType64i)
            SYNET_ERROR("GatherLayer has wrong src[1] type: " << Cpl::ToStr(_idxType) << " !");

        Shape srcShape = data.Shape();
        Shape idxShape = index.Shape();

        const GatherParam& gather = this->Param().gather();
        _axis = src[0]->Index(gather.axis());
        if(_axis >= srcShape.size())
            SYNET_ERROR("GatherLayer parameter axis: " << _axis << " has wrong value for input " << ToStr(srcShape) << " !");
        _version = gather.version();

        _srcOuter = src[0]->Size(0, _axis);
        _srcCount = src[0]->Axis(_axis);
        _srcInner = src[0]->Size(_axis + 1);

        Shape dstShape;
        if (_version == 0)
        {
            _gather = GetGather(_srcType, _idxType);
            if (_gather == NULL)
                SYNET_ERROR("GatherLayer can't get 'gather' worker!");

            for (size_t i = 0; i < _axis; ++i)
                dstShape.push_back(srcShape[i]);
            if (src[1]->Size() > 1)// && !src[1]->Const())
            {
                for (size_t i = 0; i < idxShape.size(); ++i)
                    dstShape.push_back(idxShape[i]);
            }
            for (size_t i = _axis + idxShape.size(); i < srcShape.size(); ++i)
                dstShape.push_back(srcShape[i]);
            if (srcShape.back() == 1 && dstShape.size() < srcShape.size() + idxShape.size() - 1)
                dstShape.push_back(1);

            _idxOuter = src[1]->Size(0, -1);
            _idxCount = src[1]->Size(-1);
        }
        else if(_version == 1)
        {
            if (srcShape.size() != idxShape.size())
                SYNET_ERROR("GatherLayer (version=1) inputs are incompatible!");

            bool check = !src[1]->Const() || NeedCheck(src[1]->RawData(), src[1]->Size(), _idxType);
            _gatherElements = GetGatherElements(_srcType, _idxType, check);
            if (_gatherElements == NULL)
                SYNET_ERROR("GatherLayer can't get 'gatherElements' worker!");

            dstShape = idxShape;

            _idxOuter = src[1]->Size(0, _axis);
            _idxCount = src[1]->Axis(_axis);
            _idxInner = src[1]->Size(_axis + 1);

            _gatherElementsSimd.Init(_srcType, _idxType, src[1]->Const(), 
                TensorUsers(Param().src()[1]) == 1, idxShape.data(), _axis, _srcCount, _srcInner, _idxCount);
            if (_gatherElementsSimd.Enable() && src[1]->Const())
                _gatherElementsSimd.SetIndex(src[1]->RawData());
        }
        else
            SYNET_ERROR("GatherLayer parameter version: " << _version << " is unsupported!");

        dst[0]->Reshape(_srcType, dstShape, src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            std::stringstream desc;
            desc << "v" << _version << ToChar(_srcType);
            this->UsePerfStat(desc.str());
            _const = false;
        }
        return true;
    }

    void GatherLayer::Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
        switch (_version)
        {
        case 0:
            _gather(src[0]->RawData(), _srcOuter, _srcCount, _srcInner, src[1]->RawData(), _idxOuter, _idxCount, dst[0]->RawData());
            break;
        case 1:
            if (_gatherElementsSimd.Enable())
                _gatherElementsSimd.Forward(src[0]->RawData(), src[1]->RawData(), dst[0]->RawData());
            else
                _gatherElements(src[0]->RawData(), _srcOuter, _srcCount, _srcInner, src[1]->RawData(), _idxCount, dst[0]->RawData());
            break;
        default:
            assert(0);
        }
    }
}
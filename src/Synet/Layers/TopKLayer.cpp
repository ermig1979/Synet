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
#include "Synet/Layers/TopKLayer.h"

#include "Synet/Utils/Math.h"

namespace Synet
{
    template <TopKMode M, TopKSort S> struct Comparer
    {
        template <class T, class I> static bool Compare(const std::pair<T, I>& a, const std::pair<T, I>& b);
    };

    template <> struct Comparer<TopKModeMax, TopKSortValue>
    {
        template <class T, class I> static bool Compare(const std::pair<T, I>& a, const std::pair<T, I>& b)
        {
            return a.first > b.first;
        }
    };

    template <> struct Comparer<TopKModeMin, TopKSortValue>
    {
        template <class T, class I> static bool Compare(const std::pair<T, I>& a, const std::pair<T, I>& b)
        {
            return a.first < b.first;
        }
    };

    template <> struct Comparer<TopKModeMax, TopKSortIndex>
    {
        template <class T, class I> static bool Compare(const std::pair<T, I>& a, const std::pair<T, I>& b)
        {
            return a.second > b.second;
        }
    };

    template <> struct Comparer<TopKModeMin, TopKSortIndex>
    {
        template <class T, class I> static bool Compare(const std::pair<T, I>& a, const std::pair<T, I>& b)
        {
            return a.second < b.second;
        }
    };

    //-------------------------------------------------------------------------------------------------


    template <class T, class I, TopKMode mode, TopKSort sort> void TopK(const uint8_t * src8, size_t outer, size_t count, size_t inner, size_t k, uint8_t* buf8, uint8_t* dst8, uint8_t* idx8)
    {
        typedef std::pair<T, I> B;
        const T* src = (const T*)src8;
        B * buf = (B*)buf8;
        T* dst = (T*)dst8;
        I* idx = (I*)idx8;
        for (size_t o = 0; o < outer; ++o)
        {
            for (size_t i = 0; i < inner; ++i)
            {
                for (size_t c = 0; c < count; ++c)
                {
                    buf[c].first = src[c * inner + i];
                    buf[c].second = I(c);
                }
                std::sort(buf, buf + count, Comparer<mode, sort>::template Compare<T, I>);
                for (size_t c = 0; c < k; ++c)
                {
                    dst[c * inner + i] = buf[c].first;
                    idx[c * inner + i] = buf[c].second;
                }
            }
            src += count * inner;
            dst += k * inner;
            idx += k * inner;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class S, class I, TopKMode mode> TopKLayer::TopKPtr GetTopK(TopKSort sort)
    {
        switch (sort)
        {
        case TopKSortValue: return TopK<S, I, mode, TopKSortValue>;
        case TopKSortIndex: return TopK<S, I, mode, TopKSortIndex>;
        default:
            return NULL;
        }
    }

    template<class S, class I> TopKLayer::TopKPtr GetTopK(TopKMode mode, TopKSort sort)
    {
        switch (mode)
        {
        case TopKModeMax: return GetTopK<S, I, TopKModeMax>(sort);
        case TopKModeMin: return GetTopK<S, I, TopKModeMin>(sort);
        default:
            return NULL;
        }
    }

    template<class S> TopKLayer::TopKPtr GetTopK(TensorType idx, TopKMode mode, TopKSort sort)
    {
        switch (idx)
        {
        case TensorType32i: return GetTopK<S, uint32_t>(mode, sort);
        case TensorType64i: return GetTopK<S, uint64_t>(mode, sort);
        default:
            return NULL;
        }
    }

    TopKLayer::TopKPtr GetTopK(TensorType src, TensorType idx, TopKMode mode, TopKSort sort)
    {
        switch (src)
        {
        case TensorType32f: return GetTopK<float>(idx, mode, sort);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    TopKLayer::TopKLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool TopKLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 2)
            SYNET_ERROR("TopKLayer supports only 1 input and 2 outputs!");
        const TopKParam & topK = this->Param().topK();
        _srcType = src[0]->GetType();
        _axis = src[0]->Index(topK.axis());
        Shape shape = src[0]->Shape();
        if(_axis >= shape.size())
            SYNET_ERROR("TopKLayer parameter axis: " << _axis << " has wrong value for input " << ToStr(shape) <<  " !");
        _k = topK.k();
        if (_k > shape[_axis])
            SYNET_ERROR("TopKLayer parameter k: " << _k << " has wrong value for input " << ToStr(shape) << " !");
        shape[_axis] = _k;
        _outer = src[0]->Size(0, _axis);
        _count = src[0]->Axis(_axis);
        _inner = src[0]->Size(_axis + 1);
        _idxType = topK.indexElementType();
        _mode = topK.mode();
        _sort = topK.sort();

        if (_srcType != TensorType32f)
            SYNET_ERROR("TopKLayer has wrong input type: " << Cpl::ToStr(_srcType) << " !");
        if (_idxType != TensorType32i && _idxType != TensorType64i)
            SYNET_ERROR("TopKLayer has wrong parameter indexElementType: " << Cpl::ToStr(_idxType) << " !");
        dst[0]->Reshape(_srcType, shape, src[0]->Format());
        dst[1]->Reshape(_idxType, shape, src[0]->Format());  
        Extend8u(buf, 0, Shp(_count, 16));
        _topK = GetTopK(_srcType, _idxType, _mode, _sort);
        if(_topK == NULL)
            SYNET_ERROR("TopKLayer can't get worker!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            dst[1]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void TopKLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _topK(src[0]->RawData(), _outer, _count, _inner, _k, Buf8u(buf, 0), dst[0]->RawData(), dst[1]->RawData());
    }
}
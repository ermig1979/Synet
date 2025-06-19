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

#include "Synet/Layers/ScatterNdLayer.h"

namespace Synet
{
    template <class T, class I> void ScatterNd(const T* src, const I* idx, size_t size, size_t inner, T* dst)
    {
        if (inner == 1)
        {
            for (size_t i = 0; i < size; ++i)
                dst[idx[i]] = src[i];
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                memcpy(dst + idx[i], src + i * inner, inner * sizeof(T));
        }
    }

    //--------------------------------------------------------------------------------------------------

    ScatterNdLayer::ScatterNdLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    size_t ScatterNdLayer::MemoryUsage() const
    {
        return _offset.RawSize();
    }

    void ScatterNdLayer::CompactWeight()
    {
        if(this->Weight().size())
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool ScatterNdLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 2 && src.size() != 3) || dst.size() != 1)
            SYNET_ERROR("ScatterNdLayer supports only 2-3 inputs and 1 output!");
        if (src.size() + Weight().size() != 3)
            SYNET_ERROR("ScatterNdLayer has wrong weight count!");
        if (src[0]->GetType() != src.back()->GetType())
            SYNET_ERROR("ScatterNdLayer the first and the last inputs must be the same type!");
        if (src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType64i)
            SYNET_ERROR("ScatterNdLayer src[0] unsupported type!");
        const Tensor& data = *src[0];
        const Tensor& index = src.size() == 2 ? this->Weight()[0] : *src[1];
        const Tensor& update = *src.back();
        if (!index.Const())
            SYNET_ERROR("ScatterNdLayer supports only constant index!");
        const ScatterParam& scatter = this->Param().scatter();
        if (index.GetType() != TensorType32i && index.GetType() != TensorType64i)
            SYNET_ERROR("ScatterNdLayer has wrong index type: " << Cpl::ToStr(index.GetType()) << " !");
        _version = scatter.version();
        if (_version == 0)
        {
            if (!SetOffsetScatterNd(data, index, update))
                SYNET_ERROR("ScatterNdLayer can't set offset for ScatterND version!");
        }
        else if (_version == 1)
        {
            _axis = data.Index(scatter.axis());
            if (!SetOffsetScatterElements(data, index, update))
                SYNET_ERROR("ScatterNdLayer can't set offset for ScatterElements version!");
        }
        else
            SYNET_ERROR("ScatterNdLayer unknown version " << _version  << " !");
        if (src[0] != dst[0])
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        if (src[0]->Const() && src.back()->Const())
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
        return true;
    }

    bool ScatterNdLayer::SetOffsetScatterNd(const Tensor& data, const Tensor& index, const Tensor& update)
    {
        size_t count = data.Count();  
        if(data.Count() + index.Count() - index.Axis(-1) - 1 != update.Count())
            SYNET_ERROR("ScatterNdLayer data, index " << ToStr(index.Shape()) << " and update have incompatible rank!");
        if (index.Axis(-1) == data.Count())
        {
            _inner = 1, _size = update.Size();
            if (index.Size() != _size * count)
                SYNET_ERROR("ScatterNdLayer has wrong weight size!");
            _offset.Reshape(TensorType32i, Shp(_size), TensorFormatUnknown);
            for (size_t o = 0, i = 0; o < _size; ++o)
            {
                Shape idx;
                for (size_t a = 0; a < count; ++a, ++i)
                    idx.push_back(index.GetType() == TensorType32i ? index.Data<int32_t>()[i] : index.Data<int64_t>()[i]);
                _offset.Data<int32_t>()[o] = (uint32_t)data.Offset(idx);
            }
        }
        else
        {
            size_t rank = index.Count() - 1;
            if(update.Size(rank) != data.Size(rank))
                SYNET_ERROR("ScatterNdLayer: data shape " << ToStr(data.Shape()) << " and update shape " << ToStr(update.Shape()) << " are incompatible!");
            _inner = update.Size(rank);
            for(size_t i = 0; i < rank; ++i)
                if (index.Axis(i) != update.Axis(i))
                    SYNET_ERROR("ScatterNdLayer: update shape " << ToStr(update.Shape()) << " and index shape " << ToStr(index.Shape()) << " are incompatible!");
            _size = index.Size(0, -1);
            _offset.Reshape(TensorType32i, Shp(_size), TensorFormatUnknown);
            for (size_t o = 0, i = 0; o < _size; ++o)
            {
                Shape idx(count, 0);
                for (size_t a = 0; a < rank; ++a, ++i)
                    idx[a] = index.GetType() == TensorType32i ? index.Data<int32_t>()[i] : index.Data<int64_t>()[i];
                _offset.Data<int32_t>()[o] = (uint32_t)data.Offset(idx);
            }
        }
        return true;
    }

    bool ScatterNdLayer::SetOffsetScatterElements(const Tensor& data, const Tensor& index, const Tensor& update)
    {
        size_t count = data.Count();
        if (data.Count() != index.Count() || index.Count() != update.Count())
            SYNET_ERROR("ScatterNdLayer data, index and update must heave the same rank!");
        if (index.Shape() != update.Shape())
            SYNET_ERROR("ScatterNdLayer index and update must have the same shape!");
        for (size_t i = 0; i < count; ++i)
            if (i != _axis && index.Axis(i) != data.Axis(i))
                SYNET_ERROR("ScatterNdLayer: data shape " << ToStr(data.Shape()) << " and index shape " << ToStr(index.Shape()) << " are incompatible!");
        _inner = 1, _size = update.Size();
        _offset.Reshape(TensorType32i, Shp(_size), TensorFormatUnknown);
        size_t outer = index.Size(0, _axis), iAxis = index.Axis(_axis), dAxis = data.Axis(_axis), inner = index.Size(_axis + 1);
        for (size_t o = 0, offs = 0; o < outer; ++o)
        {
            for (size_t a = 0; a < iAxis; ++a)
            {
                for (size_t i = 0; i < inner; ++i, ++offs)
                {
                    size_t idx = index.GetType() == TensorType32i ? index.Data<int32_t>()[offs] : index.Data<int64_t>()[offs];
                    _offset.Data<int32_t>()[offs] = (int32_t)(o * dAxis * inner + idx * inner + i);
                }
            }
        }
        return true;
    }

    void ScatterNdLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_version == 0)
            ForwardCpuNd(src, dst);
    }

    void ScatterNdLayer::ForwardCpuNd(const TensorPtrs& src, const TensorPtrs& dst)
    {
        if (src[0] != dst[0])
            memcpy(dst[0]->RawData(), src[0]->RawData(), src[0]->RawSize());
        const int32_t* pOffs = _offset.Data<int32_t>();
        size_t size = src.back()->Size();
        if (src[0]->GetType() == TensorType32f)
            ScatterNd(src.back()->Data<float>(), pOffs, _size, _inner, dst[0]->Data<float>());
        else if (src[0]->GetType() == TensorType64i)
            ScatterNd(src.back()->Data<int64_t>(), pOffs, _size, _inner, dst[0]->Data<int64_t>());
        else
            assert(0);
    }

}
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
    template <class T, class I> void ScatterNd(const T* src, const I* idx, size_t size, T* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[idx[i]] = src[i];
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
        if (src[0]->GetType() != src.back()->GetType())
            SYNET_ERROR("ScatterNdLayer the first and the last inputs must be the same type!");
        if (src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType64i)
            SYNET_ERROR("ScatterNdLayer src[0] unsupported type!");
        size_t count = src[0]->Count(), size = src.back()->Size();
        _offset.Reshape(TensorType32i, Shp(size), TensorFormatUnknown);
        if (src.size() == 2)
        {
            if (this->Weight().size() != 1)
                SYNET_ERROR("ScatterNdLayer has wrong weight count!");
            const Tensor &idx = this->Weight()[0];
            if (idx.GetType() != TensorType32i)
                SYNET_ERROR("ScatterNdLayer has wrong weight type!");
            if (idx.Axis(-1) != count)
                SYNET_ERROR("ScatterNdLayer has wrong weight shape!");
            if (idx.Size() != size * count)
                SYNET_ERROR("ScatterNdLayer has wrong weight size!");        
            for (size_t o = 0, i = 0; o < size; ++o)
            {
                Shape index;
                for (size_t a = 0; a < count; ++a, ++i)
                    index.push_back(idx.Data<int32_t>()[i]);
                _offset.Data<int32_t>()[o] = (uint32_t)src[0]->Offset(index);
            }        
        }
        else if (src.size() == 3 && src[1]->Const())
        {
            const Tensor& idx = *src[1];
            if (idx.GetType() != TensorType64i)
                SYNET_ERROR("ScatterNdLayer src[1] must be INT64 type!");
            if (idx.Axis(-1) != count)
                SYNET_ERROR("ScatterNdLayer has wrong src[1] shape!");
            if (idx.Size() != size * count)
                SYNET_ERROR("ScatterNdLayer has wrong weight size!");
            for (size_t o = 0, i = 0; o < size; ++o)
            {
                Shape index;
                for (size_t a = 0; a < count; ++a, ++i)
                    index.push_back((size_t)idx.Data<int64_t>()[i]);
                _offset.Data<int32_t>()[o] = (uint32_t)src[0]->Offset(index);
            }
        }
        else
            SYNET_ERROR("ScatterNdLayer supports only constant index!");
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

    void ScatterNdLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (src[0] != dst[0])
            memcpy(dst[0]->RawData(), src[0]->RawData(), src[0]->RawSize());
        const int32_t * pOffs = _offset.Data<int32_t>();
        size_t size = src.back()->Size();
        if (src[0]->GetType() == TensorType32f)
            ScatterNd(src.back()->Data<float>(), pOffs, size, dst[0]->Data<float>());
        else if (src[0]->GetType() == TensorType64i)
            ScatterNd(src.back()->Data<int64_t>(), pOffs, size, dst[0]->Data<int64_t>());
        else
            assert(0);
    }
}
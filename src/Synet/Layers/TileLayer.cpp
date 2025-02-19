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

#include "Synet/Layers/TileLayer.h"

namespace Synet
{
    template<class T> void Tile(const T * src, size_t outer, size_t tile, size_t inner, T * dst)
    {
        if (inner == 1)
        {
            for (size_t o = 0; o < outer; ++o, dst += tile)
                CpuSet<T>(tile, src[o], dst);
        }
        else
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t t = 0; t < tile; ++t)
                {
                    CpuCopy<T>(src, inner, dst);
                    dst += inner;
                }
                src += inner;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    TileLayer::TileLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool TileLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    { 
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("TileLayer supports only 1-2 inputs and 1 output!");

        _srcType = src[0]->GetType();
        if (_srcType != TensorType32f && _srcType != TensorType64i && _srcType != TensorType32i)
            SYNET_ERROR("TileLayer has wrong src[0] type!");
        Shape dstShape = src[0]->Shape();
        size_t axis = 0;
        if (src.size() == 1)
        {
            axis = src[0]->Index(this->Param().tile().axis());
            _tiles = this->Param().tile().tiles();
            dstShape[axis] *= _tiles;
        }
        else
        {
            if(src[1]->GetType() != TensorType64i)
                SYNET_ERROR("TileLayer has wrong src[1] type!");
            Shape shape1 = Shp(src[1]->Data<int64_t>(), src[1]->Size());
            if(!IsCompatible(src[0]->Shape(), shape1))
                SYNET_ERROR("TileLayer has wrong src[1] content: " << ToStr(shape1) << " !");
            dstShape = OutputShape(src[0]->Shape(), shape1);
            _tiles = 1;
            for (ptrdiff_t a = dstShape.size() - 1, a0 = dstShape.size() - src[0]->Count(); a >= a0; --a)
            {
                if (dstShape[a] != src[0]->Axis(a - a0))
                {
                    axis = a - a0;
                    _tiles = dstShape[a];
                    break;
                }
            }
        }
        _outer = src[0]->Size(0, axis);
        _inner = src[0]->Size(axis);
        dst[0]->Reshape(_srcType, dstShape, src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
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

    void TileLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_srcType)
        {
        case TensorType32f: 
            Tile<float>(src[0]->Data<float>(), _outer, _tiles, _inner, dst[0]->Data<float>());
            break;
        case TensorType64i:
            Tile<int64_t>(src[0]->Data<int64_t>(), _outer, _tiles, _inner, dst[0]->Data<int64_t>());
            break;
        case TensorType32i:
            Tile<int32_t>(src[0]->Data<int32_t>(), _outer, _tiles, _inner, dst[0]->Data<int32_t>());
            break;
        default:
            assert(0);
        }
    }
}
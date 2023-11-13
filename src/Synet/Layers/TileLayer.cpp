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
        : Base(param, context)
    {
    }

    bool TileLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    { 
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("TileLayer supports only 1-2 inputs and 1 output!");

        _srcType = src[0]->GetType();
        if (_srcType != TensorType32f && _srcType != TensorType64i)
            SYNET_ERROR("TileLayer has wrong src[0] type!");
        Shape shape = src[0]->Shape();
        size_t axis = 0;
        if (src.size() == 1)
        {
            axis = src[0]->Index(this->Param().tile().axis());
            _tiles = this->Param().tile().tiles();
        }
        else
        {
            if(src[0]->Count() != src[1]->Size() || src[1]->GetType() != TensorType64i)
                SYNET_ERROR("TileLayer has wrong src[1] size or type!");
            const int64_t * pSrc1 = src[1]->Data<int64_t>();
            _tiles = 1;
            for (size_t a = 0; a < src[1]->Size(); ++a)
            {
                if (pSrc1[a] != src[0]->Axis(a) && pSrc1[a] != 1)
                {
                    axis = a;
                    _tiles = (size_t)pSrc1[a];
                    break;
                }
            }
        }
        shape[axis] *= _tiles;
        _outer = src[0]->Size(0, axis);
        _inner = src[0]->Size(axis);
        dst[0]->Reshape(_srcType, shape, src[0]->Format());
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
        default:
            assert(0);
        }
    }
}
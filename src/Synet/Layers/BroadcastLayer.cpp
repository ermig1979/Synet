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

#include "Synet/Layers/BroadcastLayer.h"

namespace Synet
{
    template<class T> void Broadcast(const uint8_t* src8, size_t size, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[0];
    }

    //-------------------------------------------------------------------------------------------------

    BroadcastLayer::BroadcastPtr GetBroadcast(TensorType type)
    {
        switch (type)
        {
        case TensorType32f: return Broadcast<float>;
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    BroadcastLayer::BroadcastLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool BroadcastLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("BroadcastLayer supports only 2 inputs and 1 output!");
        if (src[0]->Size() != 1 || src[1]->Count() != 1)
            SYNET_ERROR("BroadcastLayer inputs have wrong shapes!");

        _type = src[0]->GetType();
        _broadcast = GetBroadcast(_type);
        if(_broadcast == NULL)
            SYNET_ERROR("BroadcastLayer src[0] has wrong type " << _type << " !");
        Shape shape(src[1]->Size());

        switch (src[1]->GetType())
        {
        case TensorType64i:
            for (size_t i = 0; i < src[1]->Size(); ++i)
                shape[i] = (size_t)src[1]->Data<int64_t>()[i];
            break;
        case TensorType32i:
            for (size_t i = 0; i < src[1]->Size(); ++i)
                shape[i] = (size_t)src[1]->Data<int32_t>()[i];
            break;
        default:
            SYNET_ERROR("BroadcastLayer src[1] has wrong type " << src[1]->GetType() << " !");
        }
        dst[0]->Reshape(_type, shape, src[0]->Format());
        _size = dst[0]->Size();

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
        return true;
    }

    void BroadcastLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _broadcast(src[0]->RawData(), _size, dst[0]->RawData());
    }
}
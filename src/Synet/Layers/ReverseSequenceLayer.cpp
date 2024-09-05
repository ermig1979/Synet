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

#include "Synet/Layers/ReverseSequenceLayer.h"

namespace Synet
{
    template <class T> void ReverseSequenceLayerForwardCpu(const uint8_t* src8, size_t outer, size_t reverse, size_t inner, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        if (inner == 1)
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t r = 0; r < reverse; ++r)
                    dst[r] = src[reverse - 1 - r];
                dst += reverse;
                src += reverse;
            }
        }
        else
        {
            for (size_t o = 0; o < outer; ++o)
            {
                for (size_t r = 0; r < reverse; ++r)
                    memcpy(dst + r * inner, src + (reverse - 1 - r) * inner, inner * sizeof(T));
                dst += reverse * inner;
                src += reverse * inner;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    ReverseSequenceLayer::ReverseSequenceLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool ReverseSequenceLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("ReverseSequenceLayer supports only 1 or 2 inputs and 1 output!");
        if (src[0] == dst[0])
            SYNET_ERROR("ReverseSequenceLayer input and output can't be the same tensor!");

        size_t seqAxis = src[0]->Index(this->Param().reverseSequence().seqAxis()), i;
        const Shape & shape = src[0]->Shape();
        if(seqAxis >= shape.size())
            SYNET_ERROR("ReverseSequenceLayer has wrong parameter: reverseSequence().seqAxis()!");
        for (i = 0, _outer = 1; i < seqAxis; i++)
            _outer *= shape[i];
        _reverse = shape[seqAxis];
        for (i = seqAxis + 1, _inner = 1; i < shape.size(); ++i)
            _inner *= shape[i];
        switch (src[0]->GetType())
        {
        case TensorType32f: _reverseSequence = ReverseSequenceLayerForwardCpu<float>; break;
        default:
            SYNET_ERROR("ReverseSequenceLayer: Unsupported type of input tensor!");
        }
        dst[0]->Reshape(src[0]->GetType(), shape, src[0]->Format());
        if (src[0]->Const())
        {
            _const = true;
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void ReverseSequenceLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _reverseSequence(src[0]->RawData(), _outer, _reverse, _inner, dst[0]->RawData());
    }
}
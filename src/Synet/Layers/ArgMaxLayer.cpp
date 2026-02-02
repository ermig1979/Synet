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

#include "Synet/Layers/ArgMaxLayer.h"

#include "Synet/Utils/Math.h"
#include "Synet/Utils/DebugPrint.h"

namespace Synet
{
    ArgMaxLayer::ArgMaxLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    } 

    bool ArgMaxLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ArgMaxLayer supports only 1 input and 1 output!");

        _srcT = src[0]->GetType();
        if(!(_srcT == TensorType32f))
            SYNET_ERROR("ArgMaxLayer don't support input type: " << Cpl::ToStr(_srcT) << " !");

        const ArgMaxParam & param = this->Param().argMax();
        Shape shape = src[0]->Shape();
        ptrdiff_t axis = src[0]->Index(param.axis());
        if(axis < 0 || axis >= (ptrdiff_t)shape.size())
            SYNET_ERROR("ArgMaxLayer has wrong parameter 'axis': " << param.axis() << " for src " << Detail::DebugPrint(shape) << " !");

        if (param.keepDims() || shape.size() == 1)
            shape[axis] = 1;
        else
            shape.erase(shape.begin() + axis);
        _outer = src[0]->Size(0, axis);
        _count = src[0]->Axis(axis);
        _inner = src[0]->Size(axis + 1);

        dst[0]->Reshape(TensorType64i, shape, src[0]->Format());

        if (src[0]->Const())
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

    template<class T> static void ArgMax(const T * src, size_t outer, size_t count, size_t inner, int64_t* dst)
    {
        for (size_t o = 0; o < outer; ++o)
        {
            memset(dst, 0, inner * sizeof(int64_t));
            for (size_t c = 0; c < count; ++c)
            {
                for (size_t i = 0; i < inner; ++i)
                {
                    if (src[c * inner + i] > src[dst[i] * inner + i])
                        dst[i] = c;
                }
            }
            src += count * inner;
            dst += inner;
        }
    }

    void ArgMaxLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        int64_t* pDst = dst[0]->Data<int64_t>();
        switch (_srcT)
        {
        case TensorType32f: ArgMax(src[0]->Data<float>(), _outer, _count, _inner, pDst); break;
        default:
            assert(0);
        }
    }
}
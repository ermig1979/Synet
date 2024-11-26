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

#include "Synet/Layers/NonZeroLayer.h"

namespace Synet
{
    template<class T> void NonZero(const Tensor<float>& src, Tensor<float>& dst)
    {
        size_t size = src.Size(), nonzeros = 0;
        for (size_t i = 0; i < size; ++i)
            if (src.Data<T>()[i] != T(0))
                nonzeros++;
        dst.Reshape(TensorType64i, Shp(src.Count(), nonzeros), TensorFormatUnknown);
        switch (src.Count())
        {
        case 1:
        {
            const T* data = src.Data<T>();
            int64_t* dst0 = dst.Data<int64_t>(Shp(0, 0));
            for (size_t i0 = 0, n0 = src.Axis(0), s = 0, d = 0; i0 < n0; i0++)
            {
                if (data[s] != T(0))
                {
                    dst0[d] = i0;
                    d++;
                }
                s++;
            }
            break;
        }
        case 2:
        {
            const T* data = src.Data<T>();
            int64_t* dst0 = dst.Data<int64_t>(Shp(0, 0));
            int64_t* dst1 = dst.Data<int64_t>(Shp(1, 0));
            for (size_t i0 = 0, n0 = src.Axis(0), s = 0, d = 0; i0 < n0; i0++)
            {
                for (size_t i1 = 0, n1 = src.Axis(1); i1 < n1; i1++)
                {
                    if (data[s] != T(0))
                    {
                        dst0[d] = i0;
                        dst1[d] = i1;
                        d++;
                    }
                    s++;
                }
            }
            break;
        }
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    NonZeroLayer::NonZeroLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool NonZeroLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("NonZeroLayer supports only 1 input and 1 output!");
        if (!src[0]->Const())
            SYNET_ERROR("NonZeroLayer supports only constant input!");
        if (src[0]->Count() > 2)
            SYNET_ERROR("NonZeroLayer supports only 1D and 2D input!");
        switch (src[0]->GetType())
        {
        case TensorType32f:
            NonZero<float>(*src[0], *dst[0]);
            break;
        case TensorType64i:
            NonZero<int64_t>(*src[0], *dst[0]);
            break;
        default:
            SYNET_ERROR("NonZeroLayer does not support input type: " << src[0]->GetType() << " !");
        }
        _const = true;
        dst[0]->SetConst(true);
        return true;
    }

    void NonZeroLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}
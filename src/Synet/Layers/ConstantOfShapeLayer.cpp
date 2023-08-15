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

#include "Synet/Layers/ConstantOfShapeLayer.h"

namespace Synet
{
    ConstantOfShapeLayer::ConstantOfShapeLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool ConstantOfShapeLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ConstantOfShapeLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType64i || src[0]->Count() != 1)
            SYNET_ERROR("ConstantOfShapeLayer input must be 64-bit integer 1D!");
        Shape dstShape;
        for (size_t i = 0, n = src[0]->Axis(0); i < n; ++i)
            dstShape.push_back((size_t)src[0]->Data<int64_t>()[i]);

        const TensorParam & value = this->Param().constantOfShape().value();
        if(value.shape() != Shp(1))
            SYNET_ERROR("ConstantOfShapeLayer parameter value mus be scalar!");

        switch (value.type())
        {
        case TensorType32f:
            dst[0]->Reshape(TensorType32f, dstShape, TensorFormatNchw, value.f32()[0]);
            break;
        default:
            SYNET_ERROR("ConstantOfShapeLayer usupported parameter value type: " << Cpl::ToStr(value.type()) << " !");
        }
        _const = true;
        dst[0]->SetConst(true);
        return true;
    }

    void ConstantOfShapeLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/DynamicQuantizeLinearLayer.h"
#include "Synet/Layers/Quantized/QuantizeLinearLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"

namespace Synet
{
    void DynamicQuantizeLinearMinMax(const float* src, size_t size, float& min, float& max)
    {
        min = FLT_MAX;
        max = -FLT_MAX;
        for (size_t i = 0; i < size; ++i)
        {
            float val = src[i];
            min = Min(val, min);
            max = Max(val, max);
        }
    }

    //-------------------------------------------------------------------------------------------------

    DynamicQuantizeLinearLayer::DynamicQuantizeLinearLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t DynamicQuantizeLinearLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * 4;
    }

    bool DynamicQuantizeLinearLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if (src.size() != 1 || dst.size() != 3)
            SYNET_ERROR("DynamicQuantizeLinearLayer supports only 1 input and 3 outputs!");

        _size = src[0]->Size();

        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("DynamicQuantizeLinearLayer supports only FP32 input!");

        dst[0]->Reshape(TensorType8u, src[0]->Shape(), src[0]->Format());
        dst[1]->Reshape(TensorType32f, Shp(1), src[0]->Format());
        dst[2]->Reshape(TensorType8u, Shp(1), src[0]->Format());
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

    void DynamicQuantizeLinearLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        float min, max;
        DynamicQuantizeLinearMinMax(src[0]->Data<float>(), _size, min, max);
        min = Min(min, 0.0f);
        max = Max(max, 0.0f);
        const int qmin = std::numeric_limits<uint8_t>::min(), qmax = std::numeric_limits<uint8_t>::max();
        float scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);
        float initialZeroPoint = qmin - min / scale;
        uint8_t zeroPoint = (uint8_t)NearByInt(Max(float(qmin), Min(float(qmax), initialZeroPoint)));
        QuantizeLinearUniform(src[0]->Data<float>(), scale, zeroPoint, _size, dst[0]->RawData(), TensorType8u);
        dst[1]->Data<float>()[0] = scale;
        dst[2]->Data<uint8_t>()[0] = zeroPoint;
    }
}
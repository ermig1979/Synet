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

#include "Synet/Layers/Quantized/DequantizeLinearLayer.h"

#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    static void DequantizeLinearUniform(const uint8_t* src, int bias, float norm, size_t size, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetDequantizeLinear(src, size, bias, &norm, dst);
#else
        for (size_t i = 0; i < size; ++i)
            dst[i] = DequantizeLinear(src[i], bias, norm);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    DequantizeLinearLayer::DequantizeLinearLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _uniform(NULL)
    {
    }

    int64_t DequantizeLinearLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * 2;
    }

    bool DequantizeLinearLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 0 && src.size() != 1) || dst.size() != 1)
            SYNET_ERROR("DequantizeLinearLayer supports only 0 or 1 inputs and 1 output!");

        const QuantizeParam& param = Param().quantize();
        _bias = -param.zero();
        _norm = (float)param.scale();

        _uniform = DequantizeLinearUniform;

        if (src.size() == 1)
        {
            if (src[0]->GetType() != TensorType8u)
                SYNET_ERROR("DequantizeLinearLayer supports only UINT8 input!");

            _size = src[0]->Size();

            dst[0]->Reshape(TensorType32f, src[0]->Shape(), src[0]->Format());
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
        }
        else
        {
            const Tensors& weight = this->Weight();
            if (weight.empty())
                SYNET_ERROR("DequantizeLinearLayer weight is empty!");
            if (weight[0].GetType() != TensorType8u)
                SYNET_ERROR("DequantizeLinearLayer supports only UINT8 weight!");

            _size = weight[0].Size();

            dst[0]->Reshape(TensorType32f, weight[0].Shape(), weight[0].Format());

            _uniform(weight[0].Data<uint8_t>(), _bias, _norm, _size, dst[0]->Data<float>());

            _const = true;
            dst[0]->SetConst(true);
        }

        return true;
    }

    void DequantizeLinearLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_uniform)
            _uniform(src[0]->Data<uint8_t>(), _bias, _norm, _size, dst[0]->Data<float>());
    }
}
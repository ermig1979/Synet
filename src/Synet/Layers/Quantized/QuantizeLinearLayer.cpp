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

#include "Synet/Layers/Quantized/QuantizeLinearLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"

namespace Synet
{
    void QuantizeLinearUniform(const float* src, float scale, int zero, size_t size, uint8_t* dst8, TensorType type)
    {
        if (type == TensorType8u)
        {
            uint8_t* dst = (uint8_t*)dst8;
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            SimdSynetQuantizeLinear(src, size, &scale, zero, dst);
#else
            for (size_t i = 0; i < size; ++i)
                dst[i] = (uint8_t)QuantizeLinear(src[i], scale, zero, 0, 255);
#endif
        }
        else if (type == TensorType8u)
        {
            int8_t* dst = (int8_t*)dst8;
            for (size_t i = 0; i < size; ++i)
                dst[i] = (int8_t)QuantizeLinear(src[i], scale, zero, -128, 127);
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    QuantizeLinearLayer::QuantizeLinearLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _uniform(NULL)
    {
    }

    int64_t QuantizeLinearLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * 2;
    }

    bool QuantizeLinearLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizeLinearLayer supports only 1 input and 1 output!");

        const QuantizeParam& param = Param().quantize();
        _type = param.type();
        _size = src[0]->Size();
        _zero = param.zero();
        _scale = 1.0f / float(param.scale());

        if(param.weights() != 0)
            SYNET_ERROR("QuantizeLinearLayer supports only uniform case!");

        if (_type != TensorType8u && _type != TensorType8i)
            SYNET_ERROR("QuantizeLinearLayer supports only UINT8 and INT8 quantize parameter type!");

        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("QuantizeLinearLayer supports only FP32 input!");

        _uniform = QuantizeLinearUniform;

        dst[0]->Reshape(_type, src[0]->Shape(), src[0]->Format());
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

    void QuantizeLinearLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if(_uniform)
            _uniform(src[0]->Data<float>(), _scale, _zero, _size, dst[0]->RawData(), _type);
    }
}
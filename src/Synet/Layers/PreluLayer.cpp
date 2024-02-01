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

#include "Synet/Layers/PreluLayer.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    void PreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, TensorFormat format)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetPreluLayerForward(src, slope, channels, spatial, dst, (SimdTensorFormatType)format);
#else
        if (format == TensorFormatNchw)
        {
            for (size_t i = 0; i < channels; ++i)
            {
                for (size_t s = 0; s < spatial; ++s)
                    dst[s] = CpuRelu(src[s], slope[i]);
                src += spatial;
                dst += spatial;
            }
        }
        else if (format == TensorFormatNhwc)
        {
            for (size_t s = 0; s < spatial; ++s)
            {
                for (size_t i = 0; i < channels; ++i)
                    dst[i] = CpuRelu(src[i], slope[i]);
                src += channels;
                dst += channels;
            }
        }
        else
            assert(0);
#endif    
    }   

    //-------------------------------------------------------------------------------------------------

    PreluLayer::PreluLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool PreluLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("PreluLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("PreluLayer supports only FP32 input!");
        if (this->Weight().size() != 1)
            SYNET_ERROR("PreluLayer has wrong weights number!");

        const PreluParam & param = this->Param().prelu();
        _axis = param.axis();
        _channels = this->Weight()[0].Size();
        _format = src[0]->Format();
        _batch = src[0]->Size(0, _axis);
        _spatial = src[0]->Size() / _batch / _channels;
        if(_batch*_spatial*_channels != src[0]->Size())
            SYNET_ERROR("PreluLayer has wrong weight size!");

        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
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

    void PreluLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const float * pSrc = src[0]->Data<float>();
        const float* pSlope = this->Weight()[0].Data<float>();
        float* pDst = dst[0]->Data<float>();
        for (size_t b = 0; b < _batch; ++b)
        {
            PreluLayerForward(pSrc, pSlope, _channels, _spatial, pDst, _format);
            pSrc += _channels*_spatial;
            pDst += _channels*_spatial;
        }
    }
}
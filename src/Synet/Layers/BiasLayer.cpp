/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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

#include "Synet/Layers/BiasLayer.h"

namespace Synet
{
    void BiasLayerForward(const float * src, const float* bias, size_t count, size_t size, float* dst, TensorFormat format)
    {
        if (format == TensorFormatNhwc)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = src[i] + bias[i];
                src += count;
                dst += count;
            }
        }
        else if(format == TensorFormatNchw)
        {
            for (size_t i = 0; i < count; ++i)
            {
                float b = bias[i];
                for (size_t j = 0; j < size; ++j)
                    dst[j] = src[j] + b;
                src += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    BiasLayer::BiasLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool BiasLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() + this->Weight().size() != 2 || dst.size() != 1)
            SYNET_ERROR("BiasLayer supports only 2 inputs or 1 input + 1 weight and 1 output!");
        for (size_t i = 0; i < src.size(); ++i)
            if (src[i]->GetType() != TensorType32f)
                SYNET_ERROR("BiasLayer supports only f32 inputs!");

        const ScaleParam & param = this->Param().scale();
        _axis = src[0]->Index(param.axis());
        if (_axis >= src[0]->Count())
            SYNET_ERROR("BiasLayer has wrong axis " << param.axis() << " parameter!");

        const Tensor & bias = (src.size() > 1 ? *src[1] : this->Weight()[0]);
        _format = src[0]->Format();
        _count = bias.Size();
        if (bias.Size() == src[0]->Size())
        {
            _num = 1;
            _size = 1;
        }
        else
        {
            _num = src[0]->Size(0, _axis);
            _size = src[0]->Size() / _num / _count;
        }
        if(src[0]->Size() != _num * _count * _size)
            SYNET_ERROR("BiasLayer has wrong input shapes!");
        if (src[0] != dst[0])
            dst[0]->Reshape(TensorType32f, src[0]->Shape(), src[0]->Format());
        if (src[0]->Const() && (src.size() == 1 || src[1]->Const()))
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

    void BiasLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const float * pSrc = src[0]->Data<float>();
        const float * pBias = src.size() > 1 ? src[1]->Data<float>() : this->Weight()[0].Data<float>();
        float * pDst = dst[0]->Data<float>();
        for (size_t n = 0; n < _num; ++n)
        {
            BiasLayerForward(pSrc, pBias, _count, _size, pDst, _format);
            pSrc += _count*_size;
            pDst += _count*_size;
        }
    }
}
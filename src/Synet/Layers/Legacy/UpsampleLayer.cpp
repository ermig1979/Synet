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

#include "Synet/Layers/Legacy/UpsampleLayer.h"

namespace Synet
{
    template <typename T> void UpsampleLayerForwardCpu(const T * src, size_t channel, size_t height, size_t width, size_t stride, T scale, int reverse, int trans, T * dst)
    {
        if (trans)
        {
            if (reverse)
            {
                for (size_t sy = 0; sy < height; sy += stride)
                {
                    for (size_t sx = 0; sx < width; sx += stride)
                        for (size_t i = 0; i < channel; ++i)
                            (*dst++) = scale*src[sx*channel + i];
                    src += width*stride*channel;
                }
            }
            else
            {
                for (size_t sy = 0; sy < height; ++sy)
                {
                    for (size_t ky = 0; ky < stride; ++ky)
                    {
                        for (size_t sx = 0; sx < width; ++sx)
                        {
                            for (size_t kx = 0; kx < stride; ++kx)
                                for (size_t i = 0; i < channel; ++i)
                                    (*dst++) = scale*src[sx*channel + i];
                        }
                    }
                    src += width*channel;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < channel; ++i)
            {
                if (reverse)
                {
                    for (size_t sy = 0; sy < height; sy += stride)
                    {
                        for (size_t sx = 0; sx < width; sx += stride)
                            (*dst++) = scale*src[sx];
                        src += width*stride;
                    }
                }
                else
                {
                    for (size_t sy = 0; sy < height; ++sy)
                    {
                        for (size_t ky = 0; ky < stride; ++ky)
                        {
                            for (size_t sx = 0; sx < width; ++sx)
                            {
                                for (size_t kx = 0; kx < stride; ++kx)
                                    (*dst++) = scale*src[sx];
                            }
                        }
                        src += width;
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    UpsampleLayer::UpsampleLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool UpsampleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("UpsampleLayer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("UpsampleLayer supports only FP32!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("UpsampleLayer supports only 4D tensors!");
        const UpsampleParam & param = this->Param().upsample();
        if (param.stride() < 0)
        {
            _reverse = 1;
            _stride = -param.stride();
        }
        else
        {
            _reverse = 0;
            _stride = param.stride();
        }
        _scale = param.scale();
        _trans = src[0]->Format() == TensorFormatNhwc;

        Shape shape = src[0]->Shape();
        _num = shape[0];
        if (_trans)
        {
            _height = shape[1];
            _width = shape[2];
            _channel = shape[3];
            if (_reverse)
            {
                shape[1] /= _stride;
                shape[2] /= _stride;
            }
            else
            {
                shape[1] *= _stride;
                shape[2] *= _stride;
            }
        }
        else
        {
            _channel = shape[1];
            _height = shape[2];
            _width = shape[3];
            if (_reverse)
            {
                shape[2] /= _stride;
                shape[3] /= _stride;
            }
            else
            {
                shape[2] *= _stride;
                shape[3] *= _stride;
            }            
        }
        dst[0]->Reshape(src[0]->GetType(), shape, src[0]->Format());
        this->UsePerfStat();
        _const = false;
        return true;
    }

    void UpsampleLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        UpsampleLayerForwardCpu(src[0]->Data<float>(), _channel, _height, _width, _stride, _scale, _reverse, _trans, dst[0]->Data<float>());
    }
}
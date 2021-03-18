/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"

namespace Synet
{
    namespace Detail
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
    }

    template <class T> class UpsampleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        UpsampleLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
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
            assert(shape.size() == 4);
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
            dst[0]->Reshape(shape, src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::UpsampleLayerForwardCpu(src[0]->CpuData(), _channel, _height, _width, _stride, _scale, _reverse, _trans, dst[0]->CpuData());
        }

    private:
        int _reverse, _trans;
        size_t _stride, _num, _channel, _height, _width;
        float _scale;
    };
}
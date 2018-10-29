/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
        template <typename T> void UpsampleLayerForwardCpu(const T * src, size_t number, size_t height, size_t width, size_t stride, int reverse, T scale, T * dst)
        {
            for (size_t i = 0; i < number; ++i)
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

    template <class T> class UpsampleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        UpsampleLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
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
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Shape shape = src[0]->Shape();
            size_t n = src[0]->Count();
            if (_reverse)
            {
                shape[n - 1] /= _stride;
                shape[n - 2] /= _stride;
            }
            else
            {
                shape[n - 1] *= _stride;
                shape[n - 2] *= _stride;
            }
            dst[0]->Reshape(shape);
            _number = src[0]->Size(0, -2);
            _height = src[0]->Axis(-2);
            _width = src[0]->Axis(-1);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            Detail::UpsampleLayerForwardCpu(src[0]->CpuData(), _number, _height, _width, _stride, _reverse, _scale, dst[0]->CpuData());
        }

    private:
        int _reverse;
        size_t _stride, _number, _height, _width;
        float _scale;
    };
}
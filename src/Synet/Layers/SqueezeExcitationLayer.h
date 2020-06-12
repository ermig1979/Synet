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
        template <class T, class S> void ChannelSum(const T* src, size_t channels, size_t height, size_t width, TensorFormat format, S * sum)
        {
            if (format == TensorFormatNhwc)
            {
                for (size_t c = 0; c < channels; ++c)
                    sum[c] = S(0);
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            sum[c] += src[c];
                        src += channels;
                    }
                }
            }
            else if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    sum[c] = S(0);
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                            sum[c] += src[w];
                        src += width;
                    }
                }
            }
            else
                assert(0);
        }
    }

    template <class T> class SqueezeExcitationLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SqueezeExcitationLayer(const LayerParam& param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 2 && src[0].Shape() == src[1].Shape());

            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
        }

        void ForwardCpu(const float * src0, const float* src1, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {

            }
        }

        void ForwardCpu(const uint8_t* src0, const uint8_t* src1, uint8_t* dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {

            }
        }

    private:
        TensorType _type;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width, _size; 
        Tensor _scale;
    };
}
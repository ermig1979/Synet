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
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template< class T> void ReorgLayerForwardCpu(const T * src, size_t batch, size_t channels, size_t height, size_t width, size_t stride, int forward, int trans, T * dst)
        {
            size_t out_c = channels / (stride*stride);
            for (size_t b = 0; b < batch; ++b)
            {
                if(trans)
                {
                    for (size_t j = 0; j < height; ++j)
                    {
                        for (size_t i = 0; i < width; ++i)
                        {
                            for (size_t k = 0; k < channels; ++k)
                            {
                                size_t src_index = k + channels*(i + width*(j + height*b));
                                size_t c2 = k % out_c;
                                size_t offset = k / out_c;
                                size_t w2 = i*stride + offset % stride;
                                size_t h2 = j*stride + offset / stride;
                                size_t dst_index = c2 + out_c*(w2 + width*stride*(h2 + height*stride*b));
                                if (forward)
                                    dst[dst_index] = src[src_index];
                                else
                                    dst[src_index] = src[dst_index];
                            }
                        }
                    }
                }
                else
                {
                    for (size_t k = 0; k < channels; ++k)
                    {
                        for (size_t j = 0; j < height; ++j)
                        {
                            for (size_t i = 0; i < width; ++i)
                            {
                                size_t src_index = i + width*(j + height*(k + channels*b));
                                size_t c2 = k % out_c;
                                size_t offset = k / out_c;
                                size_t w2 = i*stride + offset % stride;
                                size_t h2 = j*stride + offset / stride;
                                size_t dst_index = w2 + width*stride*(h2 + height*stride*(c2 + out_c*b));
                                if (forward) 
                                    dst[dst_index] = src[src_index];
                                else 
                                    dst[src_index] = src[dst_index];
                            }
                        }
                    }
                }
            }
        }
    }

    template <class T> class ReorgLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReorgLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ReorgParam & param = this->Param().reorg();
            _reverse = param.reverse() ? 1 : 0;
            _stride = param.stride();
            _trans = src[0]->Format() == TensorFormatNhwc ? 1 : 0;
            Shape shape = src[0]->Shape();
            assert(shape.size() == 4);
            if (_trans)
            {
                if (_reverse)
                {
                    shape[1] = shape[1] * _stride;
                    shape[2] = shape[2] * _stride;
                    shape[3] = shape[3] / (_stride*_stride);
                }
                else
                {
                    shape[1] = shape[1] / _stride;
                    shape[2] = shape[2] / _stride;
                    shape[3] = shape[3] * (_stride*_stride);
                }
            }
            else
            {
                if (_reverse) 
                {
                    shape[1] = shape[1] / (_stride*_stride);
                    shape[2] = shape[2] * _stride;
                    shape[3] = shape[3] * _stride;
                }
                else 
                {
                    shape[1] = shape[1] * (_stride*_stride);
                    shape[2] = shape[2] / _stride;
                    shape[3] = shape[3] / _stride;
                }
            }
            dst[0]->Reshape(shape, src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            const Shape & shape = dst[0]->Shape();
            if(_trans)
                Detail::ReorgLayerForwardCpu(src[0]->CpuData(), shape[0], shape[3], shape[1], shape[2], _stride, _reverse, _trans, dst[0]->CpuData());
            else
                Detail::ReorgLayerForwardCpu(src[0]->CpuData(), shape[0], shape[1], shape[2], shape[3], _stride, _reverse, _trans, dst[0]->CpuData());
        }

    private:
        size_t _stride;
        int _reverse, _trans;
    };
}
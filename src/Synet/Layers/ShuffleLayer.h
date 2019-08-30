/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
        template <class T> void ShuffleLayerForwardCpu(const T * src0, const T * src1, size_t channels, size_t spatial, T * dst0, T * dst1, TensorFormat format)
        {
            if (format == TensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t cd = 0;
                    for (size_t cs = 0; cs < channels; cs += 2, cd += 1)
                    {
                        dst0[cd] = src0[cs + 0];
                        dst1[cd] = src0[cs + 1];
                    }
                    for (size_t cs = 0; cs < channels; cs += 2, cd += 1)
                    {
                        dst0[cd] = src1[cs + 0];
                        dst1[cd] = src1[cs + 1];
                    }
                    src0 += channels;
                    src1 += channels;
                    dst0 += channels;
                    dst1 += channels;
                }
            }
            else
            {
                size_t cd = 0;
                for (size_t cs = 0; cs < channels; cs += 2, cd += 1)
                {
                    memcpy(dst0, src0 + 0 * spatial, sizeof(T) * spatial);
                    memcpy(dst1, src0 + 1 * spatial, sizeof(T) * spatial);
                    src0 += 2 * spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
                for (size_t cs = 0; cs < channels; cs += 2, cd += 1)
                {
                    memcpy(dst0, src1 + 0 * spatial, sizeof(T) * spatial);
                    memcpy(dst1, src1 + 1 * spatial, sizeof(T) * spatial);
                    src1 += 2 * spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
            }
        }
    }

    template <class T> class ShuffleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ShuffleLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->Count() == 4);
            _format = src[0]->Format();
            const Shape & shape = src[0]->Shape();
            _batch = shape[0];
            if (_format == TensorFormatNhwc)
            {
                _channels = shape[3];
                _spatial = shape[1] * shape[2];
            }
            else
            {
                _channels = shape[1];
                _spatial = shape[2] * shape[3];
            }
            assert(_channels%2 == 0);
            _size = src[0]->Size(1);
            dst[0]->Reshape(shape, _format);
            dst[1]->Reshape(shape, _format);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * src0 = src[0]->CpuData();
            const Type * src1 = src[1]->CpuData();
            Type * dst0 = dst[0]->CpuData();
            Type * dst1 = dst[1]->CpuData();
            for(size_t b = 0; b < _batch; ++b)
            {
                Detail::ShuffleLayerForwardCpu(src0, src1, _channels, _spatial, dst0, dst1, _format);
                src0 += _size;
                src1 += _size;
                dst0 += _size;
                dst1 += _size;
            }
        }
    private:
        TensorFormat _format;
        size_t _batch, _channels, _spatial, _size;
    };
}
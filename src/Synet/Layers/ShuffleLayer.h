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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void ShuffleLayerForwardCpu(const T * src0, const T * src1, size_t channels0, size_t channels1, size_t spatial, T * dst0, T * dst1, TensorFormat format, int type)
        {
            size_t channels = (channels0 + channels1) / 2, size = sizeof(T) * spatial;
            switch (type)
            {
            case 0:
                if (format == TensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0;
                        for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs + 0];
                            dst1[cd] = src0[cs + 1];
                        }
                        for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                        {
                            dst0[cd] = src1[cs + 0];
                            dst1[cd] = src1[cs + 1];
                        }
                        src0 += channels0;
                        src1 += channels1;
                        dst0 += channels;
                        dst1 += channels;
                    }
                }
                else
                {
                    size_t cd = 0;
                    for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src0 + 0 * spatial, size);
                        memcpy(dst1, src0 + 1 * spatial, size);
                        src0 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                    for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src1 + 0 * spatial, size);
                        memcpy(dst1, src1 + 1 * spatial, size);
                        src1 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                }
                break;
            case 1:
                if (format == TensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cs = 0;
                        for (size_t cd = 0; cd < channels0; cd += 2, cs += 1)
                        {
                            dst0[cd + 0] = src0[cs];
                            dst0[cd + 1] = src1[cs];
                        }
                        for (size_t cd = 0; cd < channels1; cd += 2, cs += 1)
                        {
                            dst1[cd + 0] = src0[cs];
                            dst1[cd + 1] = src1[cs];
                        }
                        src0 += channels;
                        src1 += channels;
                        dst0 += channels0;
                        dst1 += channels1;
                    }
                }
                else
                {
                    size_t cs = 0;
                    for (size_t cd = 0; cd < channels0; cs += 1, cd += 2)
                    {
                        memcpy(dst0 + 0 * spatial, src0, size);
                        memcpy(dst0 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst0 += 2 * spatial;
                    }
                    for (size_t cd = 0; cd < channels1; cs += 1, cd += 2)
                    {
                        memcpy(dst1 + 0 * spatial, src0, size);
                        memcpy(dst1 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst1 += 2 * spatial;
                    }
                }
                break;
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void ShuffleLayerForwardCpu<float>(const float * src0, const float * src1, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, TensorFormat format, int type)
        {
            ::SimdSynetShuffleLayerForward(src0, src1, channels0, channels1, spatial, dst0, dst1, (::SimdTensorFormatType)format, type);
        }
#endif
    }

    template <class T> class ShuffleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ShuffleLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 2 && src[0]->Count() == 4 && src[1]->Count() == 4 && src[0]->Format() == src[1]->Format());
            _format = src[0]->Format();
            const Shape & srcShape0 = src[0]->Shape();
            const Shape & srcShape1 = src[1]->Shape();
            _batch = srcShape0[0];
            assert(srcShape0[0] == srcShape1[0]);
            Shape dstShape = srcShape0;
            _type = this->Param().shuffle().type();
            assert(_type == 0 || _type == 1);
            if (_format == TensorFormatNhwc)
            {
                _srcC0 = srcShape0[3];
                _srcC1 = srcShape1[3];
                _dstC = (_srcC0 + _srcC1) / 2;
                assert(_srcC0  + _srcC1 == _dstC*2);
                dstShape[3] = _dstC;
                _spatial = srcShape0[1] * srcShape0[2];
                assert(srcShape0[1] == srcShape1[1] && srcShape0[2] == srcShape1[2]);
            }
            else
            {
                _srcC0 = srcShape0[1];
                _srcC1 = srcShape1[1];
                _dstC = (_srcC0 + _srcC1) / 2;
                assert(_srcC0 + _srcC1 == _dstC * 2);
                dstShape[1] = _dstC;
                _spatial = srcShape0[2] * srcShape0[3];
                assert(srcShape0[2] == srcShape1[2] && srcShape0[3] == srcShape1[3]);
            }
            dst[0]->Reshape(dstShape, _format);
            dst[1]->Reshape(dstShape, _format);
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * src0 = src[0]->CpuData();
            const Type * src1 = src[1]->CpuData();
            Type * dst0 = dst[0]->CpuData();
            Type * dst1 = dst[1]->CpuData();
            for(size_t b = 0; b < _batch; ++b)
            {
                Detail::ShuffleLayerForwardCpu(src0, src1, _srcC0, _srcC1, _spatial, dst0, dst1, _format, _type);
                src0 += _srcC0*_spatial;
                src1 += _srcC1*_spatial;
                dst0 += _dstC*_spatial;
                dst1 += _dstC*_spatial;
            }
        }
    private:
        TensorFormat _format;
        int _type;
        size_t _batch, _srcC0, _srcC1, _dstC, _spatial;
    };
}
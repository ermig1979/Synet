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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> void SpaceToDepthLayerForwardCpu(const T * src, size_t srcC, size_t srcH, size_t srcW, T * dst, TensorFormat format)
        {
            size_t dstC = srcC * 4, dstH = srcH / 2, dstW = srcW / 2;
            if (format == TensorFormatNhwc)
            {
                for (size_t y = 0; y < srcH; y += 2)
                {
                    const T * src0 = src + 0 * srcW * srcC;
                    const T * src1 = src + 1 * srcW * srcC;
                    for (size_t x = 0; x < srcW; x += 2)
                    {
                        const T * src00 = src0 + 0 * srcC;
                        const T * src01 = src0 + 1 * srcC;
                        const T * src10 = src1 + 0 * srcC;
                        const T * src11 = src1 + 1 * srcC;
                        for (size_t c = 0; c < srcC; ++c)
                            *dst++ = *src00++;
                        for (size_t c = 0; c < srcC; ++c)
                            *dst++ = *src10++;
                        for (size_t c = 0; c < srcC; ++c)
                            *dst++ = *src01++;
                        for (size_t c = 0; c < srcC; ++c)
                            *dst++ = *src11++;
                        src0 += 2 * srcC;
                        src1 += 2 * srcC;
                    }
                    src += 2 * srcW * srcC;
                }
            }
            else if (format == TensorFormatNchw)
            {
                T * dst00 = dst + 0 * srcC * dstH * dstW;
                T * dst01 = dst + 1 * srcC * dstH * dstW;
                T * dst10 = dst + 2 * srcC * dstH * dstW;
                T * dst11 = dst + 3 * srcC * dstH * dstW;
                for (size_t c = 0; c < srcC; ++c)
                {
                    for (size_t y = 0; y < srcH; y += 2)
                    {
                        const T * src0 = src + 0 * srcW;
                        const T * src1 = src + 1 * srcW;
                        for (size_t x = 0; x < srcW; x += 2)
                        {
                            *dst00++ = src0[x + 0];
                            *dst01++ = src1[x + 0];
                            *dst10++ = src0[x + 1];
                            *dst11++ = src1[x + 1];
                        }
                        src += 2 * srcW;
                    }
                }
            }
            else
                assert(0);
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
#endif
    }

    template <class T> class SpaceToDepthLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SpaceToDepthLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 1 && src[0]->Count() == 4);
            _format = src[0]->Format();
            _batch = src[0]->Axis(0);
            Shape shape;
            if (_format == TensorFormatNhwc)
            {
                _srcH = src[0]->Axis(1);
                _srcW = src[0]->Axis(2);
                _srcC = src[0]->Axis(3);
                assert(_srcH % 2 == 0 && _srcW % 2 == 0);
                shape = Shp(_batch, _srcH / 2, _srcW / 2, _srcC * 4);
            }
            else
            {
                _srcC = src[0]->Axis(1);
                _srcH = src[0]->Axis(2);
                _srcW = src[0]->Axis(3);
                assert(_srcH % 2 == 0 && _srcW % 2 == 0);
                shape = Shp(_batch, _srcC * 4, _srcH / 2, _srcW / 2);
            }
            dst[0]->Reshape(shape, _format);
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * src0 = src[0]->CpuData();
            Type * dst0 = dst[0]->CpuData();
            for(size_t b = 0; b < _batch; ++b)
            {
                Detail::SpaceToDepthLayerForwardCpu(src0, _srcC, _srcH, _srcW, dst0, _format);
                src0 += _srcC*_srcH*_srcW;
                dst0 += _srcC*_srcH*_srcW;
            }
        }
    private:
        TensorFormat _format;
        size_t _batch, _srcC, _srcH, _srcW;
    };
}
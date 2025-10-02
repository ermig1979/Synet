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

#include "Synet/Layers/Reorder/SpaceToDepthLayer.h"

namespace Synet
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

    template <class T> void SpaceToDepthLayerForwardCpu(const uint8_t* src8, size_t batch, size_t srcC, size_t srcH, size_t srcW, uint8_t* dst8, TensorFormat format)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        for (size_t b = 0; b < batch; ++b)
        {
            SpaceToDepthLayerForwardCpu(src, srcC, srcH, srcW, dst, format);
            src += srcC * srcH * srcW;
            dst += srcC * srcH * srcW;
        }
    }

    //-------------------------------------------------------------------------------------------------

    SpaceToDepthLayer::SpaceToDepthLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        _spaceToDepth = NULL;
    }

    bool SpaceToDepthLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SpaceToDepthLayer supports only 1 input and 1 output!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("SpaceToDepthLayer supports only 4D tensors!");
        _format = src[0]->Format();
        _batch = src[0]->Axis(0);
        Shape shape;
        if (_format == TensorFormatNhwc)
        {
            _srcH = src[0]->Axis(1);
            _srcW = src[0]->Axis(2);
            _srcC = src[0]->Axis(3);
            shape = Shp(_batch, _srcH / 2, _srcW / 2, _srcC * 4);
        }
        else
        {
            _srcC = src[0]->Axis(1);
            _srcH = src[0]->Axis(2);
            _srcW = src[0]->Axis(3);
            shape = Shp(_batch, _srcC * 4, _srcH / 2, _srcW / 2);
        }
        if(_srcH % 2 == 1 || _srcW % 2 == 1)
            SYNET_ERROR("SpaceToDepthLayer input has wrong shape!");
        switch (src[0]->GetType())
        {
        case TensorType32f: _spaceToDepth = SpaceToDepthLayerForwardCpu<float>; break;
        default:
            SYNET_ERROR("SpaceToDepthLayer unsupports this input type!");
        }
        dst[0]->Reshape(src[0]->GetType(), shape, _format);
        this->UsePerfStat();
        _const = false;
        return true;
    }

    void SpaceToDepthLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        _spaceToDepth(src[0]->RawData(), _batch, _srcC, _srcH, _srcW, dst[0]->RawData(), _format);
    }
}
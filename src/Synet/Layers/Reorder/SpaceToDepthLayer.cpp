/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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
    template <class T> void SpaceToDepthForward2x2s(const T * src, size_t srcC, size_t srcH, size_t srcW, T * dst, TensorFormat format)
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

    template <class T> void SpaceToDepthForward2x2s(const uint8_t* src8, size_t batch, size_t srcC, size_t srcH, size_t srcW, uint8_t* dst8, TensorFormat format)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        for (size_t b = 0; b < batch; ++b)
        {
            SpaceToDepthForward2x2s(src, srcC, srcH, srcW, dst, format);
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

    bool SpaceToDepthLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SpaceToDepthLayer supports only 1 input and 1 output!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("SpaceToDepthLayer supports only 4D tensors!");
        const SpaceToDepthParam& param = this->Param().spaceToDepth();
        _kX = param.kX();
        _kY = param.kY();
        _back = param.backOperation();
        _first = param.channelsFirst();

        _type = src[0]->GetType();
        _format = src[0]->Format();
        const Shape & srcShape = src[0]->Shape();
        _batch = srcShape[0];

        if (_format == TensorFormatNhwc)
        {
            _srcH = srcShape[1];
            _srcW = srcShape[2];
            _srcC = srcShape[3];
        }
        else
        {
            _srcC = srcShape[1];
            _srcH = srcShape[2];
            _srcW = srcShape[3];
        }
        if (_back)
        {
            SYNET_ERROR("SpaceToDepthLayer is not support back transformation!");
            if (_srcC % (_kY * _kX) != 0)
                SYNET_ERROR("SpaceToDepthLayer input has wrong shape!");
            _dstC = _srcC / _kX / _kY;
            _dstH = _srcH * _kY;
            _dstW = _srcW * _kX;
        }
        else
        {
            if (_srcH % _kY != 0 || _srcW % _kX != 0)
                SYNET_ERROR("SpaceToDepthLayer input has wrong shape!");
            _dstC = _srcC * _kX * _kY;
            _dstH = _srcH / _kY;
            _dstW = _srcW / _kX;
        }

        Shape dstShape;
        if (_format == TensorFormatNhwc)
            dstShape = Shp(_batch, _dstH, _dstW, _dstC);
        else
            dstShape = Shp(_batch, _dstC, _dstH, _dstW);

        if (_kX == 2 && _kY == 2 && _first == false)
        {
            switch (src[0]->GetType())
            {
            case TensorType32f: _spaceToDepth = SpaceToDepthForward2x2s<float>; break;
            default:
                SYNET_ERROR("SpaceToDepthLayer unsupports this input type!");
            }
        }
        else
            SYNET_ERROR("SpaceToDepthLayer unsupports these kX " << _kX << " and kY " << _kY << " !");

        dst[0]->Reshape(_type, dstShape, _format);
        std::stringstream desc;
        desc << srcShape[0] << "x" << srcShape[1] << "x" << srcShape[2] << "x" << srcShape[3];
        desc << "-" << _kY << "x" << _kX;
        desc << (_back ? "-b" : "-f") << (_first ? "c" : "s");
        desc << "-" << Cpl::ToStr(_type);
        this->UsePerfStat(desc.str());
        _const = false;
        return true;
    }

    void SpaceToDepthLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        _spaceToDepth(src[0]->RawData(), _batch, _srcC, _srcH, _srcW, dst[0]->RawData(), _format);
    }
}
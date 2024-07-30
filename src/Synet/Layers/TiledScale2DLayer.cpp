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

#include "Synet/Layers/TiledScale2DLayer.h"

namespace Synet
{
    template <class T> void TiledScale2D(const T* src, size_t channels, size_t height, size_t width, const T* ver, const T* hor, T* dst, TensorFormat format)
    {
        if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; ++x)

                        dst[x] = src[x] * ver[x] * hor[0];
                    src += width, dst += width;
                    hor += 1;
                }
                ver += width;
            }
        }
        else if (format == TensorFormatNhwc)
        {
            for (size_t y = 0; y < height; ++y)
            {
                const T* pVer = ver;
                for (size_t x = 0; x < width; ++x)
                {
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = src[c] * pVer[c] * hor[c];
                    src += channels, dst += channels, pVer += channels;
                }
                hor += channels;
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    TiledScale2DLayer::TiledScale2DLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool TiledScale2DLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("TiledScale2DLayer supports only 3 inputs and 1 output!");
        if(src[0]->GetType() != TensorType32f || src[1]->GetType() != TensorType32f || src[2]->GetType() != TensorType32f)
            SYNET_ERROR("TiledScale2DLayer supports only FP32 input type!");
        _type = src[0]->GetType();
        _format = src[0]->Format();
        if (src[1]->Format() != _format || src[2]->Format() != _format)
            SYNET_ERROR("TiledScale2DLayer inputs must have the same format!");
        if (src[0]->Count() != 4 || src[1]->Count() != 4 || src[2]->Count() != 4)
            SYNET_ERROR("TiledScale2DLayer supports only 4D inputs!");
        _batch = src[0]->Axis(0);
        if (_format == TensorFormatNchw)
        {
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
            if(src[1]->Shape() != Shp(_batch, _channels, 1, _width))
                SYNET_ERROR("TiledScale2DLayer: check src[1] shape!");
            if (src[2]->Shape() != Shp(_batch, _channels, _height, 1))
                SYNET_ERROR("TiledScale2DLayer: check src[2] shape!");
        }
        else if(_format == TensorFormatNhwc)
        {
            _height = src[0]->Axis(1);
            _width = src[0]->Axis(2);
            _channels = src[0]->Axis(3);
            if (src[1]->Shape() != Shp(_batch, 1, _width, _channels))
                SYNET_ERROR("TiledScale2DLayer: check src[1] shape!");
            if (src[2]->Shape() != Shp(_batch, _height, 1, _channels))
                SYNET_ERROR("TiledScale2DLayer: check src[2] shape!");
        }
        else
            SYNET_ERROR("TiledScale2DLayer supports only NCHW or NHWC inputs!");
        if(dst[0] != src[0])
            dst[0]->Reshape(_type, src[0]->Shape(), _format);
        _const = false;
        this->UsePerfStat();
        return true;
    }

    void TiledScale2DLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        for (size_t b = 0; b < _batch; ++b)
        {
            Index idx = Shp(b, 0, 0, 0);
            switch (_type)
            {
            case TensorType32f:
                TiledScale2D<float>(src[0]->Data<float>(idx), _channels, _height, _width, src[1]->Data<float>(idx), src[2]->Data<float>(idx), dst[0]->Data<float>(idx), _format);
                break;
            default:
                assert(0);
            }
        }
    }
}
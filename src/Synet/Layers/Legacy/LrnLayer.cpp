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

#include "Synet/Layers/Legacy/LrnLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    template<class T> void LrnLayerCrossChannelsCpu(const T * src, size_t channels, size_t size, size_t inner, T alpha, T beta, T k, T * buf, T * dst, int trans)
    {
        size_t prePad = (size - 1) / 2;
        if (trans)
        {
            size_t paddedSize = channels + size - 1;
            T * padded = buf;
            T * scale = buf + paddedSize;
            for (size_t i = 0; i < inner; ++i)
            {
                CpuSet(channels, k, scale);
                CpuSet(paddedSize, T(0), padded);
                CpuSqr(src, channels, padded + prePad);
                for (size_t c = 0; c < size; ++c)
                    CpuAxpy(padded + c, 1, alpha, scale);
                for (size_t c = 1; c < channels; ++c)
                {
                    CpuCopy(scale + c - 1, 1, scale + c);
                    CpuAxpy(padded + c + size - 1, 1, alpha, scale + c);
                    CpuAxpy(padded + c - 1, 1, -alpha, scale + c);
                }
                CpuPow(scale, channels, -beta, dst);
                CpuMul(src, dst, channels, dst);
                src += channels;
                dst += channels;
            }
        }
        else
        {
            size_t paddedSize = (channels + size - 1)*inner;
            size_t scaleSize = channels*inner;
            T * padded = buf;
            T * scale = buf + paddedSize;
            CpuSet(scaleSize, k, scale);
            CpuSet(paddedSize, T(0), padded);
            CpuSqr(src, scaleSize, padded + inner*prePad);
            for (size_t c = 0; c < size; ++c)
                CpuAxpy(padded + c*inner, inner, alpha, scale);
            for (size_t c = 1; c < channels; ++c)
            {
                CpuCopy(scale + (c - 1)*inner, inner, scale + c*inner);
                CpuAxpy(padded + (c + size - 1)*inner, inner, alpha, scale + c*inner);
                CpuAxpy(padded + (c - 1)*inner, inner, -alpha, scale + c*inner);
            }
            CpuPow(scale, scaleSize, -beta, dst);
            CpuMul(src, dst, scaleSize, dst);
        }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
    template <> SYNET_INLINE void LrnLayerCrossChannelsCpu<float>(const float * src, size_t channels, size_t size, size_t inner, float alpha, float beta, float k, float * buf, float * dst, int trans)
    {
        float _k[3] = { k, alpha, -beta };
        ::SimdSynetLrnLayerCrossChannels(src, (size - 1) / 2, channels, inner, _k, dst, (::SimdTensorFormatType)trans);
    }
#endif

    //-------------------------------------------------------------------------------------------------

    LrnLayer::LrnLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool LrnLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("LrnLayer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("LrnLayer supports only FP32!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("LrnLayer supports only 4D tensors!");

        _normRegion = this->Param().lrn().normRegion();
        _size = this->Param().lrn().localSize();
        if(_size % 2 != 1)
            SYNET_ERROR("LrnLayer lrn().localSize() must be odd!");
        _prePad = (_size - 1) / 2;
        _alpha = this->Param().lrn().alpha();
        _beta = this->Param().lrn().beta();
        _k = this->Param().lrn().k();
        _trans = src[0]->Format() == TensorFormatNhwc;
        if (_trans)
        {
            _batch = src[0]->Axis(0);
            _height = src[0]->Axis(1);
            _width = src[0]->Axis(2);
            _channels = src[0]->Axis(3);
        }
        else
        {
            _batch = src[0]->Axis(0);
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
        }
        if (_normRegion == NormRegionTypeAcrossChannels)
        {
            dst[0]->Reshape(TensorType32f, src[0]->Shape(), src[0]->Format());
            if(_trans)
                _buffer.Reshape(TensorType32f, Shp(_height, _width, _channels * 2 + _size - 1));
            else
                _buffer.Reshape(TensorType32f, Shp(_channels * 2 + _size - 1));
        }
        else
            SYNET_ERROR("LrnLayer supports only NormRegionTypeAcrossChannels!")
        this->UsePerfStat();
        return true;
    }

    size_t LrnLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _buffer.RawSize();
    }

    void LrnLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        float alpha = _alpha / _size;
        for (size_t b = 0; b < _batch; ++b)
        {
            const float * pSrc = src[0]->Data<float>({ b, 0, 0, 0 });
            float* pDst = dst[0]->Data<float>({ b, 0, 0, 0 });
            LrnLayerCrossChannelsCpu(pSrc, _channels, _size, _width*_height, alpha, _beta, _k, _buffer.Data<float>(), pDst, _trans);
        }            
    }
}
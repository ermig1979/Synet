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

#include "Synet/Layers/Reorder/ShuffleLayer.h"

namespace Synet
{
    template <class T> void ShuffleLayerForwardCpu(const uint8_t * src0_, const uint8_t* src1_, size_t channels0, size_t channels1, size_t spatial, uint8_t* dst0_, uint8_t* dst1_, TensorFormat format, int shuffleType)
    {
        const T* src0 = (const T*)src0_, * src1 = (const T*)src1_;
        T* dst0 = (T*)dst0_, *dst1 = (T*)dst1_;
        size_t channels = (channels0 + channels1) / 2, size = sizeof(T) * spatial;
        switch (shuffleType)
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
    template <> SYNET_INLINE void ShuffleLayerForwardCpu<float>(const uint8_t* src0_, const uint8_t* src1_, size_t channels0, size_t channels1, size_t spatial, uint8_t* dst0_, uint8_t* dst1_, TensorFormat format, int shuffleType)
    {
        ::SimdSynetShuffleLayerForward((float*)src0_, (float*)src1_, channels0, channels1, spatial, (float*)dst0_, (float*)dst1_, (::SimdTensorFormatType)format, shuffleType);
    }
#endif

    //-------------------------------------------------------------------------------------------------

    ShuffleLayer::ShuffleLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool ShuffleLayer::Can16b() const
    {
        return Options().BFloat16Enable();
    }

    bool ShuffleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 2)
            SYNET_ERROR("ShuffleLayer supports 2 inputs and 2 outputs!");
        if(src[0]->Count() != 4 || src[1]->Count() != 4)
            SYNET_ERROR("ShuffleLayer supports only 4D input tensors!");
        if (src[0]->Format() != src[1]->Format())
            SYNET_ERROR("ShuffleLayer inputs must have the same format!");
        if (src[0]->GetType() != src[1]->GetType() || dst[0]->GetType() != dst[1]->GetType() || src[0]->GetType() != dst[0]->GetType())
            SYNET_ERROR("ShuffleLayer inputs and outputs must have the same type!");
        if ((src[0]->GetType() != TensorType32f && src[1]->GetType() != TensorType16b))
            SYNET_ERROR("ShuffleLayer supports only FP32 or BF16 inputs and outputs!");

        _format = src[0]->Format();
        _type = src[0]->GetType();
        _elem = GetTensorTypeSize(_type);
        const Shape & srcShape0 = src[0]->Shape();
        const Shape & srcShape1 = src[1]->Shape();
        _batch = srcShape0[0];
        if (srcShape0[0] != srcShape1[0])
            SYNET_ERROR("ShuffleLayer inputs must have the same shape[0]!");
        Shape dstShape = srcShape0;
        _shuffleType = this->Param().shuffle().type();
        if(_shuffleType != 0 && _shuffleType != 1)
            SYNET_ERROR("ShuffleLayer parameter shuffle.type() must be 1 or 2!");
        if (_format == TensorFormatNhwc)
        {
            _srcC0 = srcShape0[3];
            _srcC1 = srcShape1[3];
            _dstC = (_srcC0 + _srcC1) / 2;
            if(_srcC0 + _srcC1 != _dstC * 2)
                SYNET_ERROR("ShuffleLayer: check input channel dims!");
            dstShape[3] = _dstC;
            _spatial = srcShape0[1] * srcShape0[2];
            if(srcShape0[1] != srcShape1[1] || srcShape0[2] != srcShape1[2])
                SYNET_ERROR("ShuffleLayer: check input spatial dims!");
        }
        else
        {
            _srcC0 = srcShape0[1];
            _srcC1 = srcShape1[1];
            _dstC = (_srcC0 + _srcC1) / 2;
            if (_srcC0 + _srcC1 != _dstC * 2)
                SYNET_ERROR("ShuffleLayer: check input channel dims!");
            dstShape[1] = _dstC;
            _spatial = srcShape0[2] * srcShape0[3];
            if (srcShape0[2] != srcShape1[2] || srcShape0[3] != srcShape1[3])
                SYNET_ERROR("ShuffleLayer: check input spatial dims!");
        }
        dst[0]->Reshape(_type, dstShape, _format);
        dst[1]->Reshape(_type, dstShape, _format);
        this->UsePerfStat();
        return true;
    }

    void ShuffleLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const uint8_t * src0 = src[0]->RawData();
        const uint8_t* src1 = src[1]->RawData();
        uint8_t* dst0 = dst[0]->RawData();
        uint8_t* dst1 = dst[1]->RawData();
        for (size_t b = 0; b < _batch; ++b)
        {
            switch (_type)
            {
            case TensorType32f:
                ShuffleLayerForwardCpu<float>(src0, src1, _srcC0, _srcC1, _spatial, dst0, dst1, _format, _shuffleType);
                break;
            case TensorType16b:
                ShuffleLayerForwardCpu<uint16_t>(src0, src1, _srcC0, _srcC1, _spatial, dst0, dst1, _format, _shuffleType);
                break;
            }
            src0 += _srcC0 * _spatial * _elem;
            src1 += _srcC1 * _spatial * _elem;
            dst0 += _dstC * _spatial * _elem;
            dst1 += _dstC * _spatial * _elem;
        }
    }
}
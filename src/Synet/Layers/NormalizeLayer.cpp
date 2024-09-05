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

#include "Synet/Layers/NormalizeLayer.h"

namespace Synet
{
    void NormalizeLayerForwardCpu(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, float eps, int acrossSpatial, int trans, float* buf, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetNormalizeLayerForward(src, batch, channels, spatial, scale, &eps, (SimdBool)acrossSpatial, trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
#else
        if (trans)
        {
            if (acrossSpatial)
            {
                size_t size = channels * spatial;
                for (size_t b = 0; b < batch; ++b)
                {
                    float sum = eps;
                    for (size_t i = 0; i < size; ++i)
                        sum += Square(src[i]);
                    float k = 1.0f / ::sqrt(sum);
                    for (size_t i = 0; i < spatial; ++i)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * scale[c] * k;
                        dst += channels;
                        src += channels;
                    }
                }
            }
            else
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t i = 0; i < spatial; ++i)
                    {
                        float sum = eps;
                        for (size_t c = 0; c < channels; ++c)
                            sum += Square(src[c]);
                        float k = 1.0f / ::sqrt(sum);
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * scale[c] * k;
                        dst += channels;
                        src += channels;
                    }
                }
            }
        }
        else
        {
            if (acrossSpatial)
            {
                size_t size = channels * spatial;
                for (size_t b = 0; b < batch; ++b)
                {
                    float sum = eps;
                    for (size_t i = 0; i < size; ++i)
                        sum += Square(src[i]);
                    float k0 = 1.0f / ::sqrt(sum);
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float k = scale[c] * k0;
                        for (size_t i = 0; i < spatial; ++i)
                            dst[i] = src[i] * k;
                        dst += spatial;
                        src += spatial;
                    }
                }
            }
            else
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t i = 0; i < spatial; ++i)
                        buf[i] = eps;
                    for (size_t c = 0; c < channels; ++c)
                    {
                        const float* pSrc = src + c * spatial;
                        for (size_t i = 0; i < spatial; ++i)
                            buf[i] += Square(pSrc[i]);
                    }
                    for (size_t i = 0; i < spatial; ++i)
                        buf[i] = 1.0f / ::sqrt(buf[i]);
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float k = scale[c];
                        for (size_t i = 0; i < spatial; ++i)
                            dst[i] = src[i] * buf[i] * k;
                        dst += spatial;
                        src += spatial;
                    }
                }
            }
        }
#endif
    }

    //-------------------------------------------------------------------------------------------------

    void NormalizeLayerForwardV2Cpu(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, int trans, float* buf, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetNormalizeLayerForwardV2(src, batch, channels, spatial, scale, shift, &eps, trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
#else
        float k = 1.0f / float(channels);
        if (trans)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t i = 0; i < spatial; ++i)
                {
                    float sum = 0;
                    for (size_t c = 0; c < channels; ++c)
                        sum += src[c];
                    float mean = sum * k;
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = src[c] - mean;

                    float sqsum = 0;
                    for (size_t c = 0; c < channels; ++c)
                        sqsum += Square(dst[c]);
                    float norm = 1.0f / ::sqrt(sqsum * k + eps);
                    for (size_t c = 0; c < channels; ++c)
                        dst[c] = dst[c] * norm * scale[c] + shift[c];

                    dst += channels;
                    src += channels;
                }
            }
        }
        else
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                    buf[s] = 0;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    for (size_t s = 0; s < spatial; ++s, ++o)
                        buf[s] += src[o];
                }
                for (size_t s = 0; s < spatial; ++s)
                    buf[s] = buf[s] * k;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    for (size_t s = 0; s < spatial; ++s, ++o)
                        dst[o] = src[o] - buf[s];
                }

                for (size_t s = 0; s < spatial; ++s)
                    buf[s] = 0;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    for (size_t s = 0; s < spatial; ++s, ++o)
                        buf[s] += Square(dst[o]);
                }
                for (size_t s = 0; s < spatial; ++s)
                    buf[s] = 1.0f / ::sqrt(buf[s] * k + eps);
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    for (size_t s = 0; s < spatial; ++s, ++o)
                        dst[o] = dst[o] * buf[s] * scale[c] + shift[c];
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }
#endif
    }

    //-------------------------------------------------------------------------------------------------

    void NormalizeLayerForwardV3Cpu(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, int trans, float* buf, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetNormalizeLayerForwardV3(src, batch, channels, spatial, scale, shift, &eps, trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
#else
        float k = 1.0f / float(spatial);
        if (trans)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 0;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        buf[c] += src[o];
                }
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = buf[c] * k;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        dst[o] = src[o] - buf[c];
                }

                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 0;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        buf[c] += Square(dst[o]);
                }
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 1.0f / ::sqrt(buf[c] * k + eps);
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        dst[o] = dst[o] * buf[c] * scale[c] + shift[c];
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }
        else
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float sum = 0;
                    for (size_t s = 0; s < spatial; ++s)
                        sum += src[s];
                    float mean = sum * k;
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = src[s] - mean;

                    float sqsum = 0;
                    for (size_t s = 0; s < spatial; ++s)
                        sqsum += Square(dst[s]);
                    float norm = 1.0f / ::sqrt(sqsum * k + eps);
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = dst[s] * norm * scale[c] + shift[c];

                    dst += spatial;
                    src += spatial;
                }
            }
        }
#endif
    }

    //-------------------------------------------------------------------------------------------------

    void NormalizeLayerForwardV4Cpu(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, int trans, float* buf, float* dst)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        SimdSynetNormalizeLayerForwardV4(src, batch, channels, spatial, scale, shift, &eps, trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
#else
        float k = 1.0f / float(channels);
        if (trans)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                float sum = 0;
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 0;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        buf[c] += Square(src[o]);
                }
                for (size_t c = 0; c < channels; ++c)
                {
                    buf[c] = sqrt(buf[c]);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                    buf[c] = 1.0f + scale[c] * buf[c] * norm;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        dst[o] = src[o] * buf[c] + shift[c];
                }
                src += channels * spatial;
                dst += channels * spatial;
            }
        }
        else
        {
            for (size_t b = 0; b < batch; ++b)
            {
                float sum = 0;
                for (size_t c = 0, o = 0; c < channels; ++c)
                {
                    float sqsum = 0;
                    for (size_t s = 0; s < spatial; ++s, ++o)
                        sqsum += Square(src[o]);
                    buf[c] = sqrt(sqsum);
                    sum += buf[c];
                }
                float norm = 1.0f / (sum * k + eps);
                for (size_t c = 0; c < channels; ++c)
                {
                    float alpha = 1.0f + scale[c] * buf[c] * norm;
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = src[s] * alpha + shift[c];
                    dst += spatial;
                    src += spatial;
                }
            }
        }
#endif
    }

    //-------------------------------------------------------------------------------------------------

    void NormalizeLayerForwardV5Cpu(const float* src, size_t batch, size_t channels, size_t spatial, size_t group, const float* scale, const float* shift, float eps, int trans, float* buf0, float* dst)
    {
        assert(channels % group == 0);
        size_t size = spatial * group;
        float k = 1.0f / float(size);
        if (trans)
        {
            float* buf1 = buf0 + channels;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                    buf0[c] = 0;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        buf0[c] += src[o];
                }
                for (size_t c = 0; c < channels; c += group)
                {
                    float bias = 0;
                    for (size_t g = 0; g < group; g += 1)
                        bias += buf0[c + g];
                    bias *= k;
                    for (size_t g = 0; g < group; g += 1)
                        buf0[c + g] = bias;
                }
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        dst[o] = src[o] - buf0[c];
                }

                for (size_t c = 0; c < channels; ++c)
                    buf0[c] = 0;
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c, ++o)
                        buf0[c] += Square(dst[o]);
                }
                for (size_t c = 0, p = 0; c < channels; c += group, p++)
                {
                    float norm = 0;
                    for (size_t g = 0; g < group; g += 1)
                        norm += buf0[c + g];
                    norm = scale[p] / ::sqrt(norm * k + eps);
                    for (size_t g = 0; g < group; g += 1)
                    {
                        buf0[c + g] = norm;
                        buf1[c + g] = shift[p];
                    }
                }
                for (size_t s = 0, o = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; c += 1, ++o)
                        dst[o] = dst[o] * buf0[c] + buf1[c];
                }

                src += channels * spatial;
                dst += channels * spatial;
            }
        }
        else
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0, p = 0; c < channels; c += group, p++)
                {
                    float sum = 0;
                    for (size_t s = 0; s < size; ++s)
                        sum += src[s];
                    float mean = sum * k;
                    for (size_t s = 0; s < size; ++s)
                        dst[s] = src[s] - mean;

                    float sqsum = 0;
                    for (size_t s = 0; s < size; ++s)
                        sqsum += Square(dst[s]);
                    float norm = scale[p] / ::sqrt(sqsum * k + eps), bias = shift[p];
                    for (size_t s = 0; s < size; ++s)
                        dst[s] = dst[s] * norm + bias;

                    dst += size;
                    src += size;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    NormalizeLayer::NormalizeLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t NormalizeLayer::Flop() const
    {
        return (_version == 1 ? 4 : 7) * _batch * _channels * _spatial;
    }

    bool NormalizeLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("NormalizeLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("NormalizeLayer has unsupported input types!");

        const NormalizeParam& param = this->Param().normalize();
        const Tensors& weight = this->Weight();
        _version = param.version();
        _eps = param.eps();
        if (_version == 1)
        {
            _trans = src[0]->Format() == TensorFormatNhwc ? 1 : 0;
            _acrossSpatial = param.acrossSpatial() ? 1 : 0;
            _channelShared = param.channelShared() ? 1 : 0;

            _batch = src[0]->Axis(0);
            if (src[0]->Count() == 2)
            {
                _channels = 1;
                _spatial = src[0]->Axis(1);
            }
            else if (src[0]->Count() == 3)
            {
                if (_trans)
                {
                    _channels = src[0]->Axis(2);
                    _spatial = src[0]->Axis(1);
                }
                else
                {
                    _channels = src[0]->Axis(1);
                    _spatial = src[0]->Axis(2);
                }
            }
            else if (src[0]->Count() == 4)
            {
                if (_trans)
                {
                    _channels = src[0]->Axis(3);
                    _spatial = src[0]->Axis(1) * src[0]->Axis(2);
                }
                else
                {
                    _channels = src[0]->Axis(1);
                    _spatial = src[0]->Axis(2) * src[0]->Axis(3);
                }
            }
            else
                SYNET_ERROR("NormalizeLayer has unsupported input shape!");

            if (weight.empty())
            {
                if(_channelShared == 0)
                    SYNET_ERROR("NormalizeLayer has wrong parameter channelShared!");
                _scale.Reshape(TensorType32f, Shp(_channels), TensorFormatUnknown, 1.0f);
            }
            else
            {
                if (_channelShared)
                {
                    if(weight[0].Size() != 1)
                        SYNET_ERROR("NormalizeLayer weight[0] has wrong shape!");
                    _scale.Reshape(TensorType32f, Shp(_channels), weight[0].Format(), weight[0].Data<float>()[0]);
                }
                else
                    _scale.Share(weight[0]);
            }
        }
        else if (_version >= 2 && _version <= 5)
        {
            int axis = (int)src[0]->Index(param.axis());
            _channels = src[0]->Axis(axis);
            _trans = axis == src[0]->Count() - 1 ? 1 : 0;
            if (_trans)
            {
                _batch = 1;
                _spatial = src[0]->Size(0, axis);
            }
            else
            {
                _batch = src[0]->Size(0, axis);
                _spatial = src[0]->Size(axis + 1);
            }
            if (this->Weight().size() != 2)
                SYNET_ERROR("NormalizeLayer has wrong number of weights!");
            if (this->Weight()[0].Shape() != this->Weight()[1].Shape())
            {
                if (SignificantDimsCount(this->Weight()[0].Shape()) != 1 ||
                    SignificantDimsCount(this->Weight()[1].Shape()) != 1 ||
                    this->Weight()[0].Size() != this->Weight()[1].Size())
                    SYNET_ERROR("NormalizeLayer scale and bias weights have different shapes: " << ToStr(this->Weight()[0].Shape()) << " != " << ToStr(this->Weight()[1].Shape()) << "!");
            }
            _scale.Share(weight[0]);
            _shift.Share(weight[1]);
            if (_version == 5)
            {
                size_t groups = _scale.Size();
                if(groups == 0 || _channels % groups)
                    SYNET_ERROR("NormalizeLayer groups " << groups << " incompatible with channels " << _channels << " for V5 !");
                _group = _channels / groups;
            }
        }
        else
            SYNET_ERROR("Unsupported version " << _version << " of NormalizeLayer!");

        dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        buf[0]->Extend(TensorType32f, Shp(Max(_spatial, _channels * 2)));
        _const = false;
        this->UsePerfStat();
        return true;
    }

    void NormalizeLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        const float* pSrc = src[0]->Data<float>();
        float* pDst = dst[0]->Data<float>();
        float* pBuf = buf[0]->Data<float>();
        const float* pScale = _scale.Data<float>();
        const float* pShift = _shift.Data<float>();
        if (_version == 1)
            NormalizeLayerForwardCpu(pSrc, _batch, _channels, _spatial, pScale, _eps, _acrossSpatial, _trans, pBuf, pDst);
        else if (_version == 2)
            NormalizeLayerForwardV2Cpu(pSrc, _batch, _channels, _spatial, pScale, pShift, _eps, _trans, pBuf, pDst);
        else if (_version == 3)
            NormalizeLayerForwardV3Cpu(pSrc, _batch, _channels, _spatial, pScale, pShift, _eps, _trans, pBuf, pDst);
        else if (_version == 4)
            NormalizeLayerForwardV4Cpu(pSrc, _batch, _channels, _spatial, pScale, pShift, _eps, _trans, pBuf, pDst);
        else if (_version == 5)
            NormalizeLayerForwardV5Cpu(pSrc, _batch, _channels, _spatial, _group, pScale, pShift, _eps, _trans, pBuf, pDst);
        else
            assert(0);
    }
}
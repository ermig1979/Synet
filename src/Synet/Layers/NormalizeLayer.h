/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
        template<class T> void NormalizeLayerForwardCpu(const T * src, size_t batch, size_t channels, size_t spatial, const T * scale, T eps, int acrossSpatial, int trans, T * buf, T * dst)
        {
            if (trans)
            {
                if (acrossSpatial)
                {
                    size_t size = channels*spatial;
                    for (size_t b = 0; b < batch; ++b)
                    {
                        T sum = eps;
                        for (size_t i = 0; i < size; ++i)
                            sum += Square(src[i]);
                        T k = T(1) / ::sqrt(sum);
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
                            T sum = eps;
                            for (size_t c = 0; c < channels; ++c)
                                sum += Square(src[c]);
                            T k = T(1) / ::sqrt(sum);
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
                    size_t size = channels*spatial;
                    for (size_t b = 0; b < batch; ++b)
                    {
                        T sum = eps;
                        for (size_t i = 0; i < size; ++i)
                            sum += Square(src[i]);
                        T k0 = T(1) / ::sqrt(sum);
                        for (size_t c = 0; c < channels; ++c)
                        {
                            T k = scale[c] * k0;
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
                            const T * pSrc = src + c * spatial;
                            for (size_t i = 0; i < spatial; ++i)
                                buf[i] += Square(pSrc[i]);
                        }
                        for (size_t i = 0; i < spatial; ++i)
                            buf[i] = T(1) / ::sqrt(buf[i]);
                        for (size_t c = 0; c < channels; ++c)
                        {
                            T k = scale[c];
                            for (size_t i = 0; i < spatial; ++i)
                                dst[i] = src[i] * buf[i] * k;
                            dst += spatial;
                            src += spatial;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<class T> void NormalizeLayerForwardV2Cpu(const T* src, size_t batch, size_t channels, size_t spatial, const T* scale, const T* shift, T eps, int trans, T* buf, T* dst)
        {
            T k = T(1) / T(channels);
            if (trans)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t i = 0; i < spatial; ++i)
                    {
                        T sum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sum += src[c];
                        T mean = sum * k;
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] - mean;

                        T sqsum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sqsum += Square(dst[c]);
                        T norm = T(1) / ::sqrt(sqsum * k + eps);
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
                        buf[s] = T(1) / ::sqrt(buf[s] * k + eps);
                    for (size_t c = 0, o = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s, ++o)
                            dst[o] = dst[o] * buf[s] * scale[c] + shift[c];
                    }

                    src += channels * spatial;
                    dst += channels * spatial;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)

        template<> SYNET_INLINE void NormalizeLayerForwardCpu<float>(const float* src, size_t batch, size_t channels, 
            size_t spatial, const float* scale, float eps, int acrossSpatial, int trans, float* buf, float* dst)
        {
            ::SimdSynetNormalizeLayerForward(src, batch, channels, spatial, scale, &eps,
                (SimdBool)acrossSpatial, trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
        }

        template<> SYNET_INLINE void NormalizeLayerForwardV2Cpu<float>(const float* src, size_t batch, size_t channels,
            size_t spatial, const float* scale, const float* shift, float eps, int trans, float* buf, float* dst)
        {
            ::SimdSynetNormalizeLayerForwardV2(src, batch, channels, spatial, scale, shift, &eps,
                trans ? SimdTensorFormatNhwc : SimdTensorFormatNchw, buf, dst);
        }
#endif
    }

    template <class T> class NormalizeLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        NormalizeLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const NormalizeParam & param = this->Param().normalize();
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
                    assert(0);

                if (this->Weight().empty())
                {
                    assert(_channelShared);
                    _scale.Reshape(Shape({ _channels }), 1.0f);
                }
                else
                {
                    if (_channelShared)
                    {
                        assert(this->Weight()[0].Size() == 1);
                        _scale.Reshape(Shape({ _channels }), this->Weight()[0].CpuData()[0]);
                    }
                    else
                        _scale.Share(this->Weight()[0]);
                }
            }
            else if (_version == 2)
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
                _scale.Share(this->Weight()[0]);
                _shift.Share(this->Weight()[1]);
            }
            else
                assert(0);

            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            if(_trans)
                buf[0]->Extend(Shape({ _spatial }));
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            Type * pBuf = buf[0]->CpuData();
            const Type * pScale = _scale.CpuData();
            const Type* pShift = _shift.CpuData();
            if (_version == 1)
                Detail::NormalizeLayerForwardCpu(pSrc, _batch, _channels, _spatial, pScale, _eps, _acrossSpatial, _trans, pBuf, pDst);
            else if (_version == 2)
                Detail::NormalizeLayerForwardV2Cpu(pSrc, _batch, _channels, _spatial, pScale, pShift, _eps, _trans, pBuf, pDst);
            else
                assert(0);
        }

    private:
        typedef typename Base::Tensor Tensor;

        size_t _batch, _channels, _spatial;
        Tensor _scale, _shift;
        int _trans, _acrossSpatial, _channelShared, _version;
        Type _eps;
    };
}
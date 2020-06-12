/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
        template<class T> void NormalizeLayerForwardCpu(const T * src, size_t channels, size_t spatial, const T * scale, T eps, int acrossSpatial, int trans, T * buf, T * dst)
        {
            if (trans)
            {
                if (acrossSpatial)
                {
                    size_t size = channels*spatial;
                    T sum = 0;
                    for (size_t i = 0; i < size; ++i)
                        sum += Square(src[i]);
                    T k = T(1) / ::sqrt(sum + eps);
                    for (size_t i = 0; i < spatial; ++i)
                    { 
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = src[c] * scale[c] * k;
                        dst += channels;
                        src += channels;
                    }
                }
                else
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
            else
            {
                if (acrossSpatial)
                {
                    size_t size = channels*spatial;
                    T sum = 0;
                    for (size_t i = 0; i < size; ++i)
                        sum += Square(src[i]);
                    T k0 = T(1) / ::sqrt(sum + eps);
                    for (size_t c = 0; c < channels; ++c)
                    {
                        T k = scale[c] * k0;
                        for (size_t i = 0; i < spatial; ++i)
                            dst[i] = src[i] * k;
                        dst += spatial;
                        src += spatial;
                    }
                }
                else
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

    template <class T> class NormalizeLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        NormalizeLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const NormalizeParam & param = this->Param().normalize();
            _trans = src[0]->Format() == TensorFormatNhwc ? 1 : 0;
            _acrossSpatial = param.acrossSpatial() ? 1 : 0;
            _channelShared = param.channelShared() ? 1 : 0;
            _eps = param.eps();

            assert(src[0]->Count() >= 3);
            _num = src[0]->Size(0, -3);
            if (_trans)
            {
                _channels = src[0]->Axis(-1);
                _spatial = src[0]->Axis(-3) * src[0]->Axis(-2);
            }
            else
            {
                _channels = src[0]->Axis(-3);
                _spatial = src[0]->Axis(-2) * src[0]->Axis(-1);
            }

            if (_channelShared)
            {
                assert(this->Weight()[0].Size() == 1);
                _scale.Reshape(Shape({ _channels }), this->Weight()[0].CpuData()[0]);
            }
            else
                _scale.Share(this->Weight()[0]);

            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
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

            for (size_t n = 0; n < _num; ++n)
            {
                Detail::NormalizeLayerForwardCpu(pSrc, _channels, _spatial, pScale, _eps, _acrossSpatial, _trans, pBuf, pDst);
                pSrc += _channels*_spatial;
                pDst += _channels*_spatial;
            }
        }

    private:
        typedef typename Base::Tensor Tensor;

        size_t _num, _channels, _spatial;
        Tensor _scale;
        int _trans, _acrossSpatial, _channelShared;
        Type _eps;
    };
}
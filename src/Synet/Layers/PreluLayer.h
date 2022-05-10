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
        template <class T> void PreluLayerForwardCpu(const T * src, const T * slope, size_t channels, size_t spatial, T * dst, int trans)
        {
            if (trans)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < channels; ++i)
                        dst[i] = CpuRelu(src[i], slope[i]);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t i = 0; i < channels; ++i)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = CpuRelu(src[s], slope[i]);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void PreluLayerForwardCpu(const float * src, const float * slope, size_t channels, size_t spatial, float * dst, int trans)
        {
            ::SimdSynetPreluLayerForward(src, slope, channels, spatial, dst, (::SimdTensorFormatType)trans);
        }
#endif
    }

    template <class T> class PreluLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PreluLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const PreluParam & param = this->Param().prelu();
            _axis = param.axis();
            assert(this->Weight().size() == 1);
            _channels = this->Weight()[0].Size();
            _trans = src[0]->Format() == TensorFormatNhwc;
            _batch = src[0]->Size(0, _axis);
            _spatial = src[0]->Size() / _batch / _channels;
            assert(_batch*_spatial*_channels == src[0]->Size());
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pSrc = src[0]->CpuData();
            const Type * pSlope = this->Weight()[0].CpuData();
            Type * pDst = dst[0]->CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::PreluLayerForwardCpu(pSrc, pSlope, _channels, _spatial, pDst, _trans);
                pSrc += _channels*_spatial;
                pDst += _channels*_spatial;
            }
        }

    private:
        size_t _axis, _batch, _channels, _spatial;
        int _trans;
    };
}
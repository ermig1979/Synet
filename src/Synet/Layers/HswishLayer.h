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
        template <class T> SYNET_INLINE T HswishCpu(T value, T shift, T scale)
        {
            return Max(Min(value, shift) + shift, T(0))*scale*value;
        }

        template <class T> void HswishLayerForwardCpu(const T * src, size_t size, T shift, T scale, T * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = HswishCpu(src[i], shift, scale);
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void HswishLayerForwardCpu(const float * src, size_t size, float shift, float scale, float * dst)
        {
            ::SimdSynetHswish32f(src, size, &shift, &scale, dst);
        }
#endif
    }

    template <class T> class HswishLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        HswishLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            HswishParam hswish = this->Param().hswish();
            _shift = hswish.shift();
            _scale = hswish.scale();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::HswishLayerForwardCpu(src[0]->CpuData(), src[0]->Size(), _shift, _scale, dst[0]->CpuData());
        }

    private:
        Type _shift, _scale;
    };
}
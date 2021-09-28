/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
        template <class T> SYNET_INLINE T HardSigmoidCpu(T value, T scale, T shift)
        {
            return Max(T(0), Min(value * scale + shift, T(1)));
        }

        template <class T> void HardSigmoidLayerForwardCpu(const T * src, size_t size, T scale, T shift, T * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = HardSigmoidCpu(src[i], scale, shift);
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void HardSigmoidLayerForwardCpu(const float * src, size_t size, float scale, float shift, float * dst)
        {
            ::SimdSynetHardSigmoid32f(src, size, &scale, &shift, dst);
        }
#endif
    }

    template <class T> class HardSigmoidLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        HardSigmoidLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            HardSigmoidParam hardSigmoid = this->Param().hardSigmoid();
            _scale = hardSigmoid.scale();
            _shift = hardSigmoid.shift();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::HardSigmoidLayerForwardCpu(src[0]->CpuData(), src[0]->Size(), _scale, _shift, dst[0]->CpuData());
        }

    private:
        Type _scale, _shift;
    };
}
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
#include "Synet/Layers/UnaryOperationLayer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void SoftmaxLayerForwardCpu(const T * src, size_t outer, size_t count, size_t inner, T * dst)
        {
            Synet::Tensor<T> _buffer({ inner });
            T * buffer = _buffer.CpuData();
            for (size_t o = 0; o < outer; ++o)
            {
                Synet::CpuCopy(src, inner, buffer);
                const T * s = src + inner;
                for (size_t i = 1; i < count; ++i)
                {
                    Synet::CpuMax(s, buffer, inner, buffer);
                    s += inner;
                }

                s = src;
                T * d = dst;
                for (size_t i = 0; i < count; ++i)
                {
                    Synet::CpuSub(s, buffer, inner, d);
                    s += inner;
                    d += inner;
                }

                Synet::CpuExp(dst, count*inner, dst);

                Synet::CpuCopy(dst, inner, buffer);
                d = dst + inner;
                for (size_t i = 1; i < count; ++i)
                {
                    Synet::CpuAdd(d, buffer, inner, buffer);
                    d += inner;
                }

                d = dst;
                for (size_t i = 0; i < count; ++i)
                {
                    Synet::CpuDiv(d, buffer, inner, d);
                    d += inner;
                }
                src += count*inner;
                dst += count*inner;
            }
        }

        template <typename T> void LogSoftmaxLayerForwardCpu(const T * src, size_t outer, size_t count, size_t inner, T * dst)
        {
            Synet::Tensor<T> _buffer({ inner });
            T * buffer = _buffer.CpuData();
            for (size_t o = 0; o < outer; ++o)
            {
                Synet::CpuCopy(src, inner, buffer);
                const T * s = src + inner;
                for (size_t i = 1; i < count; ++i)
                {
                    Synet::CpuMax(s, buffer, inner, buffer);
                    s += inner;
                }

                s = src;
                T * d = dst;
                for (size_t i = 0; i < count; ++i)
                {
                    Synet::CpuSub(s, buffer, inner, d);
                    s += inner;
                    d += inner;
                }

                Synet::CpuExp(dst, count*inner, dst);

                Synet::CpuCopy(dst, inner, buffer);
                d = dst + inner;
                for (size_t i = 1; i < count; ++i)
                {
                    Synet::CpuAdd(d, buffer, inner, buffer);
                    d += inner;
                }

                d = dst;
                for (size_t i = 0; i < count; ++i)
                {
                    Synet::CpuDiv(d, buffer, inner, d);
                    Synet::CpuLog(d, inner, d);
                    d += inner;
                }
                src += count*inner;
                dst += count*inner;
            }
    }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void SoftmaxLayerForwardCpu<float>(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            ::SimdSynetSoftmaxLayerForward(src, outer, count, inner, dst);
        }
#endif
    }

    template <class T> class SoftmaxLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        SoftmaxLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _axis = Min<size_t>(this->Param().softmax().axis(), src[0]->Count() - 1);
            _log = this->Param().softmax().log();
            _outer = src[0]->Size(0, _axis);
            _count = src[0]->Axis(_axis);
            _inner = src[0]->Size(_axis + 1);
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if(_log)
                Detail::LogSoftmaxLayerForwardCpu(src[0]->CpuData(), _outer, _count, _inner, dst[0]->CpuData());
            else
                Detail::SoftmaxLayerForwardCpu(src[0]->CpuData(), _outer, _count, _inner, dst[0]->CpuData());
        }

    private:
        size_t _outer, _count, _inner, _axis;
        bool _log;
    };
}
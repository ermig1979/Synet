/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
        template <class T> void PreluLayerForwardCpu(const T * src, const T * slope, size_t count, size_t size, T * dst, int trans)
        {
            if (trans)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    for (size_t i = 0; i < count; ++i)
                        dst[i] = CpuRelu(src[i], slope[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    for (size_t j = 0; j < size; ++j)
                        dst[j] = CpuRelu(src[j], slope[i]);
                    src += size;
                    dst += size;
                }
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void PreluLayerForwardCpu(const float * src, const float * slope, size_t count, size_t size, float * dst, int trans)
        {
            ::SimdSynetPreluLayerForward(src, slope, count, size, dst, (::SimdBool)trans);
        }
#endif
    }

    template <class T> class PreluLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PreluLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(this->Weight().size() == 1);
            _count = this->Weight()[0].Size();
            assert(_count == 1 || _count == src[0]->Axis(1));
            _size = src[0]->Size() / _count;
            assert(_size*_count == src[0]->Size());
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            Detail::PreluLayerForwardCpu(src[0]->CpuData(), this->Weight()[0].CpuData(), _count, _size, dst[0]->CpuData(), 0);
        }

    private:
        size_t _count, _size;
    };
}
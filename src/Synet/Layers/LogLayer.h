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
        template <class T> void LogLayerForwardCpu(const T * src, size_t size, T scale, T shift, T base, T * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::log(src[i] * scale + shift)*base;
        }
    }

    template <class T> class LogLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        LogLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const LogParam & param = this->Param().log();
            Type base = param.base();
            if (base != Type(-1))
                assert(base > 0);
            Type logBase = (base == Type(-1)) ? Type(1) : ::log(base);
            assert(!(std::isnan(logBase) || std::isinf(logBase)));
            _baseScale = Type(1) / logBase;
            assert(!(std::isnan(_baseScale) || std::isinf(_baseScale)));
            _inputScale = param.scale();
            _inputShift = param.shift();            
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Detail::LogLayerForwardCpu(src[0]->CpuData(), src[0]->Size(), _inputScale, _inputShift, _baseScale, dst[0]->CpuData());
        }

    private:
        Type _baseScale, _inputScale, _inputShift;
    };
}
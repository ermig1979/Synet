/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
#include "Synet/Layers/ScaleLayer.h"

namespace Synet
{
    namespace Detail
    {
        template <typename T> void PowerForwardCpu(const T* src, size_t size, T scale, T shift, T power, T* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = ::pow(src[i] * scale + shift, power);
        }
    }

    template <class T> class PowerLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        PowerLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const PowerParam & param = this->Param().power();
            _power = param.power();
            _scale = param.scale();
            _shift = param.shift();
            _size = src[0]->Size();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T* pSrc = src[0]->CpuData();
            T* pDst = dst[0]->CpuData();
            if (_power == 1.0f)
                Detail::ScaleLayerForwardCpu(pSrc, &_scale, &_shift, 1, 1, _size, pDst, TensorFormatNchw, 0);
            else if (_scale == 1.0f && _shift == 0.0f)
                CpuPow(pSrc, _size, _power, pDst);
            else
                Detail::PowerForwardCpu(pSrc, _size, _scale, _shift, _power, pDst);
        }

    private:
        Type _power, _scale, _shift;
        size_t _size;
    };
}
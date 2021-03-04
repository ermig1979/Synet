/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
    template <class T> class BroadcastLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        BroadcastLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Size() == 1 && src[1]->Count() == 1);
            _fixed = this->Param().broadcast().fixed();
            Shape shape(src[1]->Size());
            if (src[1]->GetType() == TensorType64i)
            {
                for (size_t i = 0; i < src[1]->Size(); ++i)
                    shape[i] = (size_t)src[1]->As64i().CpuData()[i];
            }
            else if (src[1]->GetType() == TensorType32i)
            {
                for (size_t i = 0; i < src[1]->Size(); ++i)
                    shape[i] = (size_t)src[1]->As32i().CpuData()[i];
            }
            else
                assert(0);
            dst[0]->Reshape(shape, src[0]->CpuData()[0], src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_fixed)
                return;
            CpuSet(dst[0]->Size(), src[0]->CpuData()[0], dst[0]->CpuData());
        }

    private:
        bool _fixed;
    };
}
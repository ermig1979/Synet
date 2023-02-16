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

namespace Synet
{
    template <class T> class SqueezeLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SqueezeLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0] != dst[0]);
            Shape shape;
            if (src.size() > 1)
            {
                size_t nhwc[4] = { 0, 3, 1, 2 };
                Shape remove(src[1]->Size());
                for (size_t i = 0; i < src[1]->Size(); ++i)
                {
                    switch (src[1]->GetType())
                    {
                    case TensorType32f:
                        remove[i] = ((int*)src[1]->CpuData())[i];
                        break;
                    case TensorType32i:
                        remove[i] = src[1]->As32i().CpuData()[i];
                        break;
                    case TensorType64i:
                        remove[i] = (size_t)src[1]->As64i().CpuData()[i];
                        break;
                    case TensorType64u:
                        remove[i] = (size_t)src[1]->As64u().CpuData()[i];
                        break;
                    default:
                        assert(0);
                    }
                    if (src[0]->Format() == TensorFormatNhwc && src[0]->Count() == 4)
                        remove[i] = nhwc[remove[i]];
                }
                for (size_t i = 0; i < src[0]->Count(); ++i)
                {
                    bool exist = false;
                    for (size_t j = 0; j < remove.size(); ++j)
                        if (remove[j] == i)
                            exist = true;
                    if(!exist || src[0]->Axis(i) != 1)
                        shape.push_back(src[0]->Axis(i));
                }
            }
            else
            {
                Ints axes = this->Param().squeeze().axes();
                if (axes.empty())
                {
                    for (size_t i = 0; i < src[0]->Count(); ++i)
                    {
                        if (src[0]->Axis(i) != 1 || (i == 0 && !this->IsBack()))
                            shape.push_back(src[0]->Axis(i));
                    }
                }
                else
                {
                    for (size_t i = 0; i < src[0]->Count(); ++i)
                    {
                        bool leave = true;
                        for (size_t a = 0; a < axes.size() && leave; ++a)
                            if (axes[a] == i)
                                leave = false;
                        if (src[0]->Axis(i) != 1 || leave)
                            shape.push_back(src[0]->Axis(i));
                    }
                }
            }
            dst[0]->ShareAs(*src[0], shape, src[0]->Format());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }
    };
}
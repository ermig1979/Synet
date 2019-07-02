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

namespace Synet
{
    template <class T> class CastLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        CastLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1 && dst.size() == 1);
            _srcType = src[0]->GetType();
            _dstType = this->Param().cast().type();
            if (_srcType == _dstType)
                dst[0]->Share(*src[0]);
            else
            {
                switch (_dstType)
                {
                case TensorType32f:
                    dst[0]->As32f().Reshape(src[0]->Shape(), src[0]->Format());
                    break;
                case TensorType32i:
                    dst[0]->As32i().Reshape(src[0]->Shape(), src[0]->Format());
                    break;
                default:
                    assert(0);
                }
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            if (_srcType != _dstType)
            {
                if (_srcType == TensorType32f && _dstType == TensorType32i)
                {
                    const Synet::Tensor<float> & src0 = src[0]->As32f();
                    Synet::Tensor<int32_t> & dst0 = dst[0]->As32i();
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = (int32_t)src0.CpuData()[i];
                }
                else if (_srcType == TensorType32i && _dstType == TensorType32f)
                {
                    const Synet::Tensor<int32_t> & src0 = src[0]->As32i();
                    Synet::Tensor<float> & dst0 = dst[0]->As32f();
                    for (size_t i = 0; i < src0.Size(); ++i)
                        dst0.CpuData()[i] = (float)src0.CpuData()[i];
                }
                else
                    assert(0);
            }
        }

    private:
        TensorType _srcType, _dstType;
    };
}
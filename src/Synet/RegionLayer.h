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
#include "Synet/SoftmaxLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    namespace Detail
    {
        template<class T> void FlattenCpu(const T * src, size_t size, size_t layers, size_t batch, T * dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < layers; ++c)
                {
                    for (size_t i = 0; i < size; ++i)
                    {
                        size_t i1 = b*layers*size + c*size + i;
                        size_t i2 = b*layers*size + i*layers + c;
                        dst[i2] = src[i1];
                    }
                }
            }
        }
    }

    template <class T, template<class> class A> class RegionLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        RegionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            const RegionParam & param = this->Param().region();
            _coords = param.coords();
            _classes = param.classes();
            _num = param.num();
            _softmax = param.softmax();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src[0]->Axis(1) == _num*(_coords + _classes + 1));
            dst[0]->Reshape(src[0]->Shape());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            size_t size = _coords + _classes + 1;
            size_t batch = src[0]->Axis(0);
            size_t height = src[0]->Axis(2);
            size_t width = src[0]->Axis(3);
            size_t outputs = src[0]->Size(1);
            T * pDst = dst[0]->Data();
            Detail::FlattenCpu(src[0]->Data(), width*height, size*_num, batch, pDst);


            for (size_t b = 0; b < batch; ++b) 
            {
                for (size_t i = 0; i < height*width*_num; ++i)
                {
                    size_t index = size*i + b*outputs;
                    CpuSigmoid(pDst + index + 4, 1, pDst + index + 4);
                }
            }

            if (_softmax) 
            {
                for (size_t b = 0; b < batch; ++b) 
                {
                    for (size_t i = 0; i < height*width*_num; ++i)
                    {
                        size_t index = size*i + b*outputs;
                        T buffer;
                        Detail::SoftmaxLayerForwardCpu(pDst + index + 5, _classes, 1, &buffer, pDst + index + 5);
                    }
                }
            }
        }

    private:

        size_t _coords, _classes, _num;
        bool _softmax;
    };
}
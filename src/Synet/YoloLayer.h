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

namespace Synet
{
    template <class T> class YoloLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        YoloLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const YoloParam & param = this->Param().yolo();
            _num = param.num();
            _total = param.total();
            _classes = param.classes();
            _anchors.resize(param.anchors().size());
            for (size_t i = 0; i < param.anchors().size(); ++i)
                _anchors[i] = param.anchors()[i];
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            Shape dstShape = src[0]->Shape();
            dstShape[1] = _num*(_classes + 4 + 1);
            dst[0]->Reshape(dstShape);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            size_t batch = src[0]->Axis(0);
            size_t area = src[0]->Axis(2)*src[0]->Axis(3);
            Index index(4, 0);
            for (index[0] = 0; index[0] < batch; ++index[0])
            {
                for (size_t n = 0; n < _num; ++n)
                {
                    index[1] = n*(_classes + 4 + 1);
                    CpuSigmoid(src[0]->CpuData(index), 2 * area, dst[0]->CpuData(index));
                    index[1] += 2;
                    CpuCopy(src[0]->CpuData(index), 2 * area, dst[0]->CpuData(index));
                    index[1] += 2;
                    CpuSigmoid(src[0]->CpuData(index), (_classes + 1) * area, dst[0]->CpuData(index));
                }
            }
        }

    private:
        typedef std::vector<Type> Vector;

        size_t _total, _num, _classes;
        Vector _anchors;
    };
}
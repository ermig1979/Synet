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
    template <class T, template<class> class A> class ReshapeLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReshapeLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0] != dst[0]);
            const ReshapeParam & param = this->Param().reshape();
            const Shape & shape = param.shape();
            _copyAxes.clear();
            _inferredAxis = -1;
            _constantCount = 1;
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (shape[i] == 0)
                    _copyAxes.push_back(i);
                else if (shape[i] == -1)
                {
                    assert(_inferredAxis == -1);
                    _inferredAxis = i;
                }
                else
                    _constantCount *= shape[i];
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ReshapeParam & param = this->Param().reshape();
            const ptrdiff_t inputStartAxis = param.axis();
            const ptrdiff_t startAxis = (inputStartAxis >= 0) ? inputStartAxis : src[0]->Count() + inputStartAxis + 1;
            assert(startAxis >= 0 && startAxis <= (ptrdiff_t)src[0]->Count());
            const ptrdiff_t numAxes = param.numAxes();
            assert(numAxes >= -1);
            const ptrdiff_t endAxis = (numAxes == -1) ? src[0]->Count() : (startAxis + numAxes);
            assert(endAxis <= (ptrdiff_t)src[0]->Count());
            const ptrdiff_t numAxesReplaced = endAxis - startAxis;
            const ptrdiff_t numAxesRetained = src[0]->Count() - numAxesReplaced;
            Shape shape(numAxesRetained + param.shape().size());
            int topShapeIndex = 0;
            for (ptrdiff_t i = 0; i < startAxis; ++i)
                shape[topShapeIndex++] = src[0]->Axis(i);
            for (size_t i = 0; i < param.shape().size(); ++i)
                shape[topShapeIndex++] = param.shape()[i];
            for (size_t i = endAxis; i < src[0]->Count(); ++i)
                shape[topShapeIndex++] = src[0]->Axis(i);
            assert(topShapeIndex == shape.size());
            for (size_t i = 0; i < _copyAxes.size(); ++i)
            {
                assert(src[0]->Count() >= startAxis + _copyAxes[i]);
                shape[startAxis + _copyAxes[i]] = src[0]->Axis(startAxis + _copyAxes[i]);
            }
            if (_inferredAxis >= 0)
            {
                size_t explicitCount = _constantCount;
                explicitCount *= src[0]->Size(0, startAxis);
                explicitCount *= src[0]->Size(endAxis);
                for (size_t i = 0; i < _copyAxes.size(); ++i)
                    explicitCount *= shape[startAxis + _copyAxes[i]];
                assert(0 == src[0]->Size() % explicitCount);
                shape[startAxis + _inferredAxis] = src[0]->Size() / explicitCount;
            }
            dst[0]->ShareAs(*src[0], shape);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }
    private:
        Shape _copyAxes;
        ptrdiff_t _inferredAxis, _constantCount;
    };
}
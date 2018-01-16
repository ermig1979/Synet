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
    struct InputLayerOptions : public LayerOptions
    {
        Shape shape;

        InputLayerOptions(const String & n, const Shape & s)
            : LayerOptions(LayerOptions::InputLayer, n)
            , shape(s)
        {
        }
    };

    template <class T, template<class> class Allocator = std::allocator> class InputLayer : public Synet::Layer<T, Allocator>
    {
    public:
        typedef T Type;
        typedef Layer<T, Allocator> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        InputLayer(const InputLayerOptions & options)
            : Base(options)
            , _options(options)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst) {}

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst);

        virtual inline size_t SrcNum() const { return 0; }
        virtual inline size_t DstMin() const { return 1; }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst) {}

    private:
        InputLayerOptions _options;
    };
}
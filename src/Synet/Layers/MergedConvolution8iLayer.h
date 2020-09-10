/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/MergedConvolutionLayer.h"

namespace Synet
{
    namespace Detail
    {
    }

    template <class T> class MergedConvolution8iLayer : public MergedConvolutionLayer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;       
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolution8iLayer(const LayerParam & param, QuantizationMethod method)
            : MergedConvolutionLayer<T>(param)
            , _method(method)
        {
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _mergedConvolution8i.InternalBufferSize() +
                + _weight8i[0].MemoryUsage() + _norm32f[0].MemoryUsage() + _bias32f[0].MemoryUsage()
                + _weight8i[1].MemoryUsage() + _norm32f[1].MemoryUsage() + _bias32f[1].MemoryUsage();
        }

    protected:

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
        {

        }

        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
            if (_mergedConvolution8i.Enable())
                _mergedConvolution8i.Forward(src, buf, dst);
            else
            {

            }
        }

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u;
        Converter _srcCvt, _intCvt, _dstCvt;
        Tensor8i _weight8i[2];
        Tensor32f _norm32f[2], _bias32f[2];

        MergedConvolution8i _mergedConvolution8i;
    };
}
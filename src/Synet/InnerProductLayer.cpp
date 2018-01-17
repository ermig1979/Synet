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

#include "Synet/InnerProductLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> void InnerProductLayer<T, A>::Setup(const InnerProductLayer::TensorPtrs & src, const InnerProductLayer::TensorPtrs & dst)
    {
        _biasTerm = _options.biasTerm;
        _transpose = _options.transpose;
        _N = _options.outputNum;
        _K = src[0]->AxisSize(_options.axis);
        if (this->_tensors.empty())
        {
            if (_biasTerm)
                this->_tensors.resize(2);
            else
                this->_tensors.resize(1);
            Shape weightShape(2);
            if (_transpose) 
            {
                weightShape[0] = _K;
                weightShape[1] = _N;
            }
            else 
            {
                weightShape[0] = _N;
                weightShape[1] = _K;
            }
            this->_tensors[0].reset(new Tensor(weightShape));
            if (_biasTerm)
            {
                Shape biasShape(1, _N);
                this->_tensors[1].reset(new Tensor(biasShape));
            }
        }
    }

    template <class T, template<class> class A> void InnerProductLayer<T, A>::Reshape(const InnerProductLayer::TensorPtrs & src, const InnerProductLayer::TensorPtrs & dst)
    {
        const size_t newK = src[0]->AxisSize(_options.axis);
        _M = src[0]->Size(0, _options.axis);
        Shape dstShape = src[0]->GetShape();
        dstShape.resize(_options.axis + 1);
        dstShape[_options.axis] = _N;
        dst[0]->Reshape(dstShape);
        if (_biasTerm)
        {
            Shape biasShape(1, _M);
            _biasMultiplier.Reshape(biasShape, Type(1.0f));
        }
    }

    template <class T, template<class> class A> void InnerProductLayer<T, A>::ForwardCpu(const InnerProductLayer::TensorPtrs & src, const InnerProductLayer::TensorPtrs & dst)
    {
        CpuGemm<Type>(CblasNoTrans, _transpose ? CblasNoTrans : CblasTrans, _M, _N, _K, 
            (Type)1.0, src[0]->Data(), this->_tensors[0]->Data(), (Type)0.0, dst[0]->Data());
        if (_biasTerm) 
        {
            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _M, _N, 1,
                (Type)1.0, _biasMultiplier.Data(), this->_tensors[1]->Data(), (Type)1.0, dst[0]->Data());
        }
    }

    SYNET_CLASS_INSTANCE(InnerProductLayer);
}
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
            assert(this->_add == 0);

            const ConvParam * conv = this->_conv;
            _src8u = src->GetType() == TensorType8u;
            _dst8u = dst->GetType() == TensorType8u;
            Shape shape = conv[this->_count - 1].DstShape(this->_batch);
            if (_dst8u)
                dst->As8u().Reshape(shape, src->Format());
            else
                dst->As32f().Reshape(shape, src->Format());

            _mergedConvolution8i.Init(this->_batch, this->_conv, this->_count, _method);
            if (_mergedConvolution8i.Enable())
            {
                Base::Extend8u(buf, 0, Shp(_mergedConvolution8i.ExternalBufferSize()));
                const float* stats[6] = { 
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                    this->Stats(1).empty() ? NULL : this->Stats(1).back()->min.data(),
                    this->Stats(1).empty() ? NULL : this->Stats(1).back()->max.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data()};
                _mergedConvolution8i.SetParams(this->_weight, this->_internal, this->_bias, this->_params, stats);

            }
            else
            {
                if (!conv[0].IsDepthwise())
                {
                    if (!_src8u)
                        Base::Extend8u(buf, 0, src->Shape());
                    if(!conv[0].Is1x1())
                        Base::Extend8u(buf, 1, Shp(conv[0].ImgSize()));
                    Base::Extend32i(buf, 0, conv[0].DstShape(1));
                    Base::Extend32f(buf, 0, conv[0].DstShape(1));
                    if (this->_count == 3)
                    {
                        Base::Extend32f(buf, 1, conv[1].DstShape(1));
                        Base::Extend8u(buf, 0, conv[1].DstShape(1));
                        Base::Extend32i(buf, 0, conv[2].DstShape(1));
                    }
                    if(_dst8u)
                        Base::Extend32f(buf, 1, shape, src->Format());//?
                }
                if (_dst8u)
                    Base::Extend32f(buf, 0, shape, src->Format());
            //    Quantize();
            }
            //alg.internal = 1;
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
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
    template <class T> class DeconvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        DeconvolutionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1);

            const ConvolutionParam & param = this->Param().convolution();
            const Tensors & weight = this->Weight();

            _conv.Set(param);
            _conv.Set(*src[0], false);

            _is1x1 = _conv.Is1x1();
            _biasTerm = param.biasTerm();
            if (_biasTerm)
                assert(weight[1].Size() == _conv.dstC);

            _params[0] = param.activationParam0();
            _params[1] = param.activationParam1();

            assert(weight.size() == 1 + _biasTerm + (_conv.activation == ActivationFunctionTypePrelu));
            if (_conv.activation == ActivationFunctionTypePrelu)
            {
                if (weight.back().Size() == 1)
                {
                    _conv.activation = ActivationFunctionTypeLeakyRelu;
                    _params[0] = weight.back().CpuData()[0];
                }
                else
                    assert(weight.back().Size() == _conv.dstC);
            }

            _axis = param.axis();
            assert(src[0]->Count() == _axis + 3);

            _num = src[0]->Size(0, _axis);
            _trans = src[0]->Format() == TensorFormatNhwc;
            assert(weight[0].Shape() == _conv.WeightShape(_trans != 0) && weight[0].Format() == src[0]->Format());

            Shape dstShape(src[0]->Shape().begin(), src[0]->Shape().begin() + _axis);
            if (_trans)
            {
                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);
                dstShape.push_back(_conv.dstC);

            }
            else
            {
                dstShape.push_back(_conv.dstC);
                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);

            }

            buf[TensorType32f*BUFFER_COUNT]->Extend(Shape({ _conv.kernelY * _conv.kernelX * _conv.srcC, _conv.dstH * _conv.dstW }));

            dst[0]->Reshape(dstShape, src[0]->Format());

            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

    private:
        bool _is1x1, _biasTerm;
        int _trans;
        ConvParam _conv;
        size_t _axis, _num, _srcSize, _dstSize, _ldW, _ldS, _ldD, _grW, _grS, _grD, _siW, _siS, _siD;
        float _params[2];
    };
}
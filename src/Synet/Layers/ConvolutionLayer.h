/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Winograd.h"
#include "Synet/Utils/Convolution.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/HswishLayer.h"
#include "Synet/Layers/HardSigmoidLayer.h"

namespace Synet
{
    template <class T> class ConvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;

        ConvolutionLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
            _alg.internal = 0;
        }

        virtual int64_t Flop() const
        {
            return _alg.batch* _conv.kernelY* _conv.kernelX* _conv.srcC* _conv.dstH* _conv.dstW* _conv.dstC / _conv.group * 2;
        }

        virtual void CompactWeight()
        {
            if (_alg.internal)
                ((Tensor&)this->Weight()[0]).Clear();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1 && src[0]->Count() == 4);

            const ConvolutionParam & param = this->Param().convolution();
            const Tensors & weight = this->Weight();

            _conv.Set(param);
            _conv.Set(*src[0], *dst[0], true, param.autoPad());

            _alg.is1x1 = _conv.Is1x1() ? 1 : 0;
            _alg.bias = param.biasTerm() ? 1 : 0;
            if (_alg.bias)
                assert(weight[1].Size() == _conv.dstC);

            _alg.params[0] = param.activationParam0();
            _alg.params[1] = param.activationParam1();

            assert(weight.size() == 1 + _alg.bias + (_conv.activation == ActivationFunctionTypePrelu));
            if (_conv.activation == ActivationFunctionTypePrelu)
            {
                if (weight.back().Size() == 1)
                {
                    _conv.activation = ActivationFunctionTypeLeakyRelu;
                    _alg.params[0] = weight.back().CpuData()[0];
                }
                else
                    assert(weight.back().Size() == _conv.dstC);
            }

            _alg.batch = src[0]->Axis(0);
            _alg.trans = _conv.Trans();
            assert(weight[0].Shape() == _conv.WeightShape(_alg.trans != 0, true) && weight[0].Format() == src[0]->Format());

            if (_alg.trans)
            {
                _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
                _alg.ldW = _conv.dstC;
                _alg.grW = _conv.dstC / _conv.group;

                _alg.siS = _conv.dstH * _conv.dstW;
                _alg.ldS = _alg.siW * (_alg.is1x1 ? _conv.group : 1);
                _alg.grS = _alg.siW * (_alg.is1x1 ? 1 : _alg.siS);

                _alg.siD = _conv.dstC / _conv.group;
                _alg.ldD = _conv.dstC;
                _alg.grD = _alg.siD;
            }
            else
            {
                _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
                _alg.ldW = _alg.siW;
                _alg.grW = _conv.dstC * _alg.siW / _conv.group;

                _alg.siS = _conv.dstH * _conv.dstW;
                _alg.ldS = _alg.siS;
                _alg.grS = _alg.siS * _alg.siW;

                _alg.siD = _conv.dstC / _conv.group;
                _alg.ldD = _conv.dstH * _conv.dstW;
                _alg.grD = _alg.siD * _alg.siS;
            }

            Reshape(src[0], buf, dst[0]);

            _alg.sSize = src[0]->Size(1);
            _alg.dSize = dst[0]->Size(1);
            std::stringstream desc;
            desc << _alg.batch << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
            desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
            desc << "-" << Max(_conv.dilationY, _conv.dilationX) << "-" << Max(_conv.strideY, _conv.strideX);
            desc << "-" << _conv.group << InternalInfo();
            this->UsePerfStat(desc.str(), Flop());
        }

    protected:

        virtual void Reshape(const TensorPtr & src, const TensorPtrs& buf, const TensorPtr & dst) = 0;
        virtual String InternalInfo() const = 0;

    protected:
        ConvParam _conv;
        struct AlgParam
        {
            int is1x1, bias, trans, internal;
            size_t batch, sSize, dSize, ldW, ldS, ldD, grW, grS, grD, siW, siS, siD;
            float params[2];
        } _alg;
    };
}
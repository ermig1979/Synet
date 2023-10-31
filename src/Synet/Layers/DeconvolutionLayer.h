/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Deconvolution.h"
#include "Synet/Layers/ActivationLayers.h"

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

        DeconvolutionLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
            _transW = false;
            _internal = 0;
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + (_deconvolution32f.InternalBufferSize() + _weightT.Size())*sizeof(Type);
        }

        virtual int64_t Flop() const
        {
            return _num * _conv.kernelY * _conv.kernelX * _conv.srcC * _conv.srcH * _conv.srcW * _conv.dstC / _conv.group * 2;
        }

        virtual void CompactWeight()
        {
            if (_internal || _transW)
                ((Tensor&)this->Weight()[0]).Clear();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1);

            const ConvolutionParam & param = this->Param().convolution();
            const Tensors & weight = this->Weight();

            _conv.Set(param);
            _conv.Set(*src[0], *dst[0], false, false);

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
            assert(weight[0].Shape() == _conv.WeightShape(_trans != 0, false) && weight[0].Format() == src[0]->Format());

            Shape dstShape(src[0]->Shape().begin(), src[0]->Shape().begin() + _axis);
            if (_trans)
            {
                assert(_conv.group == 1);

                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);
                dstShape.push_back(_conv.dstC);

                _siW = _conv.srcC / _conv.group;
                _ldW = _conv.kernelY * _conv.kernelX * _conv.dstC / _conv.group;
                _grW = 0;// _conv.dstC / _conv.group;

                _siS = _conv.srcH * _conv.srcW;
                _ldS = _siW;
                _grS = 0;//_siS * _siW;

                _siD = _conv.kernelY * _conv.kernelX * _conv.dstC / _conv.group;
                _ldD = _siD;
                _grD = 0;//_siD;
            }
            else
            {
                dstShape.push_back(_conv.dstC);
                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);

                _transW = true;

                _siW = _conv.srcC / _conv.group;
                _ldW = _transW ? _siW : _conv.dstC * _conv.kernelY * _conv.kernelX / _conv.group;
                _grW = _siW * _conv.dstC * _conv.kernelY * _conv.kernelX / _conv.group;

                _siS = _conv.srcH * _conv.srcW;
                _ldS = _siS;
                _grS = _siS * _siW;

                _siD = _conv.dstC * _conv.kernelY * _conv.kernelX / _conv.group;
                _ldD = _conv.srcH * _conv.srcW;
                _grD = _siD * _siS;
            }

            _deconvolution32f.Init(_num, &_conv);
            if (_deconvolution32f.Enable())
            {
                buf[TensorType32f*BUFFER_COUNT]->Extend({ _deconvolution32f.ExternalBufferSize() });
                _deconvolution32f.SetParams(weight[0].CpuData(), &_internal, _biasTerm ? weight[1].CpuData() : NULL,
                    _conv.activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : _params);
            }
            else
            {
                buf[TensorType32f*BUFFER_COUNT]->Extend(Shape({ _conv.dstC * _conv.kernelY * _conv.kernelX * _conv.srcH * _conv.srcW }));
                if (_transW)
                {
                    const Shape & shape = weight[0].Shape();
                    _weightT.Reshape({ shape[1], shape[2], shape[3], shape[0] });
                    size_t m = shape[0], n = shape[1] * shape[2] * shape[3];
                    const T * s = weight[0].CpuData();
                    T * d = _weightT.CpuData();
                    for (size_t i = 0; i < m; ++i)
                        for (size_t j = 0; j < n; ++j)
                            d[j*m + i] = s[i * n + j];
                }
            }
            dst[0]->Reshape(dstShape, src[0]->Format());
            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
            std::stringstream desc;
            desc << _num << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
            desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
            desc << "-" << Max(_conv.strideY, _conv.strideX) << "-" << _conv.group;
            if (_deconvolution32f.Enable())
                desc << " " << _deconvolution32f.Info();
            this->UsePerfStat(desc.str(), Flop());
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            ForwardCpu(src[0]->CpuData(), buf[TensorType32f*BUFFER_COUNT]->CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
            if (_deconvolution32f.Enable())
                _deconvolution32f.Forward(src, buf, dst);
            else
            {
                const Type * weight = _transW ? _weightT.CpuData() : this->Weight()[0].CpuData();
                for (size_t n = 0; n < _num; ++n)
                {
                    Type * tmp = _is1x1 ? dst : buf;
                    if (_trans)
                    {
                        assert(_conv.group == 1);// || _conv.group == _conv.srcC);
                        for (size_t g = 0; g < _conv.group; ++g)
                            CpuGemm(CblasNoTrans, CblasNoTrans, _siS, _siD, _siW, Type(1), src + _grS * g, _ldS, weight + _grW * g, _ldW, Type(0), tmp + _grD * g, _ldD);
                    }
                    else
                    {
                        for (size_t g = 0; g < _conv.group; ++g)
                            CpuGemm(_transW ? CblasNoTrans : CblasTrans, CblasNoTrans, _siD, _siS, _siW,
                                Type(1), weight + _grW * g, _ldW, src + _grS * g, _ldS, Type(0), tmp + _grD * g, _ldD);
                    }
                    if (!_is1x1)
                    {
                        if (_trans)
                        {
                            Synet::RowToImg(tmp, _conv.dstH, _conv.dstW, _conv.dstC, _conv.kernelY, _conv.kernelX,
                                _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, _conv.group, (const Type*)NULL, dst);
                        }
                        else
                            Synet::ColToImg(tmp, _conv.dstC, _conv.dstH, _conv.dstW, _conv.kernelY, _conv.kernelX,
                                _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, (const Type*)NULL, dst);
                    }
                    if (_biasTerm)
                        CpuAddBias(this->Weight()[1].CpuData(), _conv.dstC, _conv.dstH*_conv.dstW, dst, _trans);
                    switch (_conv.activation)
                    {
                    case ActivationFunctionTypeIdentity:
                        break;
                    case ActivationFunctionTypeRelu:
                        CpuRelu(dst, _dstSize, 0.0f, dst);
                        break;
                    case ActivationFunctionTypeLeakyRelu:
                        CpuRelu(dst, _dstSize, _params[0], dst);
                        break;
                    case ActivationFunctionTypeRestrictRange:
                        CpuRestrictRange(dst, _dstSize, _params[0], _params[1], dst);
                        break;
                    case ActivationFunctionTypePrelu:
                        Detail::PreluLayerForwardCpu(dst, this->Weight().back().CpuData(), _conv.dstC, _conv.dstH * _conv.dstW, dst, _trans);
                        break;
                    case ActivationFunctionTypeElu:
                        CpuElu(dst, _dstSize, _params[0], dst);
                        break;
                    case ActivationFunctionTypeHswish:
                        CpuHswish(dst, _dstSize, _params[0], _params[1], dst);
                        break;
                    case ActivationFunctionTypeMish:
                        CpuMish(dst, _dstSize, _params[0], dst);
                        break;
                    case ActivationFunctionTypeHardSigmoid:
                        CpuHardSigmoid(dst, _dstSize, _params[0], _params[1], dst);
                        break;
                    case ActivationFunctionTypeSwish:
                        CpuSwish(dst, _dstSize, dst);
                        break;
                    default:
                        assert(0);
                    }
                    src += _srcSize;
                    dst += _dstSize;
                }
            }
        }

    private:
        bool _is1x1, _biasTerm, _transW;
        int _trans, _internal;
        ConvParam _conv;
        size_t _axis, _num, _srcSize, _dstSize, _ldW, _ldS, _ldD, _grW, _grS, _grD, _siW, _siS, _siD;
        float _params[2];

        Deconvolution32f _deconvolution32f;

        Tensor _weightT;
    };
}
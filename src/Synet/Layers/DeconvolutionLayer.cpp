/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/DeconvolutionLayer.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    DeconvolutionLayer::DeconvolutionLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
        _transW = false;
        _internal = 0;
    }

    size_t DeconvolutionLayer::MemoryUsage() const
    {
        return Base::MemoryUsage() + (_deconvolution32f.InternalBufferSize() + _weightT.Size())*sizeof(Type);
    }

    int64_t DeconvolutionLayer::Flop() const
    {
        return _num * _conv.kernelY * _conv.kernelX * _conv.srcC * _conv.srcH * _conv.srcW * _conv.dstC / _conv.group * 2;
    }

    void DeconvolutionLayer::CompactWeight()
    {
        if (_internal || _transW)
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool DeconvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("DeconvolutionLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("DeconvolutionLayer supports only 32f input!");

        const ConvolutionParam & param = this->Param().convolution();
        const Tensors & weight = this->Weight();

        if(!_conv.Set(param))
            SYNET_ERROR("DeconvolutionLayer has invalid parameters!");
        if (!_conv.Set(*src[0], *dst[0], false, false))
            return false;

        _is1x1 = _conv.Is1x1();
        _biasTerm = param.biasTerm();
        if (_biasTerm)
        {
            if(weight.size() < 2 || weight[1].Size() != _conv.dstC)
                SYNET_ERROR("DeconvolutionLayer has invalid weight[1] parameter!");
        }

        _params[0] = param.activationParam0();
        _params[1] = param.activationParam1();

        if(weight.size() != 1 + _biasTerm + (_conv.activation == ActivationFunctionTypePrelu))
            SYNET_ERROR("DeconvolutionLayer has invalid weight number!");
        if (_conv.activation == ActivationFunctionTypePrelu)
        {
            if (weight.back().Size() == 1)
            {
                _conv.activation = ActivationFunctionTypeLeakyRelu;
                _params[0] = weight.back().CpuData()[0];
            }
            else
            {
                if(weight.back().Size() != _conv.dstC)
                    SYNET_ERROR("DeconvolutionLayer has invalid last weight size!");
            }
        }

        _axis = src[0]->Index(param.axis());
        if(src[0]->Count() != _axis + 3)
            SYNET_ERROR("DeconvolutionLayer has invalid input shape for given axis " << param.axis() << " parameter!");

        _num = src[0]->Size(0, _axis);
        _trans = src[0]->Format() == TensorFormatNhwc;
        if(weight[0].Shape() != _conv.WeightShape(_trans != 0, false) || weight[0].Format() != src[0]->Format())
            SYNET_ERROR("DeconvolutionLayer has invalid input weigth[0] size " << ToStr(weight[0].Shape()) << " instead of " << ToStr(_conv.WeightShape(_trans != 0, false)) << " !");

        Shape dstShape(src[0]->Shape().begin(), src[0]->Shape().begin() + _axis);
        if (_trans)
        {
            if(_conv.group != 1)
                SYNET_ERROR("DeconvolutionLayer does not support group != 1 for NHWC format!");

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
            _deconvolution32f.SetParams(weight[0].Data<float>(), &_internal, _biasTerm ? weight[1].Data<float>() : NULL,
                _conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _params);
        }
        else
        {
            buf[TensorType32f*BUFFER_COUNT]->Extend(Shape({ _conv.dstC * _conv.kernelY * _conv.kernelX * _conv.srcH * _conv.srcW }));
            if (_transW)
            {
                const Shape & shape = weight[0].Shape();
                _weightT.Reshape(TensorType32f, Shp(shape[1], shape[2], shape[3], shape[0]), src[0]->Format());
                size_t m = shape[0], n = shape[1] * shape[2] * shape[3];
                const float * s = weight[0].Data<float>();
                float * d = _weightT.Data<float>();
                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        d[j*m + i] = s[i * n + j];
            }
        }
        dst[0]->Reshape(TensorType32f, dstShape, src[0]->Format());
        _srcSize = src[0]->Size(_axis);
        _dstSize = dst[0]->Size(_axis);

        std::stringstream desc;
        desc << _num << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW;
        desc << "-" << _conv.dstC << "x" << _conv.kernelY << "x" << _conv.kernelX;
        desc << "-" << Max(_conv.strideY, _conv.strideX) << "-" << _conv.group;
        if (_deconvolution32f.Enable())
            desc << " " << _deconvolution32f.Info();
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }

    void DeconvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        ForwardCpu(src[0]->Data<float>(), buf[TensorType32f*BUFFER_COUNT]->Data<float>(), dst[0]->Data<float>());
    }

    void DeconvolutionLayer::ForwardCpu(const float* src, float* buf, float* dst)
    {
        if (_deconvolution32f.Enable())
            _deconvolution32f.Forward(src, buf, dst);
        else
        {
            const float* weight = _transW ? _weightT.CpuData() : this->Weight()[0].CpuData();
            for (size_t n = 0; n < _num; ++n)
            {
                float* tmp = _is1x1 ? dst : buf;
                if (_trans)
                {
                    assert(_conv.group == 1);
                    for (size_t g = 0; g < _conv.group; ++g)
                        CpuGemm(CblasNoTrans, CblasNoTrans, _siS, _siD, _siW, 1.0f, src + _grS * g, _ldS, weight + _grW * g, _ldW, 0.0f, tmp + _grD * g, _ldD);
                }
                else
                {
                    for (size_t g = 0; g < _conv.group; ++g)
                        CpuGemm(_transW ? CblasNoTrans : CblasTrans, CblasNoTrans, _siD, _siS, _siW,
                            1.0f, weight + _grW * g, _ldW, src + _grS * g, _ldS, 0.0f, tmp + _grD * g, _ldD);
                }
                if (!_is1x1)
                {
                    if (_trans)
                    {
                        Synet::RowToImg(tmp, _conv.dstH, _conv.dstW, _conv.dstC, _conv.kernelY, _conv.kernelX,
                            _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, _conv.group, (const float*)NULL, dst);
                    }
                    else
                        Synet::ColToImg(tmp, _conv.dstC, _conv.dstH, _conv.dstW, _conv.kernelY, _conv.kernelX,
                            _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, (const float*)NULL, dst);
                }
                if (_biasTerm)
                    CpuAddBias(this->Weight()[1].Data<float>(), _conv.dstC, _conv.dstH * _conv.dstW, dst, _trans);
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
                    PreluLayerForward(dst, this->Weight().back().Data<float>(), _conv.dstC, _conv.dstH * _conv.dstW, dst, _trans ? TensorFormatNhwc : TensorFormatNchw);
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
                case ActivationFunctionTypeGelu:
                    CpuGelu(dst, _dstSize, dst);
                    break;
                default:
                    assert(0);
                }
                src += _srcSize;
                dst += _dstSize;
            }
        }
    }
}
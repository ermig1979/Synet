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

#include "Synet/Layers/Deconvolution/Deconvolution32fLayer.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    Deconvolution32fLayer::Deconvolution32fLayer(const LayerParam & param, Context* context)
        : DeconvolutionLayer(param, context)
    {
        _transW = false;
    }

    size_t Deconvolution32fLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + (_deconvolution32f.InternalBufferSize() + _weightT.Size())*sizeof(float);
    }

    bool Deconvolution32fLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        if (src->GetType() != TensorType32f)
            SYNET_ERROR("DeconvolutionLayer supports only 32f input!");

        const Tensors& weight = this->Weight();

        Shape dstShape(src->Shape().begin(), src->Shape().begin() + _axis);
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
            buf[TensorType32f*BUFFER_COUNT]->Extend(TensorType32f, Shp(_deconvolution32f.ExternalBufferSize()));
            _deconvolution32f.SetParams(weight[0].Data<float>(), &_internal, _biasTerm ? weight[1].Data<float>() : NULL,
                _conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _params);
        }
        else
        {
            buf[TensorType32f*BUFFER_COUNT]->Extend(TensorType32f, Shp(_conv.dstC * _conv.kernelY * _conv.kernelX * _conv.srcH * _conv.srcW));
            if (_transW)
            {
                const Shape & shape = weight[0].Shape();
                _weightT.Reshape(TensorType32f, Shp(shape[1], shape[2], shape[3], shape[0]), src->Format());
                size_t m = shape[0], n = shape[1] * shape[2] * shape[3];
                const float * s = weight[0].Data<float>();
                float * d = _weightT.Data<float>();
                for (size_t i = 0; i < m; ++i)
                    for (size_t j = 0; j < n; ++j)
                        d[j*m + i] = s[i * n + j];
                _internal = true;
            }
        }
        dst->Reshape(TensorType32f, dstShape, src->Format());

        return true;
    }

    String Deconvolution32fLayer::InternalInfo() const
    {
        return String(" fp32") + (_deconvolution32f.Enable() ? String(" ") + _deconvolution32f.Info() : String());
    }

    void Deconvolution32fLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        ForwardCpu(src[0]->Data<float>(), buf[TensorType32f*BUFFER_COUNT]->Data<float>(), dst[0]->Data<float>());
    }

    void Deconvolution32fLayer::ForwardCpu(const float* src, float* buf, float* dst)
    {
        if (_deconvolution32f.Enable())
            _deconvolution32f.Forward(src, buf, dst);
        else
        {
            const float* weight = _transW ? _weightT.Data<float>() : this->Weight()[0].Data<float>();
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
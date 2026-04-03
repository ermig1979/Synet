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

#include "Synet/Utils/Gemm.h"
#include "Synet/Layers/InnerProduct/InnerProduct32fLayer.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Layers/Activation/PreluLayer.h"

namespace Synet
{
    InnerProduct32fLayer::InnerProduct32fLayer(const LayerParam & param, Context* context)
        : InnerProductLayer(param, context)
        , _internal(0)
    {

    }

    size_t InnerProduct32fLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage() + _innerProduct32f.InternalBufferSize() * sizeof(float);
    }

    void InnerProduct32fLayer::CompactWeight()
    {
        if (_internal)
            ((Tensor&)this->Weight()[0]).Clear();
    }

    bool InnerProduct32fLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (!InnerProductLayer::Reshape(src, buf, dst))
            return false;
        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(TensorType32f, dstShape, TensorFormatNchw);
        if (!_transA)
        {
            _innerProduct32f.Init(_M, _N, _K, _transB ? 0 : 1, src.size() == 1 ? 1 : 0, _biasTerm ? 1 : 0, _activation);
            if (_innerProduct32f.Enable())
            {
                const float* weight = src.size() == 1 ? this->Weight()[0].Data<float>() : NULL;
                size_t biasIndex = src.size() == 1 ? 1 : 0;
                const float* bias = _biasTerm ? this->Weight()[biasIndex].Data<float>() : NULL;
                size_t paramsIndex = biasIndex + (_biasTerm ? 1 : 0);
                const float* params = _activation == ActivationFunctionTypePrelu ? this->Weight()[paramsIndex].Data<float>() : _params;
                _innerProduct32f.SetParams(weight, &_internal, bias, params);
                Layer::Extend32f(buf, 0, Shp(_innerProduct32f.ExternalBufferSize()), src[0]->Format());
            }
        }
        return true;
    }

    void InnerProduct32fLayer::Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
        const float* A = src[0]->Data<float>();
        const float* B = src.size() > 1 ? src[1]->Data<float>() : NULL;
        float* buf0 = Layer::Buf32f(buf, 0);
        float* C = dst[0]->Data<float>();
        if (_innerProduct32f.Enable())
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _innerProduct32f.Forward(A, B, buf0, C);
                A += _M * _K;
                B += _K * _N;
                C += _M * _N;
            }
        }
        else if (src.size() > 1)
        {
            for(size_t b = 0; b < _batch; ++b)
            {
                Forward(A, B, C);
                A += _M * _K;
                B += _K * _N;
                C += _M * _N;
            }
        }
        else
        {
            const float* wgt = this->Weight()[0].Data<float>();
            Forward(A, wgt, C);
        }
    }

    void InnerProduct32fLayer::Forward(const float * src, const float* wgt, float* dst)
    {
        const float* bias = _biasTerm ? this->Weight()[1].Data<float>() : NULL;
        if (!_transB && _M == 1 && _activation == ActivationFunctionTypeIdentity)
            Detail::InnerProductLayerForwardCpu(src, wgt, bias, _N, _K, dst);
        else
        {
            size_t lds = _transA ? _M : _K;
            size_t ldw = _transB ? _N : _K;
            CpuGemm(_transA ? CblasTrans : CblasNoTrans, _transB ? CblasNoTrans : CblasTrans, _M, _N, _K, 1.0f, src, lds, wgt, ldw, 0.0f, dst, _N);
            if (_biasTerm)
            {
                for (size_t i = 0; i < _M; ++i)
                    CpuAddBias(bias, _N, 1, dst + i * _N);
            }
            switch (_activation)
            {
            case ActivationFunctionTypeIdentity:
                break;
            case ActivationFunctionTypeRelu:
                CpuRelu(dst, _M * _N, 0.0f, dst);
                break;
            case ActivationFunctionTypeLeakyRelu:
                CpuRelu(dst, _M * _N, _params[0], dst);
                break;
            case ActivationFunctionTypeRestrictRange:
                CpuRestrictRange(dst, _M * _N, _params[0], _params[1], dst);
                break;
            case ActivationFunctionTypePrelu:
                PreluLayerForward(dst, this->Weight().back().Data<float>(), _N, _M, dst, TensorFormatNchw);
                break;
            case ActivationFunctionTypeElu:
                CpuElu(dst, _M * _N, _params[0], dst);
                break;
            case ActivationFunctionTypeHswish:
                CpuHswish(dst, _M * _N, _params[0], _params[1], dst);
                break;
            case ActivationFunctionTypeMish:
                CpuMish(dst, _M * _N, _params[0], dst);
                break;
            case ActivationFunctionTypeHardSigmoid:
                CpuHardSigmoid(dst, _M * _N, _params[0], _params[1], dst);
                break;
            case ActivationFunctionTypeSwish:
                CpuSwish(dst, _M * _N, dst);
                break;
            case ActivationFunctionTypeGelu:
                CpuGelu(dst, _M * _N, dst);
                break;
            default:
                assert(0);
            }
        }
    }
}

/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/DynamicQuantizedInnerProductLayer.h"
#include "Synet/Layers/Quantized/DynamicQuantizeLinearLayer.h"
#include "Synet/Layers/Quantized/MatMulIntegerLayer.h"
#include "Synet/Layers/Math/ScaleLayer.h"
#include "Synet/Layers/Activation/PreluLayer.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    void DynamicQuantizedInnerProductLayerCast(const int32_t* src, size_t size, float* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = (float)src[i];
    }

    //--------------------------------------------------------------------------------------------------

    DynamicQuantizedInnerProductLayer::DynamicQuantizedInnerProductLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool DynamicQuantizedInnerProductLayer::Resizable() const
    {
        return false;
    }

    size_t DynamicQuantizedInnerProductLayer::MemoryUsage() const
    {
        size_t size = Layer::MemoryUsage();
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        size += _dynamicQuantizedInnerProduct.InternalBufferSize();
#endif
        return size;
    }

    int64_t DynamicQuantizedInnerProductLayer::Flop() const
    {
        return _M * _N * (_K * 2 + (_biasTerm ? 2 : 1));
    }

    void DynamicQuantizedInnerProductLayer::CompactWeight()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (_dynamicQuantizedInnerProduct.Enable())
        {
            for (size_t i = 0; i < this->Weight().size(); ++i)
                ((Tensor&)this->Weight()[i]).Clear();
        }
#endif
    }

    bool DynamicQuantizedInnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if ((src.size() != 1) || dst.size() != 1)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 input!");

        const LayerParam& param = this->Param();
        const InnerProductParam& ip = param.innerProduct();

        _biasTerm = ip.biasTerm();
        _activation = ip.activationType();
        _params[0] = ip.activationParam0();
        _params[1] = ip.activationParam1();
        Shape shape = src[0]->Shape();
        _K = src[0]->Size(-1);
        _M = src[0]->Size(0, -1);

        const Tensors& weight = ((Tensors&)this->Weight());
        size_t weightNumNeed = 3 + (_biasTerm ? 1 : 0) + (_activation == ActivationFunctionTypePrelu ? 1 : 0);
        if (weight.size() < weightNumNeed)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer must have at least " << weightNumNeed << " weights!");
        if (weight[0].GetType() != TensorType8i || weight[1].GetType() != TensorType8i)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only INT8 weight[0] and weight[1]!");
        if (weight[2].GetType() != TensorType32f)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 weight[2]!");
        _N = weight[0].Axis(-1);
        if (weight[0].Axis(0) != _K)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer: check src[0] and weight[0] size!");
        if (weight[1].Size(0) != _N || weight[2].Size(0) != _N)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer: check weight[1] and weight[2] size!");
        if (_biasTerm)
        {
            if (weight[3].GetType() != TensorType32f)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 weight[3]!");
        }
        if (_activation == ActivationFunctionTypePrelu)
        {
            if (weight.back().GetType() != TensorType32f)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 Prelu weight!");
            if (weight.back().Size() == 1)
            {
                _activation = ActivationFunctionTypeLeakyRelu;
                _params[0] = weight.back().Data<float>()[0];
            }
            else
            {
                if (weight.back().Size() != _N)
                    SYNET_ERROR("DynamicQuantizedInnerProductLayer: check weight[" << weight.size() - 1 << "] size!");
            }
        }

        shape.back() = _N;
        dst[0]->Reshape(TensorType32f, shape, src[0]->Format());

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        _dynamicQuantizedInnerProduct.Init(_M, _N, _K, _biasTerm ? SimdTrue : SimdFalse, (SimdConvolutionActivationType)_activation);
        if (_dynamicQuantizedInnerProduct.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_dynamicQuantizedInnerProduct.ExternalBufferSize()));
            _dynamicQuantizedInnerProduct.SetParams(weight[0].Data<int8_t>(), weight[2].Data<float>(),
                _biasTerm ? weight[3].Data<float>() : NULL, 
                _activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _params);
        }
        else
#endif
        {
            Layer::Extend8u(buf, 0, src[0]->Shape(), src[0]->Format());
            Layer::Extend32i(buf, 0, shape, src[0]->Format());
            Layer::Extend32f(buf, 0, Shp(_N), src[0]->Format());
        }

        std::stringstream desc;
        desc << _M << "x" << _K << "-" << _N << " ";
        desc << (_biasTerm ? "b" : "o");
        if (_activation)
            desc << "-" << ShortStr(_activation);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (_dynamicQuantizedInnerProduct.Enable())
            desc << " " << _dynamicQuantizedInnerProduct.Info();
#endif
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    void DynamicQuantizedInnerProductLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (_dynamicQuantizedInnerProduct.Enable())
            _dynamicQuantizedInnerProduct.Forward(src[0]->Data<float>(), Layer::Buf8u(buf, 0), dst[0]->Data<float>());
        else
#endif
        {
            const Tensors& weight = ((Tensors&)this->Weight());
            float scale;
            uint8_t zero;
            DynamicQuantizeLinearLayerForward(src[0]->Data<float>(), _M * _K, Layer::Buf8u(buf, 0), scale, zero);
            float* norm = Layer::Buf32f(buf, 0);
            for (size_t j = 0; j < _N; ++j)
                norm[j] = weight[2].Data<float>()[j] * scale;

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            const bool overflow16i = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#else
            const bool overflow16i = true;
#endif
            MatMulIntegerGemm(_M, _N, _K, Layer::Buf8u(buf, 0), zero, weight[0].Data<int8_t>(), Layer::Buf32i(buf, 0), overflow16i);

            DynamicQuantizedInnerProductLayerCast(Layer::Buf32i(buf, 0), _M * _N, dst[0]->Data<float>());

            const float* bias = _biasTerm ? weight[3].Data<float>() : NULL;
            ScaleForward32f(dst[0]->Data<float>(), norm, bias, _N, 1, _M, dst[0]->Data<float>(), TensorFormatNhwc, 0);

            Activation(dst[0]->Data<float>());
        }
    }

    void DynamicQuantizedInnerProductLayer::Activation(float* dst)
    {
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
            PreluLayerForward(dst, this->Weight().back().Data<float>(), _N, _M, dst, TensorFormatNhwc);
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
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
    const int ROW_QUANTIZATION = 0;

    //--------------------------------------------------------------------------------------------------

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
        size += _dynamicQuantizedInnerProduct32f.InternalBufferSize();
        size += _dynamicQuantizedInnerProduct16b.InternalBufferSize();
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
        if (_dynamicQuantizedInnerProduct32f.Enable() || _dynamicQuantizedInnerProduct16b.Enable())
        {
            for (size_t i = 0; i < this->Weight().size(); ++i)
                ((Tensor&)this->Weight()[i]).Clear();
        }
#endif
    }

    bool DynamicQuantizedInnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only 1 or 2 inputs and 1 output!");

        _constB = src.size() == 1;
        _typeA = src[0]->GetType();
        if (_typeA != TensorType32f && _typeA != TensorType16b)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 or BF16 input[0]!");
        _typeC = dst[0]->GetType();
        if (_typeC != TensorType32f && _typeC != TensorType16b)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 or BF16 output!");

        _K = src[0]->Size(-1);
        _M = src[0]->Size(0, -1);

        const InnerProductParam& ip = this->Param().innerProduct();
        _biasTerm = ip.biasTerm();
        _activation = ip.activationType();
        _params[0] = ip.activationParam0();
        _params[1] = ip.activationParam1();

        const Tensors& weight = ((Tensors&)this->Weight());

        size_t idx = 0, idxS = 0, idxB = 0, idxA = 0;
        if (_constB)
        {
            if (weight.empty())
                SYNET_ERROR("DynamicQuantizedInnerProductLayer with 1 input must have at least 1 weights!");
            _typeB = weight[0].GetType();
            if (_typeB != TensorType32f && _typeB != TensorType8i)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 or INT8 weight[0]!");
            _N = weight[0].Axis(-1);
            if (_N != ip.outputNum() || weight[0].Axis(0) != _K || (ip.outputNum() != _N && ip.outputNum() != 0))
                SYNET_ERROR("DynamicQuantizedInnerProductLayer: wrong weight[0] size!");
            idx++;
            if (_typeB == TensorType8i)
            {
                if (weight[idx++].GetType() != TensorType8i)
                    SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only INT8 weight[1]!");
                idxS = idx++;
                if (weight[idxS].GetType() != TensorType32f)
                    SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 weight[2]!");
                if (weight[1].Size(0) != _N || weight[idxS].Size(0) != _N)
                    SYNET_ERROR("DynamicQuantizedInnerProductLayer: check weight[1] and weight[2] size!");
            }
        }
        else
        {
            _typeB = src[1]->GetType();
            if (_typeB != TensorType32f && _typeB != TensorType16b)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 or BF16 input[1]!");
            if (src[1]->Size(0, -1) != _K)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer: wrong src[1] size!");
            _N = src[1]->Size(-1);
        }
        if (_biasTerm)
        {
            idxB = idx++;
            if (weight.size() < idx)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer must have at least " << idx << " weights!");
            if (weight[idxB].GetType() != TensorType32f)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 weight[2]!");
        }
        if (_activation == ActivationFunctionTypePrelu)
        {
            idxA = idx++;
            if (weight.size() < idx)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer must have at least " << idx << " weights!");
            if (weight[idxA].GetType() != TensorType32f)
                SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 Prelu weight!");
            if (weight[idxA].Size() == 1)
            {
                _activation = ActivationFunctionTypeLeakyRelu;
                _params[0] = weight[idxA].Data<float>()[0];
            }
            else
            {
                if (weight[idxA].Size() != _N)
                    SYNET_ERROR("DynamicQuantizedInnerProductLayer: check weight[" << idxA << "] size!");
            }
        }

        Shape shape = src[0]->Shape();
        shape.back() = _N;
        dst[0]->Reshape(_typeC, shape, TensorFormatNchw);

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (idxS == 2)
        {
            _dynamicQuantizedInnerProduct32f.Init(_M, _N, _K,  _biasTerm ? SimdTrue : SimdFalse, (SimdConvolutionActivationType)_activation);
        }
        else
        {
            _dynamicQuantizedInnerProduct16b.Init(_M, _N, _K, (SimdTensorDataType)_typeA, (SimdTensorDataType)_typeB, (SimdTensorDataType)_typeC,
                _constB ? SimdTrue : SimdFalse, _biasTerm ? SimdTrue : SimdFalse, (SimdConvolutionActivationType)_activation, SimdTrue);
        }
        if (_dynamicQuantizedInnerProduct32f.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_dynamicQuantizedInnerProduct32f.ExternalBufferSize()));
            _dynamicQuantizedInnerProduct32f.SetParams(weight[0].Data<int8_t>(), weight[idxS].Data<float>(),
                _biasTerm ? weight[idxB].Data<float>() : NULL, _activation == ActivationFunctionTypePrelu ? weight[idxA].Data<float>() : _params);
        }
        else if (_dynamicQuantizedInnerProduct16b.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_dynamicQuantizedInnerProduct16b.ExternalBufferSize()));
            _dynamicQuantizedInnerProduct16b.SetParams(
                _constB ? weight[0].RawData() : NULL, 
                (_constB && idxS) ? weight[idxS].Data<float>() : NULL,
                _biasTerm ? weight[idxB].Data<float>() : NULL, 
                _activation == ActivationFunctionTypePrelu ? weight[idxA].Data<float>() : _params);
        }
        else
#endif
        {
            Layer::Extend8u(buf, 0, src[0]->Shape(), TensorFormatNchw);
            Layer::Extend32i(buf, 0, shape, TensorFormatNchw);
            Layer::Extend32f(buf, 0, Shp(_N), TensorFormatNchw);
        }

        std::stringstream desc;
        desc << _M << "x" << _K << "-" << _N << " ";
        desc << ToChar(_typeA) << ToChar(_typeB) << ToChar(_typeC) << " ";
        desc << src.size() << (_biasTerm ? "b" : "o");
        if (_activation)
            desc << "-" << ShortStr(_activation);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (_dynamicQuantizedInnerProduct32f.Enable())
            desc << " " << _dynamicQuantizedInnerProduct32f.Info();
        if (_dynamicQuantizedInnerProduct16b.Enable())
            desc << " " << _dynamicQuantizedInnerProduct16b.Info();
#endif
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    void DynamicQuantizedInnerProductLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        if (_dynamicQuantizedInnerProduct32f.Enable())
            _dynamicQuantizedInnerProduct32f.Forward(src[0]->Data<float>(), Layer::Buf8u(buf, 0), dst[0]->Data<float>());
        else if (_dynamicQuantizedInnerProduct16b.Enable())
            _dynamicQuantizedInnerProduct16b.Forward(src[0]->RawData(), src.size() > 1 ? src[1]->RawData() : NULL, Layer::Buf8u(buf, 0), dst[0]->RawData());
        else
#endif
        {
            const Tensors& weight = ((Tensors&)this->Weight());
            float scale;
            uint8_t zero;
            if (ROW_QUANTIZATION)
            {
                const float* ps = src[0]->Data<float>();
                float* pd = dst[0]->Data<float>();
                for (size_t i = 0; i < _M; ++i)
                {
                    DynamicQuantizeLinearLayerForward(ps, 1 * _K, Layer::Buf8u(buf, 0), scale, zero);
                    float* norm = Layer::Buf32f(buf, 0);
                    for (size_t j = 0; j < _N; ++j)
                        norm[j] = weight[2].Data<float>()[j] * scale;

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
                    const bool overflow16i = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#else
                    const bool overflow16i = true;
#endif
                    MatMulIntegerGemm(1, _N, _K, Layer::Buf8u(buf, 0), zero, weight[0].Data<int8_t>(), Layer::Buf32i(buf, 0), overflow16i);

                    DynamicQuantizedInnerProductLayerCast(Layer::Buf32i(buf, 0), 1 * _N, pd);

                    const float* bias = _biasTerm ? weight[3].Data<float>() : NULL;
                    ScaleForward32f(pd, norm, bias, _N, 1, 1, pd, TensorFormatNhwc, 0);

                    Activation(pd, 1);

                    ps += _K; 
                    pd += _N;
                }
            }
            else
            {
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

                Activation(dst[0]->Data<float>(), _M);
            }
        }
    }

    void DynamicQuantizedInnerProductLayer::Activation(float* dst, size_t M)
    {
        switch (_activation)
        {
        case ActivationFunctionTypeIdentity:
            break;
        case ActivationFunctionTypeRelu:
            CpuRelu(dst, M * _N, 0.0f, dst);
            break;
        case ActivationFunctionTypeLeakyRelu:
            CpuRelu(dst, M * _N, _params[0], dst);
            break;
        case ActivationFunctionTypeRestrictRange:
            CpuRestrictRange(dst, M * _N, _params[0], _params[1], dst);
            break;
        case ActivationFunctionTypePrelu:
            PreluLayerForward(dst, this->Weight().back().Data<float>(), _N, M, dst, TensorFormatNhwc);
            break;
        case ActivationFunctionTypeElu:
            CpuElu(dst, M * _N, _params[0], dst);
            break;
        case ActivationFunctionTypeHswish:
            CpuHswish(dst, M * _N, _params[0], _params[1], dst);
            break;
        case ActivationFunctionTypeMish:
            CpuMish(dst, M * _N, _params[0], dst);
            break;
        case ActivationFunctionTypeHardSigmoid:
            CpuHardSigmoid(dst, M * _N, _params[0], _params[1], dst);
            break;
        case ActivationFunctionTypeSwish:
            CpuSwish(dst, M * _N, dst);
            break;
        case ActivationFunctionTypeGelu:
            CpuGelu(dst, M * _N, dst);
            break;
        default:
            assert(0);
        }
    }
}
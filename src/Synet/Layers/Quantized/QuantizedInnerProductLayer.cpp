/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/QuantizedInnerProductLayer.h"

#include "Synet/Utils/Gemm.h"

#include "Synet/Quantization/Gemm.h"
#include "Synet/Quantization/DequantizeLinear.h"
#include "Synet/Quantization/QuantizeLinear.h"

namespace Synet
{
    QuantizedInnerProductLayer::QuantizedInnerProductLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool QuantizedInnerProductLayer::Resizable() const
    {
        return false;
    }

    int64_t QuantizedInnerProductLayer::Flop() const
    {
        return _batch * _M * _N * (_K * 2 + (_biasTerm ? 1 : 0));
    }

    LowPrecisionType QuantizedInnerProductLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedInnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("QuantizedInnerProductLayer supports only 1 or 2 inputs and 1 output!");

        const LayerParam& param = this->Param();
        const InnerProductParam& ip = param.innerProduct();

        _biasTerm = ip.biasTerm();
        _transA = ip.transposeA();
        _transB = ip.transposeB();
        _axis = (int)src[0]->Index(ip.axis());
        _batch = 1;
        _K = src[0]->Size(_axis);

        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = param.qDst().size() && param.qDst()[0].type() == TensorType8u;

        if (src.size() == 2)
        {
            SYNET_ERROR("QuantizedInnerProductLayer 2 inputs support is not implemented!");
        }
        else
        {
            _N = this->Param().innerProduct().outputNum();
            _M = src[0]->Size(0, _axis);
            _batch = 1;
        }

        if (!(Compartible() && InitParams()))
            return false;

        Shape dstShape = src[0]->Shape();
        dstShape.resize(_axis + 1);
        dstShape[_axis] = _N;
        dst[0]->Reshape(_dst8u ? TensorType8u : TensorType32f, dstShape, TensorFormatNchw);

        if (!_src8u)
            Layer::Extend8u(buf, 0, src[0]->Shape(), src[0]->Format());
        Layer::Extend32i(buf, 0, dstShape, TensorFormatNchw);

        std::stringstream desc;
        desc << _batch << "x" << _M << "x" << _K << "-" << _N << " ";
        desc << ToChar(src[0]->GetType()) << (src.size() > 1 ? ToChar(src[1]->GetType()) : "0") << ToChar(dst[0]->GetType()) << "-";
        desc << (_transB ? "n" : "t") << (_biasTerm ? "b" : "o");
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    bool QuantizedInnerProductLayer::Compartible() const
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();

        if (param.qSrc().size() < 2)
            SYNET_ERROR("QuantizedInnerProductLayer must have at least 2 input dequantizers!");
        if (param.qSrc()[0].weights() != 0)
            SYNET_ERROR("QuantizedInnerProductLayer supports only uniform input quantization!");
        if (param.qSrc()[1].weights() < 2 || weight.size() < param.qSrc()[1].weights())
            SYNET_ERROR("QuantizedInnerProductLayer: check weight or dequantizers!");
        if (weight[0].GetType() != TensorType8i)
            SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
        bool weightZeroZero = true;
        if (param.qSrc()[1].weights() == 2)
        {
            if (param.qSrc()[1].type() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
            weightZeroZero = param.qSrc()[1].zero() == 0;
        }
        else
        {
            if (weight[2].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer supports only INT8 weight!");
            for (size_t i = 0, n = weight[2].Size(); i < n && weightZeroZero; ++i)
                weightZeroZero = weight[2].Data<int8_t>()[i] == 0;
        }
        if (!weightZeroZero)
            SYNET_ERROR("QuantizedInnerProductLayer supports only weight 'zero' == 0!");

        if (_biasTerm)
        {
            if (param.qSrc().size() != 3)
                SYNET_ERROR("QuantizedInnerProductLayer must have 3 input dequantizers for when uses bias!");
            int biasStart = param.qSrc()[1].weights();
            bool biasZeroZero = true;
            if (param.qSrc()[2].weights() == 2)
            {
                if (param.qSrc()[2].type() != TensorType32i)
                    SYNET_ERROR("QuantizedInnerProductLayer supports only INT32 bias!");
                biasZeroZero = param.qSrc()[2].zero() == 0;
            }
            else
            {
                if (weight[biasStart + 2].GetType() != TensorType32i)
                    SYNET_ERROR("QuantizedInnerProductLayer supports only INT32 bias!");
                for (size_t i = 0, n = weight[biasStart + 2].Size(); i < n && biasZeroZero; ++i)
                    biasZeroZero = weight[biasStart + 2].Data<int32_t>()[i] == 0;
            }
            if (!biasZeroZero)
                SYNET_ERROR("QuantizedInnerProductLayer supports only bias 'zero' == 0!");

            bool equalScale = true;
            if (weight[0].Count() != 2 || weight[0].GetType() != TensorType8i)
                SYNET_ERROR("QuantizedInnerProductLayer: weight[0] must be 2D int8 tensor!");
            if (weight[1].Count() != 1 || weight[biasStart + 1].Count() != 1 || weight[1].Axis(0) != weight[biasStart + 1].Axis(0))
                SYNET_ERROR("QuantizedInnerProductLayer: weight scale (weight[1]) must the same size as bias scale (weight[" << biasStart + 1 << "]) !");
            float srcScale = (float)param.qSrc()[0].scale();
            for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
            {
                if (::fabs(weight[1].Data<float>()[i] * srcScale - weight[biasStart + 1].Data<float>()[i]) > 0.000001)
                    SYNET_ERROR("QuantizedInnerProductLayer: weight scale (weight[1]) and bias scale (weight[" << biasStart + 1 << "]) are not compartible!");
            }
        }

        if (param.qDst().size())
        {
            if (param.qDst()[0].weights() != 0)
                SYNET_ERROR("QuantizedInnerProductLayer supports only uniform output quantization!");
        }

        return true;
    }

    bool QuantizedInnerProductLayer::InitParams()
    {
        const LayerParam& param = this->Param();
        const Tensors& weight = this->Weight();
        int srcZero = param.qSrc()[0].zero();
        _bias32i.Reshape(TensorType32i, Shp(weight[1].Size()), TensorFormatNchw, int32_t(0));
        if (weight[0].Format() == TensorFormatNhwc)
        {
            SYNET_ERROR("QuantizedInnerProductLayer: unsupported weight[0] format: " << weight[0].Format() << " !");
        }
        else
        {
            size_t M = weight[0].Size(0, 1), K = weight[0].Size(1, 2);
            const int8_t* pw = weight[0].Data<int8_t>();
            int32_t* pb = _bias32i.Data<int32_t>();
            for (size_t i = 0; i < M; ++i)
            {
                pb[i] = 0;
                for (size_t k = 0; k < K; ++k)
                    pb[i] -= pw[i * K + k] * srcZero;
            }
            if (_biasTerm)
            {
                int biasStart = param.qSrc()[1].weights();
                const int32_t* pw = weight[biasStart + 0].Data<int32_t>();
                int32_t* pb = _bias32i.Data<int32_t>();
                for (size_t i = 0; i < M; ++i)
                    pb[i] += pw[i];
            }
        }
        _norm32f.Reshape(TensorType32f, Shp(weight[1].Size()), TensorFormatNchw, float(0));
        if (param.qDst().size())
        {
            int dstZero = param.qDst()[0].zero();
            float dstScale = (float)param.qDst()[0].scale();
            _dstZero8u.Reshape(TensorType8u, Shp(weight[1].Size()), TensorFormatNchw, uint8_t(dstZero));
            if (_biasTerm)
            {
                int biasStart = param.qSrc()[1].weights();
                for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i] / dstScale;
            }
            else
            {
                float srcScale = (float)param.qSrc()[0].scale();
                for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale / dstScale;
            }
        }
        else
        {
            if (_biasTerm)
            {
                int biasStart = param.qSrc()[1].weights();
                for (size_t i = 0, n = weight[biasStart + 1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[biasStart + 1].Data<float>()[i];
            }
            else
            {
                float srcScale = (float)param.qSrc()[0].scale();
                for (size_t i = 0, n = weight[1].Size(); i < n; ++i)
                    _norm32f.Data<float>()[i] = weight[1].Data<float>()[i] * srcScale;
            }
        }
        return true;
    }

    void QuantizedInnerProductLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        uint8_t* tmp = _src8u ? src[0]->Data<uint8_t>() : Layer::Buf8u(buf, 0);
        int32_t* sum = Layer::Buf32i(buf, 0);
        //if (!_src8u)
        //    _srcCvt.Convert(src[0]->Data<float>(), tmp);
        Gemm(tmp, sum);
        if (_dst8u)
            PostProcess(sum, dst[0]->Data<uint8_t>());
        else
            PostProcess(sum, dst[0]->Data<float>());
    }

    void QuantizedInnerProductLayer::Gemm(const uint8_t* src, int32_t* sum)
    {
        const int8_t* weight = Weight()[0].Data<int8_t>();
        const bool overflow16i = true;
        Synet::CpuGemm8iNT(_M, _N, _K, src, _K, weight, _K, sum, _N, overflow16i);
    }

    void QuantizedInnerProductLayer::PostProcess(const int32_t* sum, float* dst)
    {
        const int32_t* bias = _bias32i.Data<int32_t>();
        const float* norm = _norm32f.Data<float>();
        DequantizeLinear(sum, _M, _N, 1, 1, TensorFormatNchw, bias, norm, dst);
    }

    void QuantizedInnerProductLayer::PostProcess(const int32_t* sum, uint8_t* dst)
    {
        const float* norm = _norm32f.Data<float>();
        const int32_t* bias = _bias32i.Data<int32_t>();
        const uint8_t* zero = _dstZero8u.Data<uint8_t>();
        QuantizeSumLinear(sum, _M, _N, 1, 1, TensorFormatNchw, bias, norm, zero, dst);
    }
}
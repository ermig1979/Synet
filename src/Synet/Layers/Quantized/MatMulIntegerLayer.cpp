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

#include "Synet/Layers/Quantized/MatMulIntegerLayer.h"

namespace Synet
{
    void MatMulIntegerGemm(size_t M, size_t N, size_t K, const uint8_t* src, int32_t zero, const int8_t* weight, int32_t* dst, bool overflow16i)
    {
        const size_t K2 = overflow16i ? K / 2 * 2 : 0;
        int32_t* bias = dst + (M - 1) * N;
        for (size_t j = 0; j < N; ++j)
            bias[j] = 0;
        for (size_t k = 0; k < K; ++k)
        {
            const int8_t* pw = weight + k * N;
            for (size_t j = 0; j < N; ++j)
                bias[j] -= pw[j] * zero;
        }

        for (size_t i = 0; i < M; ++i)
        {
            if (dst < bias)
            {
                for (size_t j = 0; j < N; ++j)
                    dst[j] = bias[j];
            }
            size_t k = 0;
            for (; k < K2; k += 2)
            {
                int32_t s0 = src[k + 0];
                int32_t s1 = src[k + 1];
                const int8_t* w0 = weight + (k + 0) * N;
                const int8_t* w1 = weight + (k + 1) * N;
                for (size_t j = 0; j < N; ++j)
                    dst[j] += RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
            }
            for (; k < K; k += 1)
            {
                int32_t s0 = src[k];
                const int8_t* w0 = weight + k * N;
                for (size_t j = 0; j < N; ++j)
                    dst[j] += s0 * w0[j];
            }
            src += K;
            dst += N;
        }
    }

    //-------------------------------------------------------------------------------------------------

    MatMulIntegerLayer::MatMulIntegerLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t MatMulIntegerLayer::Flop() const
    {
        if (_const)
            return 0;
        return _M * _N * (_K * 2 + 0);
    }

    bool MatMulIntegerLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("MatMulIntegerLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != TensorType8u || src[1]->GetType() != TensorType8u)
            SYNET_ERROR("MatMulIntegerLayer supports only UINT8 inputs!");

        Shape shape = src[0]->Shape();
        _K = src[0]->Size(-1);
        _M = src[0]->Size(0, -1);

        Tensors& weight = ((Tensors&)this->Weight());
        if (weight.size() != 2)
            SYNET_ERROR("MatMulIntegerLayer supports only 2 weights!");
        if (weight[0].GetType() != TensorType8i || weight[1].GetType() != TensorType8i)
            SYNET_ERROR("MatMulIntegerLayer supports only INT8 weights!");
        _N = weight[0].Axis(-1);
        if (weight[0].Axis(0) != _K)
            SYNET_ERROR("MatMulIntegerLayer: check src[0] and weight[0] size!");

        shape.back() = _N;
        dst[0]->Reshape(TensorType32i, shape, src[0]->Format());
        if (src[0]->Const() && src[1]->Const())
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            std::stringstream desc;
            desc << _M << "x" << _K << "-" << _N << " ";
            this->UsePerfStat(desc.str(), Flop());
            _const = false;
        }

        return true;
    }

    void MatMulIntegerLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        const bool overflow16i = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#else
        const bool overflow16i = true;
#endif
        MatMulIntegerGemm(_M, _N, _K, src[0]->Data<uint8_t>(), src[1]->Data<uint8_t>()[0], this->Weight()[0].Data<int8_t>(), dst[0]->Data<int32_t>(), overflow16i);
    }
}
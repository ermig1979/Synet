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

#include "Synet/Layers/Quantized/QuantizedMergedConvolutionLayer.h"
#include "Synet/Layers/BiasLayer.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Quantization/Gemm.h"
#include "Synet/Quantization/QuantizeLinear.h"
#include "Synet/Quantization/DequantizeLinear.h"

namespace Synet
{
    QuantizedMergedConvolutionLayer::QuantizedMergedConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    size_t QuantizedMergedConvolutionLayer::MemoryUsage() const
    {
        size_t size = Layer::MemoryUsage();
        for (size_t c = 0; c < _count; ++c)
            size += _bias32i[c].RawSize() + _norm32f[c].RawSize();
        size += _dwSrcZero8u.RawSize();
        size += _quantizedMergedConvolution.InternalBufferSize();
        return size;
    }

    int64_t QuantizedMergedConvolutionLayer::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _count; ++i)
            flop += _batch * _conv[i].dstC * _conv[i].dstH * _conv[i].dstW *
            (_conv[i].kernelY * _conv[i].kernelX * _conv[i].srcC / _conv[i].group * 2 + (_bias[i] ? 1 : 0) + _conv[i].ActivalionFlop());
        if (_add)
        {
            const ConvParam& back = _conv[_count - 1];
            flop += _batch * back.dstC * back.dstH * back.dstW * 7;
        }
        return flop;
    }

    void QuantizedMergedConvolutionLayer::CompactWeight()
    {
        if (_quantizedMergedConvolution.Enable())
        {
            for (size_t i = 0; i < this->Weight().size(); ++i)
                ((Tensor&)this->Weight()[i]).Clear();
        }
    }

    LowPrecisionType QuantizedMergedConvolutionLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypeActive;
        return LowPrecisionTypeNone;
    }

    bool QuantizedMergedConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("QuantizedMergedConvolutionLayer supports only 1 input and 1 output!");
        if (src[0]->Count() != 4 || src[0]->GetType() != TensorType8u || src[0]->Format() != TensorFormatNhwc)
            SYNET_ERROR("QuantizedMergedConvolutionLayer supports only 4D UINT8 NHWC tensors!");

        const LayerParam& p = this->Param();
        const MergedConvolutionParam& mc = p.mergedConvolution();
        const ConvolutionParam* conv = mc.conv().data();
        const Tensors& weight = this->Weight();

        _batch = src[0]->Axis(0);
        _count = mc.conv().size();
        if (_count < 2 && _count > 3)
            SYNET_ERROR("QuantizedMergedConvolutionLayer supports only 2 or 3 merged convolutions!");
        _add = _count == 3 && mc.add() ? 1 : 0;

        size_t qn = _add;
        for (size_t i = 0; i < _count; ++i)
            qn += conv[i].biasTerm() ? 3 : 2;
        if(qn != p.qSrc().size())
            SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong number of input quantization parameters!");

        size_t wn = 0;
        for (size_t i = 0; i < p.qSrc().size(); ++i)
            wn += p.qSrc()[i].weights();
        if (wn != weight.size())
            SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong number of weights!");

        if (p.qDst().size() != 1 || p.qDst()[0].weights() != 0)
            SYNET_ERROR("QuantizedMergedConvolutionLayer supports only uniform output quantization!");

        for (size_t c = 0, nextW = 0, nextQ = 0; c < _count; ++c)
        {
            _indexQ[c] = nextQ;
            _indexW[c] = nextW;

            if (!_conv[c].Set(conv[c]))
                SYNET_ERROR("QuantizedMergedConvolutionLayer: check " << c << " convolution parameter!");
            if (c)
                _conv[c].Set(_conv[c - 1], true, conv[c].autoPad());
            else
            {
                _conv[c].Set(*src[0], *dst[0], true, conv[c].autoPad());
                _conv[c].dstT = TensorType8u;
            }

            const QuantizeParam& qS = p.qSrc()[nextQ++];
            if (qS.weights() != 0)
                SYNET_ERROR("QuantizedMergedConvolutionLayer supports only uniform quantization (check " << nextQ - 1 << " parameter) !");
            _ioZero[c] = (uint8_t)qS.zero();
            _ioScale[c] = (float)qS.scale();

            const Tensor& w = weight[nextW];
            if (w.Shape() != _conv[c].WeightShape(true, true) || w.Format() != src[0]->Format() || w.GetType() != TensorType8i)
                SYNET_ERROR("QuantizedMergedConvolutionLayer: check weight[" << nextW << "] size, format or type!");
            size_t K = w.Size(0, 3), M = w.Size(3, 4);
            _ptrW[c] = w.Data<int8_t>();
            const QuantizeParam& qW = p.qSrc()[nextQ++];
            nextW += qW.weights();

            _bias[c] = conv[c].biasTerm();
            if (_bias[c])
            {
                const QuantizeParam& qB = p.qSrc()[nextQ++];
                if (qB.type() != TensorType32i)
                    SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong bias[" << c << "] (quantize[" << nextQ - 1 << "]) quantize parameter!");
                const Tensor& b = weight[nextW++];
                if (b.Size() != M || b.GetType() != TensorType32i || qB.type() != TensorType32i)
                    SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong bias[" << c << "] (weight[" << nextW - 1 << "]) size or type!");
                _ptrB[c] = (int32_t*)b.Data<int32_t>();
            }

            _params[c][0] = conv[c].activationParam0();
            _params[c][1] = conv[c].activationParam1();

            const Tensor& s = weight[_indexW[c] + 1];
            if (s.Size() != M || s.GetType() != TensorType32f)
                SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong scale[" << c << "] (weight[" << _indexW[c] + 1 << "]) size or type!");
            _ptrS[c] = (float*)s.Data<float>();
        }
        if (_add)
        {
            _ioZero[_count] = (uint8_t)p.qSrc().back().zero();
            _ioZero[_count + 1] = (uint8_t)p.qDst()[0].zero();
            _ioScale[_count] = (float)p.qSrc().back().scale();
            _ioScale[_count + 1] = (float)p.qDst()[0].scale();
        }
        else
        {
            _ioZero[_count] = (uint8_t)p.qDst()[0].zero();
            _ioScale[_count] = (float)p.qDst()[0].scale();
        }

        const ConvParam& back = _conv[_count - 1];
        dst[0]->Reshape(TensorType8u, back.DstShape(_batch), src[0]->Format());

        _srcS = src[0]->Size(1);
        _dstS = dst[0]->Size(1);

        if (_add)
        {
            if (_srcS != _dstS)
                SYNET_ERROR("QuantizedMergedConvolutionLayer with add=1 parameter must have input and output of the same size!");
            _srcBias = -p.qSrc()[0].zero();
            _srcNorm = float(p.qSrc()[0].scale());
            _dstBias = -p.qSrc().back().zero();
            _dstNorm = float(p.qSrc().back().scale());
            _addZero = p.qDst()[0].zero();
            _addScale = 1.0f / float(p.qDst()[0].scale());
        }

        _quantizedMergedConvolution.Init(_batch, _conv, _count, _add);
        if (_quantizedMergedConvolution.Enable())
        {
            Layer::Extend8u(buf, 0, Shp(_quantizedMergedConvolution.ExternalBufferSize()));
            _quantizedMergedConvolution.SetParams(_ioScale, _ioZero, _ptrW, _ptrS, _ptrB);
        }
        else
        {
            for (size_t c = 0; c < _count; ++c)
            {
                size_t iQ = _indexQ[c], iW = _indexW[c];

                const QuantizeParam& qS = p.qSrc()[iQ + 0];
                const QuantizeParam& qD = c + 1 < _count ? p.qSrc()[_indexQ[c + 1]] : (_add ? p.qSrc().back() : p.qDst()[0]);
                int srcZero = qS.zero(), dstZero = qD.zero();
                float srcScale = (float)qS.scale(), dstScale = float(qD.scale());

                const Tensor& w = weight[iW];
                size_t K = w.Size(0, 3), M = w.Size(3, 4);

                _bias32i[c].Reshape(TensorType32i, Shp(M), TensorFormatNchw, int32_t(0));
                if(_conv[c].IsDepthwise())
                    _dwSrcZero8u.Reshape(TensorType8u, Shp(_conv[c].srcC), TensorFormatNchw, _ioZero[c]);
                const int8_t* pw = w.Data<int8_t>();
                int32_t* pb = _bias32i[c].Data<int32_t>();
                for (size_t i = 0; i < M; ++i)
                {
                    pb[i] = 0;
                    for (size_t k = 0; k < K; ++k)
                        pb[i] -= pw[k * M + i] * srcZero;
                }
                if (_bias[c])
                {
                    const QuantizeParam& qW = p.qSrc()[iQ + 1];
                    const Tensor& b = weight[iW + qW.weights()];
                    const int32_t* pw = b.Data<int32_t>();
                    for (size_t i = 0; i < M; ++i)
                        pb[i] += pw[i];
                }

                _norm32f[c].Reshape(TensorType32f, Shp(M), TensorFormatNchw, float(0));
                const float* ps = weight[iW + 1].Data<float>();
                float* pn = _norm32f[c].Data<float>();
                for (size_t i = 0; i < M; ++i)
                    pn[i] = ps[i] * srcScale / dstScale;

                Layer::Extend32i(buf, 0, _conv[c].DstShape(1));
                if(c + 1 < _count)
                    Layer::Extend8u(buf, 0, _conv[c].DstShape(1));
            }
        }

        dst[0]->Reshape(TensorType8u, back.DstShape(_batch), src[0]->Format());

        std::stringstream desc;
        desc << _count << ": " << _batch << "x" << _conv[0].srcC << "x" << _conv[0].srcH << "x" << _conv[0].srcW;
        for (size_t i = 0; i < _count; ++i)
            desc << "-" << (_conv[i].IsDepthwise() ? String("") : Cpl::ToStr(_conv[i].dstC) + "x") << _conv[i].kernelY << "x" << _conv[i].strideY;
        if(_quantizedMergedConvolution.Enable())
            desc << " " << _quantizedMergedConvolution.Info();
        this->UsePerfStat(desc.str(), Flop()); 

        return true;
    }

    void QuantizedMergedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_quantizedMergedConvolution.Enable())
            _quantizedMergedConvolution.Forward(src[0]->RawData(), Layer::Buf8u(buf, 0), dst[0]->RawData());
        else
            ForwardCpu(src[0]->RawData(), Layer::Buf8u(buf, 0), Layer::Buf32i(buf, 0), dst[0]->RawData());
    }

    void QuantizedMergedConvolutionLayer::ForwardCpu(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
    {
        const bool overflow16i = true;
        for (size_t b = 0; b < _batch; ++b)
        {
            for (size_t c = 0; c < _count; ++c)
            {
                const ConvParam& conv = _conv[c];
                const uint8_t* ps = c ? buf : src;
                uint8_t* pd = c + 1 < _count ? buf : dst;
                if (conv.IsDepthwise())
                    DepthwiseConvolution(ps, _dwSrcZero8u.RawData(), conv, _ptrW[c], sum);
                else
                    Synet::CpuGemm8iNN(conv.dstH * conv.dstW, conv.dstC, conv.kernelY * conv.kernelX, conv.srcC, ps, conv.srcC * conv.kernelY * conv.kernelX, _ptrW[c], conv.dstC, sum, conv.dstC, overflow16i);
                QuantizeSumLinear(sum, 1, conv.dstC, conv.dstH, conv.dstW, conv.dstF, _bias32i[c].Data<int32_t>(), _norm32f[c].Data<float>(), _ioZero[c + 1], pd);
            }
            if (_add)
                AddSrc(src, dst);
            src += _srcS;
            dst += _dstS;
        }
    }

    void QuantizedMergedConvolutionLayer::DepthwiseConvolution(const uint8_t* src, const uint8_t* zero, const ConvParam& conv, const int8_t* weight, int32_t* dst)
    {
        size_t C = conv.srcC;
        for (size_t dy = 0; dy < conv.dstH; ++dy)
        {
            for (size_t dx = 0; dx < conv.dstW; ++dx)
            {
                for (size_t c = 0; c < C; ++c)
                    dst[c] = 0;
                for (size_t ky = 0; ky < conv.kernelY; ++ky)
                {
                    size_t sy = dy * conv.strideY + ky - conv.padY;
                    for (size_t kx = 0; kx < conv.kernelX; ++kx)
                    {                    
                        size_t sx = dx * conv.strideX + kx - conv.padX;
                        const int8_t* pw = weight + (ky * conv.kernelX + kx) * C;
                        if (sy < conv.srcH && sx < conv.srcW)
                        {
                            const uint8_t* ps = src + (sy * conv.srcW + sx) * C;
                            for (size_t c = 0; c < C; ++c)
                                dst[c] += ps[c] * pw[c];
                        }
                        else
                        {
                            for (size_t c = 0; c < C; ++c)
                                dst[c] += zero[c] * pw[c];
                        }
                    }
                }
                dst += C;
            }
        }
    }

    void QuantizedMergedConvolutionLayer::AddSrc(const uint8_t* src, uint8_t* dst)
    {
        for (size_t i = 0; i < _dstS; ++i)
        {
            float _src = DequantizeLinear(src[i], _srcBias, _srcNorm);
            float _dst = DequantizeLinear(dst[i], _dstBias, _dstNorm);
            dst[i] = QuantizeLinear(_src + _dst, _addScale, _addZero, 0, 255);
        }
    }
}
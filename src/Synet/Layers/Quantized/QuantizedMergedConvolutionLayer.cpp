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
        return Layer::MemoryUsage();
    }

    int64_t QuantizedMergedConvolutionLayer::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _count; ++i)
            flop += _batch * _conv[i].dstC * _conv[i].dstH * _conv[i].dstW *
            (_conv[i].kernelY * _conv[i].kernelX * _conv[i].srcC / _conv[i].group * 2 + (_bias[i] ? 1 : 0) + _conv[i].ActivalionFlop());
        return flop;
    }

    void QuantizedMergedConvolutionLayer::CompactWeight()
    {
        //if (_quantizedMergedConvolution.Enable())
        //  for(size_t c = 0; c < _count; ++c)
        //      ((Tensor&)this->Weight()[_indexW[c]]).Clear();
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

        size_t qn = 0;
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
                _conv[c].Set(*src[0], *dst[0], true, conv[c].autoPad());

            const QuantizeParam& qS = p.qSrc()[nextQ++];
            if (qS.weights() != 0)
                SYNET_ERROR("QuantizedMergedConvolutionLayer supports only uniform quantization (check " << nextQ - 1 << " parameter) !");

            const Tensor& w = weight[nextW];
            if (w.Shape() != _conv[c].WeightShape(true, true) || w.Format() != src[0]->Format() || w.GetType() != TensorType8i)
                SYNET_ERROR("QuantizedMergedConvolutionLayer: check weight[" << nextW << "] size, format or type!");
            size_t K = w.Size(0, 3), M = w.Size(3, 4);
            _weight[c] = w.Data<int8_t>();
            const QuantizeParam& qW = p.qSrc()[nextQ++];
            nextW += qW.weights();

            _bias[c] = conv[c].biasTerm();
            if (_bias[c])
            {
                const Tensor& b = weight[nextW++];
                if (b.Size() != M || b.GetType() != TensorType32i)
                    SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong bias[" << c << "] (weight[" << nextW - 1 << "]) size or type!");
            }

            _params[c][0] = conv[c].activationParam0();
            _params[c][1] = conv[c].activationParam1();

            const Tensor& s = weight[_indexW[c] + 1];
            if (s.Size() != M || s.GetType() != TensorType32f)
                SYNET_ERROR("QuantizedMergedConvolutionLayer has wrong scale[" << c << "] (weight[" << _indexW[c] + 1 << "]) size or type!");
        }

        const ConvParam& back = _conv[_count - 1];
        dst[0]->Reshape(TensorType8u, back.DstShape(_batch), src[0]->Format());

        _srcS = src[0]->Size(1);
        _dstS = dst[0]->Size(1);

        _add = _count == 3 && mc.add() ? 1 : 0;
        if (_add)
        {
            if (_srcS != _dstS)
                SYNET_ERROR("QuantizedMergedConvolutionLayer with add=1 parameter must have input and output of the same size!");
        }

        {
            for (size_t c = 0; c < _count; ++c)
            {
                size_t iQ = _indexQ[c], iW = _indexW[c];

                const QuantizeParam& qS = p.qSrc()[iQ + 0];
                const QuantizeParam& qD = c + 1 < _count ? p.qSrc()[_indexQ[c + 1]] : p.qDst()[0];
                int srcZero = qS.zero(), dstZero = qD.zero();
                float srcScale = (float)qS.scale(), dstScale = float(qD.scale());

                const Tensor& w = weight[iW];
                size_t K = w.Size(0, 3), M = w.Size(3, 4);

                _bias32i[c].Reshape(TensorType32i, Shp(M), TensorFormatNchw, int32_t(0));
                _srcZero8u[c].Reshape(TensorType8u, Shp(_conv[c].srcC), TensorFormatNchw, uint8_t(srcZero));
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


        //if (_alg.trans)
        //{
        //    _alg.siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
        //    _alg.ldW = _conv.dstC;
        //    _alg.grW = _conv.dstC / _conv.group;

        //    _alg.siS = _conv.dstH * _conv.dstW;
        //    _alg.ldS = _alg.siW * (_alg.is1x1 ? _conv.group : 1);
        //    _alg.grS = _alg.siW * (_alg.is1x1 ? 1 : _alg.siS);

        //    _alg.siD = _conv.dstC / _conv.group;
        //    _alg.ldD = _conv.dstC;
        //    _alg.grD = _alg.siD;
        //}

        //dst[0]->Reshape(_dst8u ? TensorType8u: TensorType32f, _conv.DstShape(_alg.batch), src[0]->Format());

        //_alg.sSize = src[0]->Size(1);
        //_alg.dSize = dst[0]->Size(1);

        //_quantizedMergedConvolution.Init(_alg.batch, &_conv);
        //if (_quantizedMergedConvolution.Enable())
        //{
        //    Layer::Extend8u(buf, 0, Shp(_quantizedMergedConvolution.ExternalBufferSize()));
        //    const Tensors& weight = this->Weight();
        //    int bias = param.qSrc()[1].weights();
        //    const float* params = _conv.activation == ActivationFunctionTypePrelu ? weight.back().Data<float>() : _alg.params;
        //    float srcScale = (float)param.qSrc()[0].scale(), dstScale = (float)param.qDst()[0].scale();
        //    uint8_t srcZero = (uint8_t)param.qSrc()[0].zero(), dstZero = (uint8_t)param.qDst()[0].zero();
        //    _quantizedConvolution.SetParams(&srcScale, &srcZero, weight[0].Data<int8_t>(), weight[1].Data<float>(), 
        //        _alg.bias ? weight[bias + 0].Data<int32_t>() : NULL, params, &dstScale, &dstZero);
        //}
        //else
        {
            //if (!_src8u)
            //    Layer::Extend8u(buf, 0, _conv.SrcShape(1));
            //if (!_conv.Is1x1())
            //    Layer::Extend8u(buf, 1, Shp(_conv.ImgSize()));
            //Layer::Extend32i(buf, 0, _conv.DstShape(1));
            //if (_dst8u)
            //    Layer::Extend32f(buf, 0, _conv.DstShape(1));
        }

        std::stringstream desc;
        desc << _count << ": " << _batch << "x" << _conv[0].srcC << "x" << _conv[0].srcH << "x" << _conv[0].srcW;
        for (size_t i = 0; i < _count; ++i)
            desc << "-" << (_conv[i].IsDepthwise() ? String("") : Cpl::ToStr(_conv[i].dstC) + "x") << _conv[i].kernelY << "x" << _conv[i].strideY;

        //if(_quantizedMergedConvolution.Enable())
        //    desc << " " << _quantizedMergedConvolution.Info();
        this->UsePerfStat(desc.str(), Flop()); 

        return true;
    }

    void QuantizedMergedConvolutionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {

    }
}
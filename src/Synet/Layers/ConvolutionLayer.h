/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
#include "Synet/Utils/Winograd.h"
#include "Synet/Utils/Convolution.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Layers/PreluLayer.h"
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Layers/HswishLayer.h"

namespace Synet
{
    template <class T> class ConvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        ConvolutionLayer(const LayerParam & param)
            : Base(param)
        {
            const ConvolutionParam & p = this->Param().convolution();
            _is8i = p.quantizationLevel() == TensorType8i;
            _src8u = false;
            _dst8u = false;
            _internal = 0;
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _convolution32f.InternalBufferSize() * sizeof(Type);
        }

        virtual int64_t Flop() const
        {
            return _num* _conv.kernelY* _conv.kernelX* _conv.srcC* _conv.dstH* _conv.dstW* _conv.dstC / _conv.group * 2;
        }

        virtual void CompactWeight()
        {
            if (_internal)
                ((Tensor&)this->Weight()[0]).Clear();
        }

        virtual bool Can8i() const
        {
            return _is8i;
        }

        virtual bool Is8i() const
        {
            return _is8i;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1);

            const ConvolutionParam & param = this->Param().convolution();
            const Tensors & weight = this->Weight();

            _conv.Set(param);
            _conv.Set(*src[0], *dst[0], true);

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
            assert(weight[0].Shape() == _conv.WeightShape(_trans != 0, true) && weight[0].Format() == src[0]->Format());

            Shape dstShape(src[0]->Shape().begin(), src[0]->Shape().begin() + _axis);
            if (_trans)
            {
                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);
                dstShape.push_back(_conv.dstC);

                _siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
                _ldW = _conv.dstC;
                _grW = _conv.dstC / _conv.group;

                _siS = _conv.dstH * _conv.dstW;
                _ldS = _siW;
                _grS = _siS * _siW;

                _siD = _conv.dstC / _conv.group;
                _ldD = _conv.dstC;
                _grD = _siD;
            }
            else
            {
                dstShape.push_back(_conv.dstC);
                dstShape.push_back(_conv.dstH);
                dstShape.push_back(_conv.dstW);

                _siW = _conv.srcC * _conv.kernelY * _conv.kernelX / _conv.group;
                _ldW = _siW;
                _grW = _conv.dstC * _siW / _conv.group;

                _siS = _conv.dstH * _conv.dstW;
                _ldS = _siS;
                _grS = _siS * _siW;

                _siD = _conv.dstC / _conv.group;
                _ldD = _conv.dstH * _conv.dstW;
                _grD = _siD * _siS;
            }

            if (_is8i)
            {
                _src8u = src[0]->GetType() == TensorType8u;
                _dst8u = dst[0]->GetType() == TensorType8u;
                if(_dst8u)
                    dst[0]->As8u().Reshape(dstShape, src[0]->Format());
                else
                    dst[0]->As32f().Reshape(dstShape, src[0]->Format());
                _convolution8i.Init(_num, &_conv);
                if (_convolution8i.Enable())
                {
                    buf[TensorType8u * BUFFER_COUNT]->As8u().Extend({ _convolution8i.ExternalBufferSize() });
                    const float* bias = _biasTerm ? weight[1].CpuData() : NULL;
                    const float* params = _conv.activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : _params;
                    const float* stats[4] = {
                        this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                        this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                        this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                        this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data() };
                    _convolution8i.SetParams(weight[0].CpuData(), bias, params, stats);
                }
                else
                {
                    if (!_src8u)
                        buf[TensorType8u * BUFFER_COUNT + 1]->As8u().Extend(src[0]->Shape());
                    buf[TensorType8u * BUFFER_COUNT]->As8u().Extend(Shape({ _conv.kernelY * _conv.kernelX * _conv.srcC * _conv.dstH * _conv.dstW }));
                    buf[TensorType32i * BUFFER_COUNT]->As32i().Extend(dstShape, src[0]->Format());
                    Init8i();
                }
            }
            else
            {
                dst[0]->Reshape(dstShape, src[0]->Format());
                _convolution32f.Init(_num, &_conv, SYNET_EXTERNAL_GEMM);
                if (_convolution32f.Enable())
                {
                    buf[TensorType32f*BUFFER_COUNT]->Extend({ _convolution32f.ExternalBufferSize() });
                    _convolution32f.SetParams(weight[0].CpuData(), &_internal, _biasTerm ? weight[1].CpuData() : NULL,
                        _conv.activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : _params);
                }
                else
                    buf[TensorType32f*BUFFER_COUNT]->Extend(Shape({ _conv.kernelY * _conv.kernelX * _conv.srcC, _conv.dstH * _conv.dstW }));
            }
            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
            std::stringstream desc;
            desc << "i=" << _num << "x" << _conv.srcC << "x" << _conv.srcH << "x" << _conv.srcW << " o=" << _conv.dstC;
            desc << " k=" << _conv.kernelY << " s=" << _conv.strideY << " g=" << _conv.group << " " << (Is8i() ? "int8" : "fp32");
            this->UsePerfStat(desc.str(), Flop());
        }

        virtual void DebugPrint(std::ostream& os, int flag, int first, int last, int precision)
        {
            const Stat& statS = *this->Stats(0)[0];
            Synet::DebugPrint(os, statS.scale8uTo32f, "pSrcScaleInv", first, last, precision);
            Synet::DebugPrint(os, statS.scale32fTo8u, "pSrcScale", first, last, precision);
            Synet::DebugPrint(os, statS.shift32fTo8u, "pSrcShift", first, last, precision);
            const Stat& statD = *this->Stats(2)[0];
            Synet::DebugPrint(os, statD.scale8uTo32f, "pDstScale", first, last, precision);
            Synet::DebugPrint(os, statD.shift8uTo32f, "pDstShift", first, last, precision);
            _weight8i.DebugPrint(os, "_weight8i", true, first, last, precision);
            _norm32i.DebugPrint(os, "_norm32i", false, first, last, precision);
            Synet::DebugPrint(os, _dstCvt.scale, _dstCvt.channels, "_dstCvt.scale", first, last, precision);
            Synet::DebugPrint(os, _dstCvt.shift, _dstCvt.channels, "_dstCvt.shift", first, last, precision);

            Synet::DebugPrint(_weight8i.CpuData(), Shape({ _weight8i.Size() }), "sy_weight8i");
            Synet::DebugPrint(_norm32i.CpuData(Shape{ 1u, 0u }), Shape({ _norm32i.Size()/2 }), "sy_norm32i");
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_is8i)
            {
                if (_convolution8i.Enable())
                    _convolution8i.Forward(src[0]->RawCpuData(), buf[TensorType8u * BUFFER_COUNT]->RawCpuData(), dst[0]->RawCpuData());
                else
                {
                    uint8_t* buf0 = buf[TensorType8u * BUFFER_COUNT]->As8u().CpuData();
                    uint8_t* tmp = _src8u ? src[0]->As8u().CpuData() : buf[TensorType8u * BUFFER_COUNT + 1]->As8u().CpuData();
                    int32_t* sum = buf[TensorType32i * BUFFER_COUNT]->As32i().CpuData();
                    if (!_src8u)
                        Convert32fTo8u(src[0]->As32f().CpuData(), _srcCvt, tmp);
                    ForwardCpu8i(tmp, buf0, sum);
                    if (_dst8u)
                        Convert32iTo8u(sum, _dstCvt, dst[0]->As8u().CpuData());
                    else
                        Convert32iTo32f(sum, _dstCvt, dst[0]->As32f().CpuData());
                }
            }
            else
                ForwardCpu(src[0]->CpuData(), buf[TensorType32f*BUFFER_COUNT]->CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
            if (_convolution32f.Enable())
                _convolution32f.Forward(src, buf, dst);
            else
            {
                const Type * weight = this->Weight()[0].CpuData();
                for (size_t n = 0; n < _num; ++n)
                {
                    const Type * tmp = src;
                    if (!_is1x1)
                    {
                        if (_trans)
                            Synet::ImgToRow(tmp, _conv.srcH, _conv.srcW, _conv.srcC, _conv.kernelY, _conv.kernelX, 
                                _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, _conv.group, (const Type*)NULL, buf);
                        else
                            Synet::ImgToCol(tmp, _conv.srcC, _conv.srcH, _conv.srcW, _conv.kernelY, _conv.kernelX, 
                                _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, (const Type*)NULL, buf);
                        tmp = buf;
                    }
                    if (_trans)
                    {
                        assert(_conv.group == 1 || _conv.group == _conv.srcC);
                        for (size_t g = 0; g < _conv.group; ++g)
                            CpuGemm(CblasNoTrans, CblasNoTrans, _siS, _siD, _siW, Type(1), tmp + _grS * g, _ldS, weight + _grW * g, _ldW, Type(0), dst + _grD * g, _ldD);
                    }
                    else
                    {
                        for (size_t g = 0; g < _conv.group; ++g)
                            CpuGemm(CblasNoTrans, CblasNoTrans, _siD, _siS, _siW, Type(1), weight + _grW * g, _ldW, tmp + _grS * g, _ldS, Type(0), dst + _grD * g, _ldD);
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
                        Detail::HswishLayerForwardCpu(dst, _dstSize, _params[0], _params[1], dst);
                        break;
                    default:
                        assert(0);
                    }
                    src += _srcSize;
                    dst += _dstSize;
                }
            }
        }

        void Init8i()
        {
            Stat & statS = *this->Stats(0)[0];
            Stat & statD = *this->Stats(2)[0];
            statS.Init8u();
            statD.Init8u();
            _negSrc = statS.negative;
            _weight8i.Reshape(this->Weight()[0].Shape(), _trans ? TensorFormatNhwc : TensorFormatNchw);
            _norm32i.Reshape(Shape({ size_t(2), _conv.dstC }));
            _norm32f.Reshape(Shape({ size_t(2), _conv.dstC }));
            if (!_src8u)
            {
                _srcCvt.batch = _num;
                _srcCvt.channels = _conv.srcC;
                _srcCvt.spatial = _conv.srcH * _conv.srcW;
                _srcCvt.format = (TensorFormat)_trans;
                _srcCvt.scale = statS.scale32fTo8u.data();
                _srcCvt.shift = statS.shift32fTo8u.data();
            }
            size_t G = _conv.group, D = _conv.dstC / G, C = _conv.srcC / G, K = _conv.kernelY*_conv.kernelX, CK = C * K, GD = G*D;
            Floats normW(CK);
            const float * pSrcW = this->Weight()[0].CpuData();
            const float * pSrcB = _biasTerm ? this->Weight()[1].CpuData() : NULL;
            const float * pSrcScaleInv = statS.scale8uTo32f.data();
            const float * pSrcScale = statS.scale32fTo8u.data();
            const float * pSrcShift = statS.shift32fTo8u.data();
            const float * pDstScale = statD.scale8uTo32f.data();
            const float * pDstScaleInv = statD.scale32fTo8u.data();
            const float * pDstShift = statD.shift8uTo32f.data();
            float * pNormW = normW.data();
            int8_t * pDstW = _weight8i.CpuData();
            int32_t * pDstS = _norm32i.CpuData();
            int32_t * pDstB = pDstS + _conv.dstC;
            float * pNormScale = _norm32f.CpuData();
            float * pNormShift = pNormScale + _conv.dstC;
            _dstCvt.batch = _num;
            _dstCvt.channels = _conv.dstC;
            _dstCvt.spatial = _conv.dstH * _conv.dstW;
            _dstCvt.format = (TensorFormat)_trans;
            _dstCvt.scale = pNormScale;
            _dstCvt.shift = pNormShift;
            for (size_t g = 0; g < G; ++g)
            {
                for (size_t d = 0; d < D; ++d)
                {
                    float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
                    if (_trans)
                    {
                        for (size_t k = 0, kc = 0; k < K; ++k)
                            for (size_t c = 0; c < C; ++c, ++kc)
                            {
#ifdef SYNET_INT8_INPUT_ROUND_BUGFIX
                                pNormW[kc] = pSrcW[kc * GD + d] * pSrcScaleInv[c];
#else
                                pNormW[kc] = pSrcW[kc * GD + d] / pSrcScale[c];
#endif
                                minW = std::min(minW, pNormW[kc]);
                                maxW = std::max(maxW, pNormW[kc]);
                            }
                        scale = 127.0f / std::max(abs(maxW), abs(minW));
                        for (size_t k = 0, kc = 0; k < K; ++k)
                            for (size_t c = 0; c < C; ++c, ++kc)
                                if (_negSrc)
                                {
#ifdef SYNET_INT8_INT8_DISABLE
                                    int w = Detail::Convert32fTo8i(pNormW[kc], scale, 0.0f);
                                    if (w & 1)
                                        w = Round(w*0.25f) * 4;
                                    pDstW[kc*GD + d] = w / 2;
                                    normB -= w * pSrcShift[c];
#else
                                    pDstW[kc*GD + d] = Detail::Convert32fTo8i(pNormW[kc], scale, 0.0f);
#endif
                                }
                                else
                                {
                                    pDstW[kc*GD + d] = Detail::Convert32fTo8i(pNormW[kc], scale, 0.0f);
                                    normB -= pDstW[kc*GD + d] * pSrcShift[c];
                                }
                    }
                    else
                    {
                        for (size_t c = 0, ck = 0; c < C; ++c)
                            for (size_t k = 0; k < K; ++k, ++ck)
                            {
#ifdef SYNET_INT8_INPUT_ROUND_BUGFIX
                                pNormW[ck] = pSrcW[d * CK + ck] * pSrcScaleInv[c];
#else
                                pNormW[ck] = pSrcW[d * CK + ck] / pSrcScale[c];
#endif
                                minW = std::min(minW, pNormW[ck]);
                                maxW = std::max(maxW, pNormW[ck]);
                            }
                        scale = 127.0f / std::max(abs(maxW), abs(minW));
                        for (size_t c = 0, ck = 0; c < C; ++c)
                            for (size_t k = 0; k < K; ++k, ++ck)
                                if (_negSrc)
                                {
#ifdef SYNET_INT8_INT8_DISABLE
                                    int w = Detail::Convert32fTo8i(pNormW[ck], scale, 0.0f);
                                    if (w & 1)
                                        w = Round(w*0.25f) * 4;
                                    pDstW[d*CK + ck] = w / 2;
                                    normB -= w * pSrcShift[c];
#else
                                    pDstW[d*CK + ck] = Detail::Convert32fTo8i(pNormW[ck], scale, 0.0f);
#endif
                                }
                                else
                                {
                                    pDstW[d*CK + ck] = Detail::Convert32fTo8i(pNormW[ck], scale, 0.0f);
                                    normB -= pDstW[d*CK + ck] * pSrcShift[c];
                                }
                    }
#ifdef SYNET_INT8_INT8_DISABLE
                    pDstS[d] = _negSrc ? 2 : 1;
#else
                    pDstS[d] = 1;
#endif
                    if (pSrcB)
                        normB += pSrcB[d] * scale;
                    pDstB[d] = Synet::Quantize(normB);
                    if (_dst8u)
                    {
                        pNormScale[d] = (1.0f / scale) * pDstScaleInv[d];
                        pNormShift[d] = -pDstShift[d] / pDstScale[d];
                    }
                    else
                    {
                        pNormScale[d] = 1.0f / scale;
                        pNormShift[d] = 0;
                    }
                    //if (g * D + d == 3)
                    //    std::cout << std::fixed << std::setprecision(10) << " Synet : " << scale << " , " << pDstScaleInv[d] << " , " << pDstScale[d] << std::endl;
                }
                if (_trans)
                {
                    pSrcW += D;
                    pDstW += D;
                }
                else
                {
                    pSrcW += CK*D;
                    pDstW += CK*D;
                }
                pSrcB += D;
                pDstB += D;
                pDstS += D;
                pSrcScale += C;
                pSrcScaleInv += C;
                pSrcShift += C;
                pDstScale += D;
                pDstScaleInv += D;
                pDstShift += D;
                pNormScale += D;
                pNormShift += D;           
            }
        }

        void ForwardCpu8i(const uint8_t * src, uint8_t * buf, int32_t * dst)
        {
            const uint8_t * zero = this->Stats(0)[0]->zero8u.data();
            const int8_t * weight = _weight8i.CpuData();
            const int32_t * scale = _norm32i.CpuData();
            const int32_t * shift = scale + _conv.dstC;
            for (size_t n = 0; n < _num; ++n)
            {
                const uint8_t * tmp = src;
                if (!_is1x1)
                {
                    if (_trans)
                        Synet::ImgToRow(tmp, _conv.srcH, _conv.srcW, _conv.srcC, _conv.kernelY, _conv.kernelX,
                            _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, _conv.group, zero, buf);
                    else
                        Synet::ImgToCol(tmp, _conv.srcC, _conv.srcH, _conv.srcW, _conv.kernelY, _conv.kernelX,
                            _conv.padY, _conv.padX, _conv.padH, _conv.padW, _conv.strideY, _conv.strideX, _conv.dilationY, _conv.dilationX, zero, buf);
                    tmp = buf;
                }
                if (_trans)
                {
                    assert(_conv.group == 1 || _conv.group == _conv.srcC);
                    if(_conv.group == 1)
                        Synet::CpuGemm8iNN(_siS, _siD, _conv.kernelY*_conv.kernelX, _conv.srcC, tmp, _ldS, weight, _ldW, dst, _ldD, _negSrc);
                    else
                        for (size_t g = 0; g < _conv.group; ++g)
                            Synet::CpuGemmNN(_siS, _siD, _siW, tmp + _grS * g, _ldS, weight + _grW * g, _ldW, dst + _grD * g, _ldD);
                }
                else
                {
                    if (_conv.group == 1)
                        Synet::CpuGemm8iNN(_siD, _siS, _conv.srcC, _conv.kernelY*_conv.kernelX, weight, _ldW, tmp, _ldS, dst, _ldD, _negSrc);
                    else
                        for (size_t g = 0; g < _conv.group; ++g)
                            Synet::CpuGemmNN(_siD, _siS, _siW, weight + _grW * g, _ldW, tmp + _grS * g, _ldS, dst + _grD * g, _ldD);
                }
                Detail::ScaleLayerForwardCpu(dst, scale, shift, _conv.dstC, _conv.dstH, _conv.dstW, dst, _trans, 1);

                switch (_conv.activation)
                {
                case ActivationFunctionTypeIdentity:
                    break;
                case ActivationFunctionTypeRelu:
                    CpuRelu(dst, _dstSize, 0, dst);
                    break;
                //case ActivationFunctionTypeLeakyRelu:
                //    CpuRelu(dst, _dstSize, _params[0], dst);
                //    break;
                //case ActivationFunctionTypeRestrictRange:
                //    CpuRestrictRange(dst, _dstSize, _params[0], _params[1], dst);
                //    break;
                //case ActivationFunctionTypePrelu:
                //    Detail::PreluLayerForwardCpu(dst, this->Weight().back().CpuData(), _conv.dstC, _conv.dstH * _conv.dstW, dst, _trans);
                //    break;
                default:
                    assert(0);
                }
                src += _srcSize;
                dst += _dstSize;
            }

        }

    private:
        bool _is1x1, _biasTerm, _is8i, _src8u, _dst8u, _negSrc;
        ConvertParam _srcCvt, _dstCvt;
        int _trans, _internal;
        ConvParam _conv;
        size_t _axis, _num, _srcSize, _dstSize, _ldW, _ldS, _ldD, _grW, _grS, _grD, _siW, _siS, _siD;
        float _params[2];

        Convolution32f _convolution32f;
        Convolution8i _convolution8i;

        Tensor8i _weight8i;
        Tensor32i _norm32i;
        Tensor32f _norm32f;
    };
}
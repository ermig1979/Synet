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

#include "Synet/Layers/MergedConvolution32fLayer.h"

namespace Synet
{
    namespace Detail
    {
        template<class T, int update> struct Update
        {
            static void Func(T * ptr, T val);
        };

        template<class T> struct Update<T, 0>
        {
            static SYNET_INLINE void Func(T * ptr, T val)
            {
                *ptr = val;
            }
        };

        template<class T> struct Update<T, 1>
        {
            static SYNET_INLINE void Func(T * ptr, T val)
            {
                *ptr += val;
            }
        };

        template<class T, ActivationFunctionType activation, int update> void MergedConvolutionLayerDirect(
            const T * src, const ConvParam & conv, const T * weight, const T * bias, const T * params, T * dst)
        {
            Tensor<T> buffer({ conv.dstC });
            T * buf = buffer.CpuData();
            for (size_t dy = 0; dy < conv.dstH; ++dy)
            {
                for (size_t dx = 0; dx < conv.dstW; ++dx)
                {
                    if (bias)
                        memcpy(buf, bias, conv.dstC * sizeof(T));
                    else
                        memset(buf, 0, conv.dstC * sizeof(T));
                    for (size_t ky = 0; ky < conv.kernelY; ++ky)
                    {
                        size_t sy = dy * conv.strideY + ky - conv.padY;
                        if (sy < conv.srcH)
                        {
                            for (size_t kx = 0; kx < conv.kernelX; ++kx)
                            {
                                size_t sx = dx * conv.strideX + kx - conv.padX;
                                if (sx < conv.srcW)
                                {
                                    const float * pw = weight + (ky*conv.kernelX + kx)*conv.srcC*conv.dstC;
                                    const float * ps = src + (sy*conv.srcW + sx)*conv.srcC;
                                    for (size_t sc = 0; sc < conv.srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < conv.dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += conv.dstC;
                                    }
                                }
                            }
                        }
                    }
                    for (size_t dc = 0; dc < conv.dstC; ++dc)
                        Update<T, update>::Func(dst + dc, Activation<T, activation>::Func(buf[dc], params, dc));
                    dst += conv.dstC;
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    MergedConvolution32fLayer::MergedConvolution32fLayer(const LayerParam & param, Context* context)
        : MergedConvolutionLayer(param, context)
    {
    }

    size_t MergedConvolution32fLayer::MemoryUsage() const
    {
        return Base::MemoryUsage() + _mergedConvolution32f.InternalBufferSize() * sizeof(float);
    }

    String MergedConvolution32fLayer::InternalInfo() const
    {
        return String(" fp32") + (_mergedConvolution32f.Enable() ? String(" ") + _mergedConvolution32f.Info() : String());
    }

    bool MergedConvolution32fLayer::Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
    {
        if (src->GetType() != TensorType32f || dst->GetType() != TensorType32f)
            SYNET_ERROR("MergedConvolution32fLayer supports only FP32 input and output!");
        AlgParam& a = this->_alg;
        size_t directIdx, depthwiseIdx;
        if (a.conv[0].group == 1 && a.conv[1].IsDepthwise())
            directIdx = 0, depthwiseIdx = 1;
        else if (a.count == 2 && a.conv[0].IsDepthwise() && a.conv[1].Is1x1())
            directIdx = 1, depthwiseIdx = 0;
        else
            assert(0);

        switch (a.conv[directIdx].activation)
        {
        case ActivationFunctionTypeIdentity: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeIdentity, 0>; break;
        case ActivationFunctionTypeRelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRelu, 0>; break;
        case ActivationFunctionTypeLeakyRelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeLeakyRelu, 0>; break;
        case ActivationFunctionTypeRestrictRange: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRestrictRange, 0>; break;
        case ActivationFunctionTypePrelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypePrelu, 0>; break;
        case ActivationFunctionTypeElu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeElu, 0>; break;
        case ActivationFunctionTypeHswish: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHswish, 0>; break;
        case ActivationFunctionTypeMish: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeMish, 0>; break;
        case ActivationFunctionTypeHardSigmoid: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHardSigmoid, 0>; break;
        case ActivationFunctionTypeSwish: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeSwish, 0>; break;
        case ActivationFunctionTypeGelu: _convolution[directIdx] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeGelu, 0>; break;
        default: assert(0);
        }

        switch (a.conv[depthwiseIdx].activation)
        {
        case ActivationFunctionTypeIdentity: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeIdentity>; break;
        case ActivationFunctionTypeRelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeRelu>; break;
        case ActivationFunctionTypeLeakyRelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeLeakyRelu>; break;
        case ActivationFunctionTypeRestrictRange: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeRestrictRange>; break;
        case ActivationFunctionTypePrelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypePrelu>; break;
        case ActivationFunctionTypeElu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeElu>; break;
        case ActivationFunctionTypeHswish: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeHswish>; break;
        case ActivationFunctionTypeMish: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeMish>; break;
        case ActivationFunctionTypeHardSigmoid: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeHardSigmoid>; break;
        case ActivationFunctionTypeSwish: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeSwish>; break;
        case ActivationFunctionTypeGelu: _convolution[depthwiseIdx] = Detail::MergedConvolutionLayerDepthwise<float, ActivationFunctionTypeGelu>; break;
        default: assert(0);
        }

        if (a.count > 2)
        {
            assert(a.conv[2].Is1x1());
            if (a.add)
            {
                switch (a.conv[2].activation)
                {
                case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeIdentity, 1>; break;
                case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRelu, 1>; break;
                case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeLeakyRelu, 1>; break;
                case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRestrictRange, 1>; break;
                case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypePrelu, 1>; break;
                case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeElu, 1>; break;
                case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHswish, 1>; break;
                case ActivationFunctionTypeMish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeMish, 1>; break;
                case ActivationFunctionTypeHardSigmoid: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHardSigmoid, 1>; break;
                case ActivationFunctionTypeSwish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeSwish, 1>; break;
                case ActivationFunctionTypeGelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeGelu, 1>; break;
                default: assert(0);
                }
            }
            else
            {
                switch (a.conv[2].activation)
                {
                case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeIdentity, 0>; break;
                case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRelu, 0>; break;
                case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeLeakyRelu, 0>; break;
                case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeRestrictRange, 0>; break;
                case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypePrelu, 0>; break;
                case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeElu, 0>; break;
                case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHswish, 0>; break;
                case ActivationFunctionTypeMish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeMish, 0>; break;
                case ActivationFunctionTypeHardSigmoid: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeHardSigmoid, 0>; break;
                case ActivationFunctionTypeSwish: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeSwish, 0>; break;
                case ActivationFunctionTypeGelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<float, ActivationFunctionTypeGelu, 0>; break;
                default: assert(0);
                }
            }
        }

        const ConvParam& back = a.conv[a.count - 1];
        dst->Reshape(TensorType32f, Shp(a.batch, back.dstH, back.dstW, back.dstC), src->Format());

        _mergedConvolution32f.Init(a.batch, a.conv, a.count, a.add, Bf16(), this->Options().bf16RoundTest);
        if (_mergedConvolution32f.Enable())
        {
            Base::Extend32f(buf, 0, Shp(_mergedConvolution32f.ExternalBufferSize()));
            _mergedConvolution32f.SetParams(a.weight, a.internal, a.bias, a.params);
        }
        else
        {
            Base::Extend32f(buf, 0, a.conv[0].DstShape(1));
            if (a.count > 2)
                Base::Extend32f(buf, 1, a.conv[1].DstShape(1));
        }

        return true;
    }

    void MergedConvolution32fLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
            ForwardCpu(src[0]->CpuData(), Base::Buf32f(buf, 0), Base::Buf32f(buf, 1), dst[0]->CpuData());
    }

    void MergedConvolution32fLayer::ForwardCpu(const float * src, float* buf0, float* buf1, float* dst)
    {
        if (_mergedConvolution32f.Enable())
            _mergedConvolution32f.Forward(src, buf0, dst);
        else
        {
            const AlgParam& a = this->_alg;
            for (size_t b = 0; b < a.batch; ++b)
            {
                _convolution[0](src, a.conv[0], a.weight[0], a.bias[0], a.params[0], buf0);
                if (a.count > 2)
                {
                    _convolution[1](buf0, a.conv[1], a.weight[1], a.bias[1], a.params[1], buf1);
                    if (a.add)
                        memcpy(dst, src, sizeof(float) * a.dSize);
                    _convolution[2](buf1, a.conv[2], a.weight[2], a.bias[2], a.params[2], dst);
                }
                else
                    _convolution[1](buf0, a.conv[1], a.weight[1], a.bias[1], a.params[1], dst);
                src += a.sSize;
                dst += a.dSize;
            }
        }
    }

    bool MergedConvolution32fLayer::Bf16() const
    {
        const MergedConvolutionParam& p = this->Param().mergedConvolution();
        for (size_t c = 0; c < p.conv().size(); ++c)
        {
            if (p.conv()[c].quantizationLevel() == TensorType16b)
                return true;
        }
        return false;
    }
}
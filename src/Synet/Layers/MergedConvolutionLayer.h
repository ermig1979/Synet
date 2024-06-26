/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layer.h"
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Layers/PreluLayer.h"

namespace Synet
{
    namespace Detail
    {
        template<class T, ActivationFunctionType activation> struct Activation
        {
            static T Func(T value, const T * params, size_t offset);
        };

        template<class T> struct Activation<T, ActivationFunctionTypeIdentity>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return value;
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeRelu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return Max(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeLeakyRelu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return Max(T(0), value) + params[0] * Min(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeRestrictRange>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return Min(Max(params[0], value), params[1]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypePrelu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return Max(T(0), value) + params[offset] * Min(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeElu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return CpuElu(value, params[0]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeHswish>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return CpuHswish(value, params[0], params[1]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeMish>
        {
            static SYNET_INLINE T Func(T value, const T* params, size_t offset)
            {
                return CpuMish(value, params[0]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeHardSigmoid>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return CpuHardSigmoid(value, params[0], params[1]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeSwish>
        {
            static SYNET_INLINE T Func(T value, const T* params, size_t offset)
            {
                return CpuSwish(value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeGelu>
        {
            static SYNET_INLINE T Func(T value, const T* params, size_t offset)
            {
                return CpuGelu(value);
            }
        };

        template<class T, ActivationFunctionType activation> void MergedConvolutionLayerDepthwise(
            const T * src, const ConvParam & conv, const T * weight, const T * bias, const T * params, T * dst)
        {
            for (size_t dy = 0; dy < conv.dstH; ++dy)
            {
                for (size_t dx = 0; dx < conv.dstW; ++dx)
                {
                    for (size_t c = 0; c < conv.srcC; ++c)
                    {
                        T sum = bias ? bias[c] : 0;
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
                                        const T * pw = weight + (ky * conv.kernelX + kx) * conv.srcC + c;
                                        const T * ps = src + (sy * conv.srcW + sx) * conv.srcC + c;
                                        sum += ps[0]*pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activation<T, activation>::Func(sum, params, c);
                    }
                    dst += conv.srcC;
                }
            }
        }

        const size_t MCC_MIN = 2, MCC_MAX = 3;
    }

    //-------------------------------------------------------------------------------------------------

    class MergedConvolutionLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolutionLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual void CompactWeight();

        virtual int64_t Flop() const;

    protected:

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst) = 0;
        virtual String InternalInfo() const = 0;

        struct AlgParam
        {
            bool biasTerm[Detail::MCC_MAX];
            int internal[Detail::MCC_MAX], add;
            size_t index[Detail::MCC_MAX], sSize, dSize, batch, count;
            ConvParam conv[Detail::MCC_MAX];
            float actParam[Detail::MCC_MAX][2];
            const float * weight[Detail::MCC_MAX], * bias[Detail::MCC_MAX], * params[Detail::MCC_MAX];

            bool IsCdc() const { return count == 3; }
            bool IsCd() const { return count == 2 && conv[0].group == 1; }
            bool IsDc() const { return conv[0].group != 1; }
        } _alg;
    };
}
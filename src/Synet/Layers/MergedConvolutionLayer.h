/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/HswishLayer.h"

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
                return Detail::HswishCpu(value, params[0], params[1]);
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

    template <class T> class MergedConvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolutionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 1 && src[0]->Count() == 4 && src[0]->Format() == TensorFormatNhwc);

            const MergedConvolutionParam & p = this->Param().mergedConvolution();
            const ConvolutionParam * conv = p.conv().data();
            _count = p.conv().size();
            assert(_count >= Detail::MCC_MIN && _count <= Detail::MCC_MAX);
            const Tensors & weight = this->Weight();

            for (size_t i = 0, next = 0; i < _count; ++i)
            {
                _conv[i].Set(conv[i]);
                if(i)
                    _conv[i].Set(_conv[i - 1], true);
                else
                    _conv[i].Set(*src[0], *dst[0], true);

                _index[i] = next++;
                const Tensor & w = weight[_index[i]];
                assert(w.Shape() == _conv[i].WeightShape(true, true) && w.Format() == src[0]->Format());
                _weight[i] = w.CpuData();

                _biasTerm[i] = conv[i].biasTerm();
                if (_biasTerm[i])
                {
                    const Tensor & b = weight[next++];
                    assert(b.Size() == _conv[i].dstC);
                    _bias[i] = b.CpuData();
                }
                else
                    _bias[i] = NULL;

                if (_conv[i].activation == ActivationFunctionTypePrelu)
                {
                    const Tensor & p = weight[next++];
                    if (p.Size() == 1)
                        _conv[i].activation = ActivationFunctionTypeLeakyRelu;
                    else
                        assert(p.Size() == _conv[i].dstC);
                    _params[i] = p.CpuData();
                }
                else
                {
                    _actParam[i][0] = conv[i].activationParam0();
                    _actParam[i][1] = conv[i].activationParam1();
                    _params[i] = _actParam[i];
                }
                _internal[i] = 0;
            }

            _add = (!this->Is8i() && _count == 3 && p.add()) ? 1 : 0;
            _batch = src[0]->Axis(0);

            Reshape(src[0], buf, dst[0]);

            _sSize = src[0]->Size(1);
            _dSize = dst[0]->Size(1);
            if(_add)
                assert(_sSize == _dSize);

            std::stringstream desc;
            desc << _count << ": " << _batch << "x" << _conv[0].srcC << "x" << _conv[0].srcH << "x" << _conv[0].srcW;
            for(size_t i = 0; i < _count; ++i)
                desc << "-" << (_conv[i].IsDepthwise() ? String("") : ValueToString(_conv[i].dstC) + "x") << _conv[i].kernelY << "x" << _conv[i].strideY;
            desc << " " << (this->Is8i() ? "int8" : "fp32");
            this->UsePerfStat(desc.str(), Flop());
        }

        virtual void CompactWeight()
        {
            for(size_t i = 0; i < _count; ++i)
                if (_internal[i])
                    ((Tensor&)this->Weight()[_index[i]]).Clear();
        }

        virtual int64_t Flop() const
        {
            int64_t flop = 0;
            for (size_t i = 0; i < _count; ++i)
                flop += _batch * _conv[i].kernelY * _conv[i].kernelX * _conv[i].srcC * _conv[i].dstH * _conv[i].dstW * _conv[i].dstC / _conv[i].group * 2;
            return flop;
        }

    protected:

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst) = 0;

        bool _biasTerm[Detail::MCC_MAX];
        int _internal[Detail::MCC_MAX], _add;
        size_t _index[Detail::MCC_MAX];
        ConvParam _conv[Detail::MCC_MAX];
        size_t _sSize, _dSize, _batch, _count;
        float _actParam[Detail::MCC_MAX][2];
        const Type * _weight[Detail::MCC_MAX], * _bias[Detail::MCC_MAX], * _params[Detail::MCC_MAX];
    };
}
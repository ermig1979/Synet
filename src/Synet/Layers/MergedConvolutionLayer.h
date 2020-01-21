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
                return std::max(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeLeakyRelu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return std::max(T(0), value) + params[0] * std::min(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeRestrictRange>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return std::min(std::max(params[0], value), params[1]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypePrelu>
        {
            static SYNET_INLINE T Func(T value, const T * params, size_t offset)
            {
                return std::max(T(0), value) + params[offset] * std::min(T(0), value);
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

        const size_t MCC = 3;
    }

    template <class T> class MergedConvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolutionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0]->Format() == TensorFormatNhwc);
            assert(this->Param().mergedConvolution().conv().size() == Detail::MCC);

            const MergedConvolutionParam & p = this->Param().mergedConvolution();
            const ConvolutionParam * conv = p.conv().data();
            const Tensors & weight = this->Weight();

            for (size_t i = 0, next = 0; i < Detail::MCC; ++i)
            {
                _conv[i].Set(conv[i]);
                if(i)
                    _conv[i].Set(_conv[i - 1], true);
                else
                    _conv[i].Set(*src[0], true);

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

            assert(_conv[0].group == 1);
            switch (_conv[0].activation)
            {
            case ActivationFunctionTypeIdentity: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 0>; break;
            case ActivationFunctionTypeRelu: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 0>; break;
            case ActivationFunctionTypeLeakyRelu: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 0>; break;
            case ActivationFunctionTypeRestrictRange: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 0>; break;
            case ActivationFunctionTypePrelu: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 0>; break;
            case ActivationFunctionTypeElu: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 0>; break;
            case ActivationFunctionTypeHswish: _convolution[0] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 0>; break;
            default: assert(0);
            }

            assert(_conv[1].IsDepthwise());
            switch (_conv[1].activation)
            {
            case ActivationFunctionTypeIdentity: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeIdentity>; break;
            case ActivationFunctionTypeRelu: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeRelu>; break;
            case ActivationFunctionTypeLeakyRelu: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeLeakyRelu>; break;
            case ActivationFunctionTypeRestrictRange: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeRestrictRange>; break;
            case ActivationFunctionTypePrelu: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypePrelu>; break;
            case ActivationFunctionTypeElu: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeElu>; break;
            case ActivationFunctionTypeHswish: _convolution[1] = Detail::MergedConvolutionLayerDepthwise<T, ActivationFunctionTypeHswish>; break;
            default: assert(0);
            }

            _add = p.add() ? 1 : 0;
            assert(_conv[2].group == 1);
            if (_add)
            {
                switch (_conv[2].activation)
                {
                case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 1>; break;
                case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 1>; break;
                case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 1>; break;
                case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 1>; break;
                case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 1>; break;
                case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 1>; break;
                case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 1>; break;
                default: assert(0);
                }
            }
            else
            {
                switch (_conv[2].activation)
                {
                case ActivationFunctionTypeIdentity: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeIdentity, 0>; break;
                case ActivationFunctionTypeRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRelu, 0>; break;
                case ActivationFunctionTypeLeakyRelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeLeakyRelu, 0>; break;
                case ActivationFunctionTypeRestrictRange: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeRestrictRange, 0>; break;
                case ActivationFunctionTypePrelu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypePrelu, 0>; break;
                case ActivationFunctionTypeElu: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeElu, 0>; break;
                case ActivationFunctionTypeHswish: _convolution[2] = Detail::MergedConvolutionLayerDirect<T, ActivationFunctionTypeHswish, 0>; break;
                default: assert(0);
                }
            }

            _axis = conv[0].axis();
            assert(_axis == conv[1].axis() && _axis == conv[2].axis() && src[0]->Count() == _axis + 3);

            _num = src[0]->Size(0, _axis);
            Shape dstShape = src[0]->Shape();
            dstShape[_axis + 0] = _conv[2].dstH;
            dstShape[_axis + 1] = _conv[2].dstW;
            dstShape[_axis + 2] = _conv[2].dstC;

            for (size_t i = 0; i < dst.size(); ++i)
                dst[i]->Reshape(dstShape, src[0]->Format());

            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
            if(_add)
                assert(_srcSize == _dstSize);

            _mergedConvolution32f.Init(_num, _conv, Detail::MCC, _add);
            if (_mergedConvolution32f.Enable())
            {
                buf[0]->Extend({ _mergedConvolution32f.ExternalBufferSize() });
                _mergedConvolution32f.SetParams(_weight, _internal, _bias, _params);
            }
            else
            {
                buf[0]->Extend(Shape({ _conv[0].dstC*_conv[0].dstH*_conv[0].dstW + _conv[1].dstC *_conv[1].dstH *_conv[1].dstW }));
            }
            std::stringstream desc;
            desc << _num << "x" << _conv[0].srcC << "x" << _conv[0].srcH << "x" << _conv[0].srcW;
            desc << "-" << _conv[0].dstC << "x" << _conv[0].kernelY << "x" << _conv[0].strideY;
            desc << "-" << _conv[1].kernelY << "x" << _conv[1].strideY << "-" << _conv[2].dstC;
            int64_t flop = 0;
            for(size_t i = 0; i < Detail::MCC; ++i)
                flop += _num * _conv[i].kernelY * _conv[i].kernelX * _conv[i].srcC * _conv[i].dstH * _conv[i].dstW * _conv[i].dstC / _conv[i].group * 2;
            this->UsePerfStat(desc.str(), flop);
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _mergedConvolution32f.InternalBufferSize() * sizeof(Type);
        }

        virtual void CompactWeight()
        {
            for(size_t i = 0; i < Detail::MCC; ++i)
                if (_internal[i])
                    ((Tensor&)this->Weight()[_index[i]]).Clear();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            for (int i = 0; i < src.size(); ++i)
                ForwardCpu(src[i]->CpuData(), buf[0]->CpuData(), dst[i]->CpuData());
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
            if (_mergedConvolution32f.Enable())
                _mergedConvolution32f.Forward(src, buf, dst);
            else
            {
                T * buf1 = buf + _conv[0].dstC * _conv[0].dstH * _conv[0].dstW;
                for (size_t n = 0; n < _num; ++n)
                {
                    _convolution[0](src, _conv[0], _weight[0], _bias[0], _params[0], buf);
                    _convolution[1](buf, _conv[1], _weight[1], _bias[1], _params[1], buf1);
                    if (_add)
                        memcpy(dst, src, sizeof(T) * _dstSize);
                    _convolution[2](buf1, _conv[2], _weight[2], _bias[2], _params[2], dst);
                    src += _srcSize;
                    dst += _dstSize;
                }
            }
        }

    private:
        bool _biasTerm[Detail::MCC];
        int _internal[Detail::MCC], _add;
        size_t _index[Detail::MCC];
        ConvParam _conv[Detail::MCC];
        size_t _axis, _srcSize, _dstSize, _num;
        float _actParam[Detail::MCC][2];
        const Type * _weight[Detail::MCC], * _bias[Detail::MCC], * _params[Detail::MCC];

        typedef void(*ConvolutionBiasActivationPtr)(const T * src, const ConvParam & conv, const T * weight, const T * bias, const T * params, T * dst);
        ConvolutionBiasActivationPtr _convolution[Detail::MCC];

        MergedConvolution32f<Type> _mergedConvolution32f;
    };
}
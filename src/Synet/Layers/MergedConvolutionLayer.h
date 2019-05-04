/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

namespace Synet
{
    namespace Detail
    {
        template<class T, ActivationFunctionType type> struct Activation
        {
            static T Function(T value, const T * params, size_t offset);
        };

        template<class T> struct Activation<T, ActivationFunctionTypeIdentity>
        {
            static SYNET_INLINE T Function(T value, const T * params, size_t offset)
            {
                return value;
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeRelu>
        {
            static SYNET_INLINE T Function(T value, const T * params, size_t offset)
            {
                return std::max(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeLeakyRelu>
        {
            static SYNET_INLINE T Function(T value, const T * params, size_t offset)
            {
                return std::max(T(0), value) + params[0] * std::min(T(0), value);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypeRestrictRange>
        {
            static SYNET_INLINE T Function(T value, const T * params, size_t offset)
            {
                return std::min(std::max(params[0], value), params[1]);
            }
        };

        template<class T> struct Activation<T, ActivationFunctionTypePrelu>
        {
            static SYNET_INLINE T Function(T value, const T * params, size_t offset)
            {
                return std::max(T(0), value) + params[offset] * std::min(T(0), value);
            }
        };

        template<class T, ActivationFunctionType type> void MergedConvolutionLayerDepthwiseConvolutionBiasActivation(const T * src,
            size_t srcH, size_t srcW, size_t srcC, size_t dstH, size_t dstW, 
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX,
            const T * weight, const T * bias, const T * params, T * dst)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        T sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const T * pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const T * ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0]*pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activation<T, type>::Function(sum, params, c);
                    }
                    dst += srcC;
                }
            }
        }

        template<class T, ActivationFunctionType type> void MergedConvolutionLayerBiasActivation(const T * src, size_t size, size_t count, const T * bias, const T * params, T * dst)
        {
            if (bias)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    for (size_t c = 0; c < count; ++c)
                        dst[c] = Activation<T, type>::Function(src[c] + bias[c], params, c);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                for (size_t i = 0; i < size; ++i)
                {
                    for (size_t c = 0; c < count; ++c)
                        dst[c] = Activation<T, type>::Function(src[c], params, c);
                    src += count;
                    dst += count;
                }
            }
        }
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

            const ConvolutionParam & conv0 = this->Param().mergedConvolution().conv0();
            const ConvolutionParam & conv1 = this->Param().mergedConvolution().conv1();
            const Tensors & weight = this->Weight();
            size_t next = 0;

            const Shape & ker0 = conv0.kernel();
            assert(ker0.size() == 1 || ker0.size() == 2);
            _kernelY = ker0[0];
            _kernelX = ker0.size() > 1 ? ker0[1] : _kernelY;
            assert(_kernelY > 0 && _kernelX > 0);

            const Shape & str0 = conv0.stride();
            assert(str0.size() == 1 || str0.size() == 2);
            _strideY = str0[0];
            _strideX = str0.size() > 1 ? str0[1] : _strideY;
            assert(_strideY > 0 && _strideX > 0);

            const Shape & pad0 = conv0.pad();
            assert(pad0.size() <= 4 && pad0.size() != 3);
            _padY = pad0.size() > 0 ? pad0[0] : 0;
            _padX = pad0.size() > 1 ? pad0[1] : _padY;
            _padH = pad0.size() > 2 ? pad0[2] : _padY;
            _padW = pad0.size() > 3 ? pad0[3] : _padX;
            assert(_padY >= 0 && _padX >= 0 && _padH >= 0 && _padW >= 0);

            const Shape & dil0 = conv0.dilation();
            assert(dil0.size() == 0 || (dil0.size() == 1 && dil0[0] == 1) || (dil0.size() == 2 && dil0[0] == 1 && dil0[1] == 1));

            _srcC = conv0.outputNum();
            assert(conv0.group() == _srcC);
            _index[0] = next++;
            assert(weight[_index[0]].Shape() == Shape({ _kernelY, _kernelX, 1, _srcC }) && weight[_index[0]].Format() == TensorFormatNhwc);
            _weight[0] = weight[_index[0]].CpuData();
            _biasTerm0 = conv0.biasTerm();
            _index[1] = _biasTerm0 ? next++ : -1;
            if (_biasTerm0)
            {
                assert(weight[_index[1]].Size() == _srcC);
                _bias[0] = weight[_index[1]].CpuData();
            }
            else
                _bias[0] = NULL;

            _activation0 = conv0.activationType();
            _params0[0] = conv0.activationParam0();
            _params0[1] = conv0.activationParam1();
            _index[2] = _activation0 == ActivationFunctionTypePrelu ? next++ : -1;
            if (_activation0 == ActivationFunctionTypePrelu)
            {
                const Tensor & params0 = weight[_index[2]];
                if (params0.Size() == 1)
                    _activation0 = ActivationFunctionTypeLeakyRelu;
                else
                    assert(params0.Size() == _srcC);
                _params[0] = params0.CpuData();
            }
            else
                _params[0] = _params0;
            switch (_activation0)
            {
            case ActivationFunctionTypeIdentity: _preProcessor = Detail::MergedConvolutionLayerDepthwiseConvolutionBiasActivation<T, ActivationFunctionTypeIdentity>; break;
            case ActivationFunctionTypeRelu: _preProcessor = Detail::MergedConvolutionLayerDepthwiseConvolutionBiasActivation<T, ActivationFunctionTypeRelu>; break;
            case ActivationFunctionTypeLeakyRelu: _preProcessor = Detail::MergedConvolutionLayerDepthwiseConvolutionBiasActivation<T, ActivationFunctionTypeLeakyRelu>; break;
            case ActivationFunctionTypeRestrictRange: _preProcessor = Detail::MergedConvolutionLayerDepthwiseConvolutionBiasActivation<T, ActivationFunctionTypeRestrictRange>; break;
            case ActivationFunctionTypePrelu: _preProcessor = Detail::MergedConvolutionLayerDepthwiseConvolutionBiasActivation<T, ActivationFunctionTypePrelu>; break;
            default: assert(0);
            }

            const Shape & ker1 = conv1.kernel();
            assert((ker1.size() == 1 && ker1[0] == 1) || (ker1.size() == 2 && ker1[0] == 1 && ker1[1] == 1));

            const Shape & str1 = conv1.stride();
            assert(str1.size() == 0 || (str1.size() == 1 && str1[0] == 1) || (str1.size() == 2 && str1[0] == 1 && str1[1] == 1));

            const Shape & pad1 = conv1.pad();
            assert(pad1.size() == 0 || (pad1.size() == 1 && pad1[0] == 0) || (pad1.size() == 2 && pad1[0] == 0 && pad1[1] == 0) ||
                (pad1.size() == 4 && pad1[0] == 0 && pad1[1] == 0 && pad1[2] == 0 && pad1[3] == 0));

            const Shape & dil1 = conv1.dilation();
            assert(dil1.size() == 0 || (dil1.size() == 1 && dil1[0] == 1) || (dil1.size() == 2 && dil1[0] == 1 && dil1[1] == 1));

            _dstC = conv1.outputNum();
            assert(conv1.group() == 1);
            _biasTerm1 = conv1.biasTerm();
            _index[3] = next++;
            assert(weight[_index[3]].Shape() == Shape({ 1, 1, _srcC, _dstC }) && weight[_index[3]].Format() == TensorFormatNhwc);
            _weight[1] = weight[_index[3]].CpuData();
            _index[4] = _biasTerm1 ? next++ : -1;
            if (_biasTerm1)
            {
                assert(weight[_index[4]].Size() == _dstC);
                _bias[1] = weight[_index[4]].CpuData();
            }
            else
                _bias[1] = NULL;

            _activation1 = conv1.activationType();
            _params1[0] = conv1.activationParam0();
            _params1[1] = conv1.activationParam1();
            _index[5] = _activation1 == ActivationFunctionTypePrelu ? next++ : -1;
            if (_activation1 == ActivationFunctionTypePrelu)
            {
                const Tensor & params1 = weight[_index[5]];
                if (params1.Size() == 1)
                    _activation1 = ActivationFunctionTypeLeakyRelu;
                else
                    assert(params1.Size() == _dstC);
                _params[1] = params1.CpuData();
            }
            else
                _params[1] = _params1;
            switch (_activation1)
            {
            case ActivationFunctionTypeIdentity: _postProcessor = Detail::MergedConvolutionLayerBiasActivation<T, ActivationFunctionTypeIdentity>; break;
            case ActivationFunctionTypeRelu: _postProcessor = Detail::MergedConvolutionLayerBiasActivation<T, ActivationFunctionTypeRelu>; break;
            case ActivationFunctionTypeLeakyRelu: _postProcessor = Detail::MergedConvolutionLayerBiasActivation<T, ActivationFunctionTypeLeakyRelu>; break;
            case ActivationFunctionTypeRestrictRange: _postProcessor = Detail::MergedConvolutionLayerBiasActivation<T, ActivationFunctionTypeRestrictRange>; break;
            case ActivationFunctionTypePrelu: _postProcessor = Detail::MergedConvolutionLayerBiasActivation<T, ActivationFunctionTypePrelu>; break;
            default: assert(0);
            }
            assert(weight.size() == next);

            _axis = conv0.axis();
            assert(_axis = conv1.axis() && src[0]->Count() == _axis + 3);

            _num = src[0]->Size(0, _axis);
            _srcH = src[0]->Axis(-3);
            _srcW = src[0]->Axis(-2);
            assert(_srcC == src[0]->Axis(-1));
            _dstH = (_srcH + _padY + _padH - _kernelY) / _strideY + 1;
            _dstW = (_srcW + _padX + _padW - _kernelX) / _strideX + 1;
            Shape dstShape = src[0]->Shape();
            dstShape[_axis + 0] = _dstH;
            dstShape[_axis + 1] = _dstW;
            dstShape[_axis + 2] = _dstC;

            for (size_t i = 0; i < dst.size(); ++i)
                dst[i]->Reshape(dstShape, src[0]->Format());

            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);

            _mergedConvolution.Init(_num, _srcC, _srcH, _srcW, _dstC, _kernelY, _kernelX,
                _strideY, _strideX, _padY, _padX, _padH, _padW, _activation0, _activation1,
#if defined(SYNET_BLIS_ENABLE)
                Synet::BlisGemm32fNN
#else
                NULL
#endif
            );
            if (_mergedConvolution.Enable())
            {
                buf[0]->Extend({ _mergedConvolution.ExternalBufferSize() });
                _mergedConvolution.SetParams(_weight[0], _weight[1], &_internal, _bias[0], _bias[1], _params[0], _params[1]);
            }
            else
            {
                _internal = 0;
                buf[0]->Extend(Shape({ _dstH, _dstW, _srcC }));
            }
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _mergedConvolution.InternalBufferSize() * sizeof(Type);
        }

        virtual void CompactWeight()
        {
            if (_internal)
                ((Tensor&)this->Weight()[_index[3]]).Clear();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            for (int i = 0; i < src.size(); ++i)
                ForwardCpu(src[i]->CpuData(), buf[0]->CpuData(), dst[i]->CpuData());
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
#ifdef SYNET_SIZE_STATISTIC
            std::stringstream ss;
            ss << "i=" << _num << "x" << _srcC << "x" << _srcH << "x" << _srcW << " o=" << _dstC << " k=" << _kernelY << " s=" << _strideY;
            SYNET_PERF_BLOCK(ss.str().c_str());
#else
            SYNET_PERF_FUNC();
#endif
            if (_mergedConvolution.Enable())
                _mergedConvolution.Forward(src, buf, dst);
            else
            {
                for (size_t n = 0; n < _num; ++n)
                {
                    _preProcessor(src, _srcH, _srcW, _srcC, _dstH, _dstW, _kernelY, _kernelX, _strideY, _strideX, _padY, _padX, _weight[0], _bias[0], _params[0], buf);
                    CpuGemm(CblasNoTrans, CblasNoTrans, _dstH * _dstW, _dstC, _srcC, Type(1), buf, _srcC, _weight[1], _dstC, Type(0), dst, _dstC);
                    _postProcessor(dst, _dstH * _dstW, _dstC, _bias[1], _params[1], dst);
                    src += _srcSize;
                    dst += _dstSize;
                }
            }
        }

    private:
        bool _biasTerm0, _biasTerm1;
        int _internal;
        size_t _index[6];
        size_t _kernelY, _kernelX, _strideY, _strideX, _padY, _padX, _padH, _padW;
        size_t _axis, _num, _srcC, _srcH, _srcW, _dstC, _dstH, _dstW, _srcSize, _dstSize;
        ActivationFunctionType _activation0, _activation1;
        float _params0[2], _params1[2];
        const Type * _weight[2], * _bias[2], * _params[2];

        typedef void(*DepthwiseConvolutionBiasActivationPtr)(const T * src, size_t srcH, size_t srcW, size_t srcC, size_t dstH, size_t dstW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, const T * weight, const T * bias, const T * params, T * dst);
        DepthwiseConvolutionBiasActivationPtr _preProcessor;
        typedef void(*BiasActivationPtr)(const T * src, size_t size, size_t count, const T * bias, const T * params, T * dst);
        BiasActivationPtr _postProcessor;

        MergedConvolution<Type> _mergedConvolution;
    };
}
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
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Winograd.h"
#include "Synet/Utils/Convolution.h"
#include "Synet/Layers/PreluLayer.h"

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
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ConvolutionParam & param = this->Param().convolution();
            const Tensors & weight = this->Weight();

            const Shape & kernel = param.kernel();
            assert(kernel.size() == 1 || kernel.size() == 2);
            _kernelY = kernel[0];
            _kernelX = kernel.size() > 1 ? kernel[1] : _kernelY;
            assert(_kernelY > 0 && _kernelX > 0);

            const Shape & stride = param.stride();
            assert(stride.size() <= 2);
            _strideY = stride.size() > 0 ? stride[0] : 1;
            _strideX = stride.size() > 1 ? stride[1] : _strideY;
            assert(_strideY > 0 && _strideX > 0);

            const Shape & dilation = param.dilation();
            assert(dilation.size() <= 2);
            _dilationY = dilation.size() > 0 ? dilation[0] : 1;
            _dilationX = dilation.size() > 1 ? dilation[1] : _dilationY;
            assert(_dilationY > 0 && _dilationX > 0);

            const Shape & pad = param.pad();
            assert(pad.size() <= 4 && pad.size() != 3);
            _padY = pad.size() > 0 ? pad[0] : 0;
            _padX = pad.size() > 1 ? pad[1] : _padY;
            _padH = pad.size() > 2 ? pad[2] : _padY;
            _padW = pad.size() > 3 ? pad[3] : _padX;
            assert(_padY >= 0 && _padX >= 0 && _padH >= 0 && _padW >= 0);

            _is1x1 = _kernelY == 1 && _kernelX == 1 && _strideY == 1 && _strideX == 1 && _dilationY == 1 && _dilationX == 1 && _padY == 0 && _padX == 0 && _padH == 0 && _padW == 0;

            _group = param.group();
            _dstC = this->Param().convolution().outputNum();
            assert(_dstC  > 0 && _dstC % _group == 0);

            _biasTerm = param.biasTerm();
            if (_biasTerm)
                assert(weight[1].Size() == _dstC);

            _activation = param.activationType();
            _params[0] = param.activationParam0();
            _params[1] = param.activationParam1();
            assert(weight.size() == 1 + _biasTerm + (_activation == ActivationFunctionTypePrelu));
            if (_activation == ActivationFunctionTypePrelu)
            {
                if (weight.back().Size() == 1)
                {
                    _activation = ActivationFunctionTypeLeakyRelu;
                    _params[0] = weight.back().CpuData()[0];
                }
                else
                    assert(weight.back().Size() == _dstC);
            }

            _axis = param.axis();
            assert(src[0]->Count() == _axis + 3);

            _num = src[0]->Size(0, _axis);
            _trans = src[0]->Format() == TensorFormatNhwc;
            if (_trans)
            {
                _srcH = src[0]->Axis(-3);
                _srcW = src[0]->Axis(-2);
                _srcC = src[0]->Axis(-1);

                assert(weight[0].Shape() == Shape({ _kernelY, _kernelX, _srcC / _group, _dstC }) && weight[0].Format() == TensorFormatNhwc);
            }
            else
            {
                _srcC = src[0]->Axis(-3);
                _srcH = src[0]->Axis(-2);
                _srcW = src[0]->Axis(-1);

                assert(weight[0].Shape() == Shape({ _dstC, _srcC / _group, _kernelY, _kernelX }) && weight[0].Format() == TensorFormatNchw);
            }

            _dstH = (_srcH + _padY + _padH - (_dilationY * (_kernelY - 1) + 1)) / _strideY + 1;
            _dstW = (_srcW + _padX + _padW - (_dilationX * (_kernelX - 1) + 1)) / _strideX + 1;

            Shape dstShape(src[0]->Shape().begin(), src[0]->Shape().begin() + _axis);
            if (_trans)
            {
                dstShape.push_back(_dstH);
                dstShape.push_back(_dstW);
                dstShape.push_back(_dstC);

                _siW = _srcC * _kernelY * _kernelX / _group;
                _ldW = _dstC;
                _grW = _dstC / _group;

                _siS = _dstH * _dstW;
                _ldS = _siW;
                _grS = _siS * _siW;

                _siD = _dstC / _group;
                _ldD = _dstC;
                _grD = _siD;
            }
            else
            {
                dstShape.push_back(_dstC);
                dstShape.push_back(_dstH);
                dstShape.push_back(_dstW);

                _siW = _srcC * _kernelY * _kernelX / _group;
                _ldW = _siW;
                _grW = _dstC * _siW / _group;

                _siS = _dstH * _dstW;
                _ldS = _siS;
                _grS = _siS * _siW;

                _siD = _dstC / _group;
                _ldD = _dstH * _dstW;
                _grD = _siD * _siS;
            }

            for (size_t i = 0; i < dst.size(); ++i)
                dst[i]->Reshape(dstShape, Type(), src[0]->Format());

            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);

            _convolution.Init(_srcC, _srcH, _srcW, _trans, _dstC, _trans, _kernelY, _kernelX, _dilationY, _dilationX, _strideY, _strideX, _padY, _padX, _padH, _padW, _group, _activation);
            if (_convolution.Enable())
            {
                buf[0]->Extend({ _convolution.BufferSize() });
                int internal;
                _convolution.SetParams(weight[0].CpuData(), _trans, &internal, _biasTerm ? weight[1].CpuData() : NULL, 
                    _activation == ActivationFunctionTypePrelu ? weight.back().CpuData() : _params);
                if (internal && 0)
                    const_cast<Tensor&>(weight[0]).Clear();
            }
            else
                buf[0]->Extend(Shape({ _kernelY*_kernelX*_srcC, _dstH*_dstW }));
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            for (int i = 0; i < src.size(); ++i)
                for (int n = 0; n < this->_num; ++n)
                    ForwardCpu(src[i]->CpuData() + _srcSize * n, buf[0]->CpuData(), dst[i]->CpuData() + _dstSize * n);
        }

        void ForwardCpu(const T * src, T * buf, T * dst)
        {
#ifdef SYNET_SIZE_STATISTIC
            std::stringstream ss;
            ss << " i=" << _srcC << "x" << _srcH << "x" << _srcW << " o=" << _dstC << " k=" << _kernelY << " s=" << _strideY << " g=" << _group;
            SYNET_PERF_BLOCK(ss.str().c_str());
#else
            SYNET_PERF_FUNC();
#endif
            if (_convolution.Enable())
                _convolution.Forward(src, buf, dst);
            else
            {
                const Type * weight = this->Weight()[0].CpuData();
                if (!_is1x1)
                {
                    if (_trans)
                        Synet::ImgToRow(src, _srcH, _srcW, _srcC, _kernelY, _kernelX, _padY, _padX, _padH, _padW, _strideY, _strideX, _dilationY, _dilationX, _group, buf);
                    else
                        Synet::ImgToCol(src, _srcC, _srcH, _srcW, _kernelY, _kernelX, _padY, _padX, _padH, _padW, _strideY, _strideX, _dilationY, _dilationX, buf);
                    src = buf;
                }
                if (_trans)
                {
                    assert(_group == 1 || _group == _srcC);
                    for (size_t g = 0; g < _group; ++g)
                        CpuGemm(CblasNoTrans, CblasNoTrans, _siS, _siD, _siW, Type(1), src + _grS * g, _ldS, weight + _grW * g, _ldW, Type(0), dst + _grD * g, _ldD);
                }
                else
                {
                    for (size_t g = 0; g < _group; ++g)
                        CpuGemm(CblasNoTrans, CblasNoTrans, _siD, _siS, _siW, Type(1), weight + _grW * g, _ldW, src + _grS * g, _ldS, Type(0), dst + _grD * g, _ldD);
                }
                if (_biasTerm)
                    CpuAddBias(this->Weight()[1].CpuData(), _dstC, _dstH*_dstW, dst, _trans);
                switch (_activation)
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
                    Detail::PreluLayerForwardCpu(dst, this->Weight().back().CpuData(), _dstC, _dstH*_dstW, dst, _trans);
                    break;
                default:
                    assert(0);
                }
            }
        }

    private:
        bool _is1x1, _biasTerm;
        int _trans;
        size_t _kernelY, _kernelX, _strideY, _strideX, _dilationY, _dilationX, _padY, _padX, _padH, _padW;
        size_t _axis, _group, _num, _srcC, _srcH, _srcW, _dstC, _dstH, _dstW, _srcSize, _dstSize;
        size_t _ldW, _ldS, _ldD, _grW, _grS, _grD, _siW, _siS, _siD;
        ActivationFunctionType _activation;
        float _params[2];

        Convolution<Type> _convolution;
    };
}
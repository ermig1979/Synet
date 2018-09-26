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
#include "Synet/Gemm.h"
#include "Synet/ImgToCol.h"
#include "Synet/Winograd.h"

namespace Synet
{
    template <class T> class ConvolutionLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        ConvolutionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _biasTerm = this->Param().convolution().biasTerm();
            _axis = this->Param().convolution().axis();
            _group = this->Param().convolution().group();
            size_t firstSpatialAxis = _axis + 1;
            _spatialAxisNum = src[0]->Count() - firstSpatialAxis;

            const Shape & kernel = this->Param().convolution().kernel();
            assert(kernel.size() == 1 || kernel.size() == _spatialAxisNum);
            if (kernel.size() == 1)
                _kernelShape.resize(_spatialAxisNum, kernel[0]);
            else
                _kernelShape = kernel;
            for (size_t i = 0; i < _kernelShape.size(); ++i)
                assert(_kernelShape[i] > 0);

            const Shape & stride = this->Param().convolution().stride();
            if (stride.empty())
                _strideShape.resize(_spatialAxisNum, 1);
            else
            {
                assert(stride.size() == 1 || stride.size() == _spatialAxisNum);
                if (stride.size() == 1)
                    _strideShape.resize(_spatialAxisNum, stride[0]);
                else
                    _strideShape = stride;
            }
            for (size_t i = 0; i < _strideShape.size(); ++i)
                assert(_strideShape[i] > 0);

            const Shape & pad = this->Param().convolution().pad();
            if (pad.empty())
                _padShape.resize(_spatialAxisNum*2, 0);
            else
            {
                assert(pad.size() == 1 || pad.size() == _spatialAxisNum || pad.size() == _spatialAxisNum*2);
                if (pad.size() == 1)
                    _padShape.resize(_spatialAxisNum*2, pad[0]);
                else if (pad.size() == _spatialAxisNum)
                {
                    _padShape.resize(_spatialAxisNum * 2);
                    for (size_t i = 0; i < _spatialAxisNum; ++i)
                    {
                        _padShape[0 * _spatialAxisNum + i] = pad[i];
                        _padShape[1 * _spatialAxisNum + i] = pad[i];
                    }
                }
                else
                    _padShape = pad;
            }

            const Shape & dilation = this->Param().convolution().dilation();
            if (dilation.empty())
                _dilationShape.resize(_spatialAxisNum, 1);
            else
            {
                assert(dilation.size() == 1 || dilation.size() == _spatialAxisNum);
                if (dilation.size() == 1)
                    _dilationShape.resize(_spatialAxisNum, dilation[0]);
                else
                    _dilationShape = dilation;
            }
            for (size_t i = 0; i < _dilationShape.size(); ++i)
                assert(_dilationShape[i] > 0);

            _is1x1 = true;
            for (size_t i = 0; i < _spatialAxisNum; ++i)
            {
                if (_kernelShape[i] != 1 || _padShape[i] != 0 || _padShape[_spatialAxisNum + i] != 0 || _strideShape[i] != 1)
                {
                    _is1x1 = false;
                    break;
                }
            }
            _srcChannels = src[0]->Axis(_axis);
            _dstChannels = this->Param().convolution().outputNum();
            assert(_dstChannels  > 0 && _dstChannels % _group == 0);
            Shape weightShape(2 + _spatialAxisNum);
            weightShape[0] = _dstChannels;
            weightShape[1] = _srcChannels / _group;
            for (size_t i = 0; i < _spatialAxisNum; ++i)
                weightShape[2 + i] = _kernelShape[i];
            Shape biasShape(_biasTerm, _dstChannels);
            assert(this->Weight().size() == _biasTerm + 1);
            assert(this->Weight()[0].Shape() == weightShape);
            if (_biasTerm)
                assert(this->Weight()[1].Shape() == biasShape);
            _kernelSize = this->Weight()[0].Size(1);
            _weightOffset = _dstChannels * _kernelSize / _group;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            size_t firstSpatialAxis = _axis + 1;
            _num = src[0]->Size(0, _axis);
            _srcShape = src[0]->Shape();
            _dstShape.resize(_spatialAxisNum);
            for (size_t i = 0; i < _spatialAxisNum; ++i)
            {
                size_t kernelExtent = _dilationShape[i] * (_kernelShape[i] - 1) + 1;
                _dstShape[i] = (_srcShape[firstSpatialAxis + i] + _padShape[i] + _padShape[_spatialAxisNum + i] - kernelExtent) / _strideShape[i] + 1;
            }
            Shape dstShape(_srcShape.begin(), _srcShape.begin() + _axis);
            dstShape.push_back(_dstChannels);
            for (size_t i = 0; i < _spatialAxisNum; ++i)
                dstShape.push_back(_dstShape[i]);
            for (size_t i = 0; i < dst.size(); ++i)
                dst[i]->Reshape(dstShape);
            _dstSpatialSize = dst[0]->Size(firstSpatialAxis);
            _colOffset = _kernelSize * _dstSpatialSize;
            _dstOffset = _dstChannels * _dstSpatialSize / _group;
            _srcConvShape.resize(_spatialAxisNum + 1);
            for (size_t i = 0; i < _spatialAxisNum + 1; ++i)
                _srcConvShape[i] = src[0]->Axis(_axis + i);
            Shape colBufferShape;
            colBufferShape.push_back(_kernelSize * _group);
            for (int i = 0; i < _spatialAxisNum; ++i)
                colBufferShape.push_back(_dstShape[i]);
            _winograd.Init(_srcConvShape, _dstChannels, _kernelShape, _strideShape, _dilationShape, _padShape, _group);
            if (_winograd.Enable())
            {
                _winograd.SetFilter(this->Weight()[0].CpuData());
                buf[0]->Extend({ _winograd.SrcBufSize() });
                buf[1]->Extend({ _winograd.DstBufSize() });
#ifdef SYNET_WINOGRAD_COMPARE
                buf[0]->Extend(colBufferShape);
#endif
            }
            else
            {
                _direct.Init(_srcConvShape, _dstChannels, _kernelShape, _strideShape, _dilationShape, _padShape, _group);
                if(_direct.Enable())
                    buf[0]->Extend({ _direct.BufferSize() });
                else
                    buf[0]->Extend(colBufferShape);
            }
            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            for (int i = 0; i < src.size(); ++i)
                for (int n = 0; n < this->_num; ++n)
                    ForwardCpu(src[i]->CpuData() + _srcSize * n, buf[0]->CpuData(), buf[1]->CpuData(), dst[i]->CpuData() + _dstSize * n);
        }

        void ForwardCpu(const T * src, T * buf0, T * buf1, T * dst)
        {
#ifdef SYNET_CONVOLUTION_STATISTIC
            std::stringstream ss;
            ss << " i=" << _srcShape[1] << "x" << _srcShape[2] << "x" << _srcShape[3] << " o=" << _dstChannels << " k=" << _kernelShape[0] << " s=" << _strideShape[0] << " g=" << _group;
            SYNET_PERF_BLOCK(ss.str().c_str());
#else
            SYNET_PERF_FUNC();
#endif

            const Type * weight = this->Weight()[0].CpuData();

            if (_winograd.Enable())
            {
#ifdef SYNET_WINOGRAD_COMPARE
                {
                    std::stringstream ss;
                    ss << _dstChannels << "-" << _dstSpatialSize << "-" << _kernelSize << " img2col ";
                    SYNET_PERF_BLOCK(ss.str().c_str());
                    ImgToCol(src, buf0);
                    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstChannels, _dstSpatialSize, _kernelSize, Type(1.0), weight, buf0, Type(0.0), pDst);
                }
                {
                    std::stringstream ss;
                    ss << _dstConvChannels << "-" << _dstSpatialSize << "-" << _kernelSize << " winograd";
                    SYNET_PERF_BLOCK(ss.str().c_str());
                    _winograd.Convolution(src, buf0, buf1, dst);
                }
#else
                _winograd.Convolution(src, buf0, buf1, dst);
#endif
            }
            else if (_direct.Enable())
            {
                _direct.Convolution(src, weight, buf0, dst);
            }
            else
            {
                if (!_is1x1)
                {
                    ImgToCol(src, buf0);
                    src = buf0;
                }
                for (size_t g = 0; g < _group; ++g)
                {
                    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstChannels / _group, _dstSpatialSize, _kernelSize,
                        Type(1.0), weight + _weightOffset * g, src + _colOffset * g, Type(0.0), dst + _dstOffset * g);
                }
            }

            if (_biasTerm)
                CpuAddBias(this->Weight()[1].CpuData(), _dstChannels, _dstSpatialSize, dst);
        }

        void ImgToCol(const T * src, T * dst)
        {
            if (_spatialAxisNum == 2)
            {
                Synet::ImgToCol(src, _srcConvShape[0], _srcConvShape[1], _srcConvShape[2], _kernelShape[0], _kernelShape[1],
                    _padShape[0], _padShape[1], _padShape[2], _padShape[3], _strideShape[0], _strideShape[1], _dilationShape[0], _dilationShape[1], dst);
            }
            else
                assert(0);
        }

    private:
        Shape _srcShape, _kernelShape, _strideShape, _dilationShape, _padShape, _dstShape, _srcConvShape;
        bool _is1x1, _biasTerm;
        size_t _axis, _group, _spatialAxisNum, _srcChannels, _dstChannels, _weightOffset, _kernelSize;
        size_t _channelAxis, _num, _dstSpatialSize, _colOffset, _dstOffset, _srcSize, _dstSize;

        Winograd<Type> _winograd;

        struct Direct
        {
            Direct()
                : _enable(false)
            {
            }

            void Init(Shape src, size_t dst, Shape kernel, Shape stride, Shape dilation, Shape pad, size_t group)
            {
                if (stride[0] != 1 || stride[1] != 1 || dilation[0] != 1 || dilation[1] != 1)
                    return;
                if (!((pad[0] == 0 && pad[1] == 0) || (pad[0] == 1 && pad[1] == 1)))
                    return;
                if (group != 1)
                    return;
                if (kernel[0] == 3 && kernel[1] == 3)
                {
                    _srcC = src[0];
                    _srcH = src[1];
                    _srcW = src[2];
                    _dstC = dst;
                    if (pad[0] == 1 && pad[1] == 1)
                    {
                        _pad = true;
                        _dstH = _srcH;
                        _dstW = _srcW;
                    }
                    else
                    {
                        _pad = false;
                        _dstH = _srcH - 2;
                        _dstW = _srcW - 2;
                    }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
                    if (src[0] <= 16)
                    {
                        _enable = true;
                    }
                    else
#endif
                    return;
                }
            }

            bool Enable()
            {
                return _enable;
            }

            size_t BufferSize()
            {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
                if (_enable)
                    return _srcC*(_srcH + 2)*(_srcW + 2);
                else
#endif
                    return 1;
            }

            void Convolution(const Type * src, const Type * weight, Type * buffer, Type * dst)
            {
                SYNET_PERF_FUNC();

#ifdef SYNET_SIMD_LIBRARY_ENABLE
                int pad = _pad ? 1 : 0;
                size_t size = BufferSize();
                ::SimdNeuralConvolutionForward(src, _srcW, _srcH, _srcC, weight, 3, 3, pad, pad, 1, 1, 1, 1, buffer, &size, dst, _dstW, _dstH, _dstC, 0);
#endif
            }

        private:
            bool _enable, _pad;
            size_t _srcC, _srcW, _srcH, _dstC, _dstH, _dstW;
            size_t _group, _wStep, _dStep;

        } _direct;
    };
}
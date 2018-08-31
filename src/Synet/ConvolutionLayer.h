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
                _padShape.resize(_spatialAxisNum, 0);
            else
            {
                assert(pad.size() == 1 || pad.size() == _spatialAxisNum);
                if (pad.size() == 1)
                    _padShape.resize(_spatialAxisNum, pad[0]);
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
                if (_kernelShape[i] != 1 || _padShape[i] != 0 || _strideShape[i] != 1)
                {
                    _is1x1 = false;
                    break;
                }
            }
            _srcChannels = src[0]->Axis(_axis);
            _dstChannels = this->Param().convolution().outputNum();
            assert(_dstChannels  > 0 && _dstChannels % _group == 0);
            if (IsConv())
            {
                _srcConvChannels = _srcChannels;
                _dstConvChannels = _dstChannels;
            }
            else
            {
                _srcConvChannels = _dstChannels;
                _dstConvChannels = _srcChannels;
            }
            Shape weightShape(2 + _spatialAxisNum);
            weightShape[0] = _dstConvChannels;
            weightShape[1] = _srcConvChannels / _group;
            for (size_t i = 0; i < _spatialAxisNum; ++i)
                weightShape[2 + i] = _kernelShape[i];
            Shape biasShape(_biasTerm, _dstChannels);
            assert(this->Weight().size() == _biasTerm + 1);
            assert(this->Weight()[0].Shape() == weightShape);
            if (_biasTerm)
                assert(this->Weight()[1].Shape() == biasShape);
            _kernelSize = this->Weight()[0].Size(1);
            _weightOffset = _dstConvChannels * _kernelSize / _group;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            size_t firstSpatialAxis = _axis + 1;
            _num = src[0]->Size(0, _axis);
            _srcShape = src[0]->Shape();
            if (IsConv())
            {
                _dstShape.resize(_spatialAxisNum);
                for (size_t i = 0; i < _spatialAxisNum; ++i)
                {
                    size_t kernelExtent = _dilationShape[i] * (_kernelShape[i] - 1) + 1;
                    _dstShape[i] = (_srcShape[firstSpatialAxis + i] + 2 * _padShape[i] - kernelExtent) / _strideShape[i] + 1;
                }
            }
            Shape dstShape(_srcShape.begin(), _srcShape.begin() + _axis);
            dstShape.push_back(_dstChannels);
            for (size_t i = 0; i < _spatialAxisNum; ++i)
                dstShape.push_back(_dstShape[i]);
            for (size_t i = 0; i < dst.size(); ++i)
                dst[i]->Reshape(dstShape);
            if (IsConv())
                _dstConvSpatialSize = dst[0]->Size(firstSpatialAxis);
            else
                _dstConvSpatialSize = src[0]->Size(firstSpatialAxis);
            _colOffset = _kernelSize * _dstConvSpatialSize;
            _dstOffset = _dstChannels * _dstConvSpatialSize / _group;
            _srcConvShape.resize(_spatialAxisNum + 1);
            for (size_t i = 0; i < _spatialAxisNum + 1; ++i)
            {
                if (IsConv())
                    _srcConvShape[i] = src[0]->Axis(_axis + i);
                else
                    _srcConvShape[i] = dst[0]->Axis(_axis + i);
            }
            Shape colBufferShape;
            colBufferShape.push_back(_kernelSize * _group);
            for (int i = 0; i < _spatialAxisNum; ++i)
            {
                if (IsConv())
                    colBufferShape.push_back(_dstShape[i]);
                else
                    colBufferShape.push_back(_srcShape[i + 1]);
            }
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
                buf[0]->Extend(colBufferShape);
            }
            _srcSize = src[0]->Size(_axis);
            _dstSize = dst[0]->Size(_axis);
            _dstSpatialSize = dst[0]->Size(firstSpatialAxis);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            const Type * weight = this->Weight()[0].CpuData();
            for (int i = 0; i < src.size(); ++i)
            {
                for (int n = 0; n < this->_num; ++n)
                {
                    const Type * pSrc = src[i]->CpuData() + _srcSize * n;
                    Type * pDst = dst[i]->CpuData() + _dstSize * n;
                    Type * pBuf = (Type*)pSrc;
                    if (_winograd.Enable())
                    {
#ifdef SYNET_WINOGRAD_COMPARE
                        {
                            std::stringstream ss;
                            ss << _dstConvChannels << "-" << _dstConvSpatialSize << "-" << _kernelSize << " img2col ";
                            SYNET_PERF_BLOCK(ss.str().c_str());
                            ImgToCol(pSrc, buf[0]->CpuData());
                            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstConvChannels, _dstConvSpatialSize, _kernelSize, Type(1.0), weight, buf[0]->CpuData(), Type(0.0), pDst);
                        }
                        {
                            std::stringstream ss;
                            ss << _dstConvChannels << "-" << _dstConvSpatialSize << "-" << _kernelSize << " winograd";
                            SYNET_PERF_BLOCK(ss.str().c_str());
                            _winograd.Convolution(pSrc, buf[0]->CpuData(), buf[1]->CpuData(), pDst);
                        }
#else
                        _winograd.Convolution(pSrc, buf[0]->CpuData(), buf[1]->CpuData(), pDst);
#endif
                    }
                    else
                    {
                        if (!_is1x1)
                        {
                            pBuf = buf[0]->CpuData() + _dstSize * n;
                            ImgToCol(pSrc, pBuf);
                        }
                        for (size_t g = 0; g < _group; ++g)
                        {
                            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstConvChannels / _group, _dstConvSpatialSize, _kernelSize,
                                Type(1.0), weight + _weightOffset * g, pBuf + _colOffset * g, Type(0.0), pDst + _dstOffset * g);
                        }
                    }

                    if (_biasTerm)
                        CpuAddBias(this->Weight()[1].CpuData(), _dstChannels, _dstSpatialSize, pDst);
                }
            }
        }

        virtual bool IsConv() 
        {
            return true;
        }

        void ImgToCol(const T * src, T * dst)
        {
            if (_spatialAxisNum == 2)
            {
                Synet::ImgToCol(src, _srcConvShape[0], _srcConvShape[1], _srcConvShape[2], _kernelShape[0], _kernelShape[1],
                    _padShape[0], _padShape[1], _strideShape[0], _strideShape[1], _dilationShape[0], _dilationShape[1], dst);
            }
        }

    private:
        Shape _srcShape, _kernelShape, _strideShape, _dilationShape, _padShape, _dstShape, _srcConvShape;
        bool _is1x1, _biasTerm;
        size_t _axis, _group, _spatialAxisNum, _srcChannels, _dstChannels, _srcConvChannels, _dstConvChannels, _weightOffset, _kernelSize;
        size_t _channelAxis, _num, _dstConvSpatialSize, _dstSpatialSize, _colOffset, _dstOffset, _srcSize, _dstSize;

        Winograd<Type> _winograd;
    };
}
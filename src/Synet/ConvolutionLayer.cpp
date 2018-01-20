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

#include "Synet/ConvolutionLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> void ConvolutionLayer<T, A>::Setup(const ConvolutionLayer::TensorPtrs & src, const ConvolutionLayer::TensorPtrs & dst)
    {
        size_t firstSpatialAxis = _param.axis + 1;
        _spatialAxisNum = src[0]->Count() - firstSpatialAxis;
        if (_param.kernelX && _param.kernelY)
        {
            assert(_spatialAxisNum == 2);
            _kernelShape = { _param.kernelY, _param.kernelX };
        }
        if (_param.strideX && _param.strideY)
        {
            assert(_spatialAxisNum == 2);
            _strideShape = { _param.strideY, _param.strideX };
        }
        if (_param.padX && _param.padY)
        {
            assert(_spatialAxisNum == 2);
            _padShape = { _param.padY, _param.padX };
        }
        if (_param.dilationX && _param.dilationY)
        {
            assert(_spatialAxisNum == 2);
            _dilationShape = { _param.dilationY, _param.dilationX };
        }
        _is1x1 = true;
        for (size_t i = 0; i < _spatialAxisNum; ++i)
        {
            if (_kernelShape[i] != 1 && _padShape[i] != 0 && _strideShape[i] != 1)
            {
                _is1x1 = false;
                break;
            }
        }
        _srcChannels = src[0]->Axis(_param.axis);
        _dstChannels = _param.outputNum;
        assert(_dstChannels  > 0 && _dstChannels % _param.group == 0);
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
        weightShape[1] = _srcConvChannels / _param.group;
        for (size_t i = 0; i < _spatialAxisNum; ++i)
            weightShape[2 + i] = _kernelShape[i];
        Shape biasShape(_param.biasTerm, _dstChannels);
        if (this->_tensors.size() > 0) 
        {
            assert(this->_tensors.size() == _param.biasTerm + 1);
            assert(this->_tensors[0]->GetShape() == weightShape);
            if(_param.biasTerm)
                assert(this->_tensors[1]->GetShape() == biasShape);
        }
        else
        {
            if (_param.biasTerm) 
                this->_tensors.resize(2);
            else 
                this->_tensors.resize(1);
            this->_tensors[0].reset(new Tensor(weightShape));
            if (_param.biasTerm) 
                this->_tensors[1].reset(new Tensor(biasShape));
        }
        _kernelSize = this->_tensors[0]->Size(1);
        _weightOffset = _dstConvChannels * _kernelSize / _param.group;
    }

    template <class T, template<class> class A> void ConvolutionLayer<T, A>::Reshape(const ConvolutionLayer::TensorPtrs & src, const ConvolutionLayer::TensorPtrs & dst)
    {
        const Type * pSrc = src[0]->Data();
        Type * pDst = dst[0]->Data();

        size_t firstSpatialAxis = _param.axis + 1;
        _num = src[0]->Size(0, _param.axis);
        _srcShape = src[0]->GetShape();
        if(IsConv())
        {
            _dstShape.resize(_spatialAxisNum);
            for (size_t i = 0; i < _spatialAxisNum; ++i)
            {
                size_t kernelExtent = _dilationShape[i] * (_kernelShape[i] - 1) + 1;
                _dstShape[i] = (_srcShape[firstSpatialAxis + i] + 2 * _padShape[i] - kernelExtent)/ _strideShape[i] + 1;
            }
        }
        Shape dstShape(_srcShape.begin(), _srcShape.begin() + _param.axis);
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
        _dstOffset = _dstChannels * _dstConvChannels / _param.group;
        _srcConvShape.resize(_spatialAxisNum + 1);
        for (size_t i = 0; i < _spatialAxisNum + 1; ++i)
        {
            if (IsConv())
                _srcConvShape[i] = src[0]->Axis(_param.axis + i);
            else
                _srcConvShape[i] = dst[0]->Axis(_param.axis + i);
        }
        Shape colBufferShape;
        colBufferShape.push_back(_kernelSize * _param.group);
        for (int i = 0; i < _spatialAxisNum; ++i)
        {
            if (IsConv()) 
                colBufferShape.push_back(_dstShape[i]);
            else
                colBufferShape.push_back(_srcShape[i + 1]);
        }
        _colBuffer.Reshape(colBufferShape);
        _srcSize = src[0]->Size(_param.axis);
        _dstSize = dst[0]->Size(_param.axis);
        _dstSpatialSize = dst[0]->Size(firstSpatialAxis);
        if (_param.biasTerm)
        {
            Shape shape(1, _dstSpatialSize);
            _biasMultiplier.Reshape(shape, Type(1));
        }
    }

    template <class T, template<class> class A> void ConvolutionLayer<T, A>::ForwardCpu(const ConvolutionLayer::TensorPtrs & src, const ConvolutionLayer::TensorPtrs & dst)
    {
        const Type * weight = this->_tensors[0]->Data();
        for (int i = 0; i < src.size(); ++i) 
        {
            for (int n = 0; n < this->_num; ++n) 
            {
                const Type * pSrc = src[i]->Data() + _srcSize * n;
                Type * pDst = dst[i]->Data() + _dstSize * n;
                Type * pBuf = (Type*)pSrc;
                if (!_is1x1) 
                {
                    ImToCol(pSrc, pBuf);
                    pBuf = _colBuffer.Data();
                }
                for (size_t g = 0; g < _param.group; ++g) 
                {
                    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstConvChannels / _param.group, _dstConvSpatialSize, _kernelSize, 
                        Type(1.0), weight + _weightOffset * g, pBuf + _colOffset * g, Type(0.0), pDst +_dstOffset * g);
                }                
                if (_param.biasTerm) 
                {
                    const Type * bias = this->_tensors[1]->Data();
                    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstChannels, _dstSpatialSize, 1, Type(1.0), bias, _biasMultiplier.Data(), Type(1.0), pDst);
                }
            }
        }

    }

    template <class T, template<class> class A> void ConvolutionLayer<T, A>::ImToCol(const T * src, T * dst)
    {
        if (_spatialAxisNum == 2)
        {
            Synet::ImToCol(src, _srcConvShape[0], _srcConvShape[1], _srcConvShape[2], _kernelShape[0], _kernelShape[1],
                _padShape[0], _padShape[1], _strideShape[0], _strideShape[1], _dilationShape[0], _dilationShape[1], dst);
        }
    }

    SYNET_CLASS_INSTANCE(ConvolutionLayer);
}
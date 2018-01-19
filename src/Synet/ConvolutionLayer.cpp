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
        size_t firstSpatialAxis = _options.axis + 1;
        _spatialAxisNum = src[0]->Count() - firstSpatialAxis;
        if (_options.kernelX && _options.kernelY)
        {
            assert(_spatialAxisNum == 2);
            _kernelShape = { _options.kernelY, _options.kernelX };
        }
        if (_options.strideX && _options.strideY)
        {
            assert(_spatialAxisNum == 2);
            _strideShape = { _options.strideY, _options.strideX };
        }
        if (_options.padX && _options.padY)
        {
            assert(_spatialAxisNum == 2);
            _padShape = { _options.padY, _options.padX };
        }
        if (_options.dilationX && _options.dilationY)
        {
            assert(_spatialAxisNum == 2);
            _dilationShape = { _options.dilationY, _options.dilationX };
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
        _srcChannels = src[0]->Axis(_options.axis);
        _dstChannels = _options.outputNum;
        assert(_dstChannels  > 0 && _dstChannels % _options.group == 0);
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
        weightShape[1] = _srcConvChannels / _options.group;
        for (size_t i = 0; i < _spatialAxisNum; ++i)
            weightShape[2 + i] = _kernelShape[i];
        Shape biasShape(_options.biasTerm, _dstChannels);
        if (this->_tensors.size() > 0) 
        {
            assert(this->_tensors.size() == _options.biasTerm + 1);
            assert(this->_tensors[0]->GetShape() == weightShape);
            if(_options.biasTerm)
                assert(this->_tensors[1]->GetShape() == biasShape);
        }
        else
        {
            if (_options.biasTerm) 
                this->_tensors.resize(2);
            else 
                this->_tensors.resize(1);
            this->_tensors[0].reset(new Tensor(weightShape));
            if (_options.biasTerm) 
                this->_tensors[1].reset(new Tensor(biasShape));
        }
        _kernelSize = this->_tensors[0]->Size(1);
        _weightOffset = _dstConvChannels * _kernelSize / _options.group;
    }

    template <class T, template<class> class A> void ConvolutionLayer<T, A>::Reshape(const ConvolutionLayer::TensorPtrs & src, const ConvolutionLayer::TensorPtrs & dst)
    {
        const Type * pSrc = src[0]->Data();
        Type * pDst = dst[0]->Data();

        size_t firstSpatialAxis = _options.axis + 1;
        _num = src[0]->Size(0, _options.axis);
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
        Shape dstShape(_srcShape.begin(), _srcShape.begin() + _options.axis);
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
        _dstOffset = _dstChannels * _dstConvChannels / _options.group;
        _srcConvShape.resize(_spatialAxisNum + 1);
        for (size_t i = 0; i < _spatialAxisNum + 1; ++i)
        {
            if (IsConv())
                _srcConvShape[i] = src[0]->Axis(_options.axis + i);
            else
                _srcConvShape[i] = dst[0]->Axis(_options.axis + i);
        }
        Shape colBufferShape;
        colBufferShape.push_back(_kernelSize * _options.group);
        for (int i = 0; i < _spatialAxisNum; ++i)
        {
            if (IsConv()) 
                colBufferShape.push_back(_dstShape[i]);
            else
                colBufferShape.push_back(_srcShape[i + 1]);
        }
        _colBuffer.Reshape(colBufferShape);
        _srcSize = src[0]->Size(_options.axis);
        _dstSize = dst[0]->Size(_options.axis);
        //num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
        //num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
        _dstSpatialSize = dst[0]->Size(firstSpatialAxis);
        if (_options.biasTerm)
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
                    //if (!skip_im2col)
                    //    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
                    pBuf = _colBuffer.Data();
                }
                //this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight, pDst);

                if (_options.biasTerm) 
                {
                    const Type * bias = this->_tensors[1]->Data();
                    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _dstChannels, _dstSpatialSize, 1, Type(1.0), bias, _biasMultiplier.Data(), Type(1.0), pDst);
                }
            }
        }
        //for (int g = 0; g < group_; ++g) 
        //{
        //    CpuGemm<Type>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        //        group_, conv_out_spatial_dim_, kernel_dim_,
        //        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        //        (Dtype)0., output + output_offset_ * g);
        //}
    }

    SYNET_CLASS_INSTANCE(ConvolutionLayer);
}
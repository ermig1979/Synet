/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/QuantizedPoolingLayer.h"

#include "Synet/Quantization/QuantizeLinear.h"

namespace Synet
{
    QuantizedPoolingLayer::QuantizedPoolingLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t QuantizedPoolingLayer::Flop() const
    {
        return _const ? int64_t(0) : _batch * _kernelY * _kernelX * _dstC * _dstH * _dstW;
    }

    bool QuantizedPoolingLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 0 && src.size() != 1) || dst.size() != 1)
            SYNET_ERROR("QuantizedPoolingLayer supports only 1 inputs and 1 output!");

        const LayerParam& param = this->Param();
        const PoolingParam& pooling = this->Param().pooling();
        _roundingType = pooling.roundingType();
        _excludePad = pooling.excludePad();
        if (src[0]->Count() < 4)
            SYNET_ERROR("QuantizedPoolingLayer input must have at least 4 dimensions!");

        _src8u = src[0]->GetType() == TensorType8u;
        _dst8u = param.qDst().size() && param.qDst()[0].type() == TensorType8u;
        _format = src[0]->Format();

        _batch = src[0]->Axis(0);
        _srcC = _format == TensorFormatNhwc ? src[0]->Axis(-1) : src[0]->Axis(-3);
        _srcH = _format == TensorFormatNhwc ? src[0]->Axis(-3) : src[0]->Axis(-2);
        _srcW = _format == TensorFormatNhwc ? src[0]->Axis(-2) : src[0]->Axis(-1);

        if (pooling.globalPooling())
        {
            _kernelY = _srcH;
            _kernelX = _srcW;
        }
        else
        {
            const Shape& kernel = pooling.kernel();
            if (kernel.size() < 1 && kernel.size() > 3)
                SYNET_ERROR("QuantizedPoolingLayer parameter kernel size must be in range [1..2]!");
            _kernelY = kernel[0];
            _kernelX = kernel.size() > 1 ? kernel[1] : kernel[0];
            if (_kernelY <= 0 || _kernelX <= 0)
                SYNET_ERROR("QuntizedPoolingLayer parameter kernel must be positive!");
        }

        const Shape& stride = pooling.stride();
        if (stride.empty())
        {
            _strideY = 1;
            _strideX = 1;
        }
        else
        {
            if (stride.size() < 1 && stride.size() > 2)
                SYNET_ERROR("QuantizedPoolingLayer parameter stride has wrong size!");
            _strideY = stride[0];
            _strideX = stride.size() > 1 ? stride[1] : stride[0];
        }

        const Shape& pad = pooling.pad();
        if (pad.empty())
        {
            _padY = 0;
            _padX = 0;
            _padH = 0;
            _padW = 0;
        }
        else if (pad.size() == 1)
        {
            _padY = pad[0];
            _padX = pad[0];
            _padH = pad[0];
            _padW = pad[0];
        }
        else if (pad.size() == 2)
        {
            _padY = pad[0];
            _padX = pad[1];
            _padH = pad[0];
            _padW = pad[1];
        }
        else if (pad.size() == 4)
        {
            _padY = pad[0];
            _padX = pad[1];
            _padH = pad[2];
            _padW = pad[3];
        }
        else
            SYNET_ERROR("QuantizedPoolingLayer wrong pad parameter!");

        _dstC = _srcC;
        if (_roundingType == RoundingTypeCeil)
        {
            _dstH = (size_t)(::ceil((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
            _dstW = (size_t)(::ceil((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
        }
        else if (_roundingType == RoundingTypeFloor)
        {
            _dstH = (size_t)(::floor((float)(_srcH + _padY + _padH - _kernelY) / _strideY)) + 1;
            _dstW = (size_t)(::floor((float)(_srcW + _padX + _padW - _kernelX) / _strideX)) + 1;
        }
        else
            SYNET_ERROR("QuantizedPoolingLayer wrong roundingTupe parameter!");
        if (_padX || _padY)
        {
            if ((_dstH - 1) * _strideY >= _srcH + _padY)
                --_dstH;
            if ((_dstW - 1) * _strideX >= _srcW + _padX)
                --_dstW;
            if ((_dstH - 1) * _strideY >= _srcH + _padY || (_dstW - 1) * _strideX >= _srcW + _padX)
                SYNET_ERROR("QuantizedPoolingLayer check stride and pad parameters!");
        }

        bool skip = _kernelX == 1 && _kernelY == 1 &&
            _strideY == 1 && _strideX == 1 &&
            _padY == 0 && _padX == 0 && _padH == 0 && _padW == 0;

        if (skip)
        {
            _const = true;
            dst[0]->Share(*src[0]);
        }
        else
        {
            Shape shape = src[0]->Shape();
            if (_format == TensorFormatNhwc)
            {
                shape[shape.size() - 3] = _dstH;
                shape[shape.size() - 2] = _dstW;
                shape[shape.size() - 1] = _dstC;
            }
            else
            {
                shape[shape.size() - 3] = _dstC;
                shape[shape.size() - 2] = _dstH;
                shape[shape.size() - 1] = _dstW;
            }
            dst[0]->Reshape(_dst8u ? TensorType8u : TensorType32f, shape, _format);

            std::stringstream desc;
            desc << _batch << "x" << _srcC << "x" << _srcH << "x" << _srcW;
            desc << "-" << _kernelY << "x" << _kernelX << "-" << _strideY << "x" << _strideX;
            desc << (_method ? " avg" : " max") << "-" << (_src8u ? "u" : "f") << (_dst8u ? "u" : "f");
            this->UsePerfStat(desc.str(), Flop());
            _const = false;
        }

        return true;
    }

    void QuantizedPoolingLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}
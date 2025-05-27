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

        const PoolingParam& param = this->Param().pooling();

        _roundingType = param.roundingType();
        _excludePad = param.excludePad();
        if (src[0]->Count() < 4)
            SYNET_ERROR("QuantizedPoolingLayer input must have at least 4 dimensions!");

        return true;
    }

    void QuantizedPoolingLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}
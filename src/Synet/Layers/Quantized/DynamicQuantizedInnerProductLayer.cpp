/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Quantized/DynamicQuantizedInnerProductLayer.h"

#include "Synet/Utils/Gemm.h"

#include "Synet/Quantization/Gemm.h"
#include "Synet/Layers/Quantized/DynamicQuantizeLinearLayer.h"

namespace Synet
{
    DynamicQuantizedInnerProductLayer::DynamicQuantizedInnerProductLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool DynamicQuantizedInnerProductLayer::Resizable() const
    {
        return false;
    }

    size_t DynamicQuantizedInnerProductLayer::MemoryUsage() const
    {
        return Layer::MemoryUsage();
    }

    int64_t DynamicQuantizedInnerProductLayer::Flop() const
    {
        return _M * _N * (_K * 2 + (_biasTerm ? 2 : 1));
    }

    void DynamicQuantizedInnerProductLayer::CompactWeight()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        //if (_quantizedInnerProduct.Enable())
        //{
        //    for (size_t i = 0; i < this->Weight().size(); ++i)
        //        ((Tensor&)this->Weight()[i]).Clear();
        //}
#endif
    }

    bool DynamicQuantizedInnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if ((src.size() != 1) || dst.size() != 1)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 input!");

        const LayerParam& param = this->Param();
        const InnerProductParam& ip = param.innerProduct();

        _biasTerm = ip.biasTerm();
        Shape shape = src[0]->Shape();
        _K = src[0]->Size(-1);
        _M = src[0]->Size(0, -1);

        Tensors& weight = ((Tensors&)this->Weight());
        if (weight.size() < 3)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer must have at least 3 weights!");
        if (weight[0].GetType() != TensorType8i || weight[1].GetType() != TensorType8i)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only INT8 weight[0] and weight[1]!");
        if (weight[2].GetType() != TensorType32f)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer supports only FP32 weight[2]!");
        _N = weight[0].Axis(-1);
        if (weight[0].Axis(0) != _K)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer: check src[0] and weight[0] size!");
        if (weight[1].Size(0) != _N || weight[2].Size(0) != _N)
            SYNET_ERROR("DynamicQuantizedInnerProductLayer: check weight[1] and weight[2] size!");

        shape.back() = _N;
        dst[0]->Reshape(TensorType32f, shape, src[0]->Format());

        {
            Layer::Extend8u(buf, 0, src[0]->Shape(), src[0]->Format());
            Layer::Extend32i(buf, 0, shape, src[0]->Format());
            Layer::Extend32f(buf, 0, Shp(_N), src[0]->Format());
        }

        std::stringstream desc;
        desc << _M << "x" << _K << "-" << _N << " ";
        desc << (_biasTerm ? "b" : "o");
        this->UsePerfStat(desc.str(), Flop());

        return true;
    }

    void DynamicQuantizedInnerProductLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {

    }
}
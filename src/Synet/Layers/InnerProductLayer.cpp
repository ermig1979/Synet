/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/InnerProductLayer.h"

namespace Synet
{
    InnerProductLayer::InnerProductLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool InnerProductLayer::Resizable() const
    {
        return false;
    }

    int64_t InnerProductLayer::Flop() const
    {
        return _batch * _M * _N * _K * 2;
    }

    bool InnerProductLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("InnerProductLayer supports 1 or 2 inputs and 1 output!");

        const InnerProductParam& param = this->Param().innerProduct();
        _biasTerm = param.biasTerm();
        _transA = param.transposeA();
        _transB = param.transposeB();
        _axis = (int)src[0]->Index(param.axis());
        _batch = 1;
        _K = src[0]->Size(_axis);
        if (src.size() == 2)
        {
            if(_biasTerm)
                SYNET_ERROR("InnerProductLayer don't support bias for 2 inputs!");
            if(_K != src[1]->Size(_axis - 1, _axis))
                SYNET_ERROR("InnerProductLayer: src[0] shape " << ToStr(src[0]->Shape()) << " and src[1] shape " << ToStr(src[1]->Shape()) << " is not compatible!");
            _N = src[1]->Axis(_axis);
        }
        else
        {
            _N = this->Param().innerProduct().outputNum();
            const typename Base::Tensors & weight = this->Weight();
            if (weight.size() != (_biasTerm ? 2 : 1))
                SYNET_ERROR("InnerProductLayer has wrong weight number!");
            if (_transB && weight[0].Shape() != Shp(_K, _N))
            {
                SYNET_ERROR("InnerProductLayer: check weight[0] shape!");
            }
            else
            {
                if (weight[0].Shape()[1] != _K)
                {
                    _axis = (int)src[0]->Count() - 1;
                    _K = src[0]->Size(_axis);
                }
                if(weight[0].Shape() != Shp(_N, _K))
                    SYNET_ERROR("InnerProductLayer: check weight[0] shape!");
            }
            if (_biasTerm && weight[1].Shape() != Shp(_N))
                SYNET_ERROR("InnerProductLayer: check weight[1] shape!");
        }
        if (src.size() > 1)
        {
            _M = src[0]->Size(Max<size_t>(0, _axis - 1), _axis);
            _batch = _axis > 0 ? src[0]->Size(0, _axis - 1) : 1;
        }
        else
        {
            _M = src[0]->Size(0, _axis);
            _batch = 1;
        }
        std::stringstream desc;
        if (_batch > 1)
            desc << "B=" << _batch << " ";
        desc << "M=" << _M << " N=" << _N << " K=" << _K;
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }
}

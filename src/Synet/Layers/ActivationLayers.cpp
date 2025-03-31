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

#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    EluLayer::EluLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool EluLayer::Reshape(const TensorPtrs & src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _alpha = this->Param().elu().alpha();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("EluLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            if (TensorUsers(Param().src()[0]) == 1 && !src[0]->Const())
                dst[0]->Share(*src[0]);
            else
                dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("EluLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void EluLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Elu32f(src[0]->Data<float>(), _size, _alpha, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    GeluLayer::GeluLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool GeluLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("GeluLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("GeluLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void GeluLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Gelu32f(src[0]->Data<float>(), _size, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    HardSigmoidLayer::HardSigmoidLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool HardSigmoidLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        HardSigmoidParam hardSigmoid = this->Param().hardSigmoid();
        _scale = hardSigmoid.scale();
        _shift = hardSigmoid.shift();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("HardSigmoidLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("HardSigmoidLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void HardSigmoidLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        HardSigmoid32f(src[0]->Data<float>(), _size, _scale, _shift, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    HswishLayer::HswishLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool HswishLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        HswishParam hswish = this->Param().hswish();
        _shift = hswish.shift();
        _scale = hswish.scale();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("HswishLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("HswishLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void HswishLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Hswish32f(src[0]->Data<float>(), _size, _shift, _scale, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    MishLayer::MishLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool MishLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _threshold = this->Param().softplus().threshold(); // threshold to avoid FP32 overflow in exp() function.
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MishLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("MishLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void MishLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Mish32f(src[0]->Data<float>(), _size, _threshold, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    ReluLayer::ReluLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType ReluLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType16b)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    int64_t ReluLayer::Flop() const
    {
        if (_const)
            return 0;
        if (_negativeSlope != 0.0f)
            return _size * 4;
        return _size * 1;
    }

    bool ReluLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _negativeSlope = this->Param().relu().negativeSlope();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("ReluLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        _type = src[0]->GetType();
        if (_type != TensorType32f && _type != TensorType16b)
            SYNET_ERROR("ReluLayer supports only FP32 and BF16 input and output!");
        dst[0]->Reshape(_type, src[0]->Shape(), src[0]->Format());
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            if(Options().BFloat16Enable())
                UsePerfStat(Cpl::ToStr(_type));
            else
                UsePerfStat();
            _const = false;
        }
        return true;
    }

    void ReluLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        switch (_type)
        {
        case TensorType32f:
            Relu32f(src[0]->Data<float>(), _size, _negativeSlope, dst[0]->Data<float>());
            break;
        case TensorType16b:
            Relu16b(src[0]->Data<uint16_t>(), _size, _negativeSlope, dst[0]->Data<uint16_t>());
            break;
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    RestrictRangeLayer::RestrictRangeLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t RestrictRangeLayer::Flop() const
    {
        if (_const)
            return 0;
        return _size * 2;
    }

    bool RestrictRangeLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const RestrictRangeParam& param = this->Param().restrictRange();
        _lower = param.lower();
        _upper = param.upper();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("RestrictRangeLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        size_t srcUsers = TensorUsers(Param().src()[0]);
        if (src[0]->GetType() == TensorType32f)
        {
            if (dst[0] != src[0])
            {
                if(srcUsers == 1 && !src[0]->Const())
                    dst[0]->Share(*src[0]);
                else
                    dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
            }
        }
        else
            SYNET_ERROR("RestrictRangeLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void RestrictRangeLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        RestrictRange32f(src[0]->Data<float>(), _size, _lower, _upper, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    SigmoidLayer::SigmoidLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool SigmoidLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SigmoidLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("SigmoidLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void SigmoidLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Sigmoid32f(src[0]->Data<float>(), _size, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    SoftplusLayer::SoftplusLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool SoftplusLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const SoftplusParam& param = this->Param().softplus();
        _beta = param.beta();
        _threshold = param.threshold();
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SoftplusLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("SoftplusLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void SoftplusLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Softplus32f(src[0]->Data<float>(), _size, _beta, _threshold, dst[0]->Data<float>());
    }

    //-------------------------------------------------------------------------------------------------

    SwishLayer::SwishLayer(const LayerParam& param, Context* context)
        : Layer(param, context)
    {
    }

    bool SwishLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("SwishLayer supports only 1 input and 1 output!");
        _size = src[0]->Size();
        if (src[0]->GetType() == TensorType32f)
        {
            dst[0]->Reshape(src[0]->GetType(), src[0]->Shape(), src[0]->Format());
        }
        else
            SYNET_ERROR("SwishLayer supports only FP32 input and output!");
        if (src[0]->Const())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void SwishLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        Swish32f(src[0]->Data<float>(), _size, dst[0]->Data<float>());
    }
}
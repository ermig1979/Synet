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

#include "Synet/Layers/Math/ScaleLayer.h"
#include "Synet/Layers/Math/PowerLayer.h"

namespace Synet
{
    void Power32f(const float* src, size_t size, float scale, float shift, float power, float* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = ::powf(src[i] * scale + shift, power);
    }

    void Scale32i(const int32_t* src, size_t size, int32_t scale, int32_t shift, int32_t* dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = src[i] * scale + shift;
    }

    //-------------------------------------------------------------------------------------------------

    PowerLayer::PowerLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType PowerLayer::LowPrecision(TensorType type) const
    {
        const LayerParam& p = this->Param();
        if (type == TensorType16b && Options().BFloat16Enable() && p.power().power() == 1.0f)
            return /*p.src()[0] != p.dst()[0] ? LowPrecisionTypeActive :*/ LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    bool PowerLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("Power supports only 1 input and 1 output!");

        const PowerParam & param = Param().power();
        _power = param.power();
        _scale = param.scale();
        _shift = param.shift();
        bool disable16b = src[0]->GetType() == TensorType32i || src[0]->Const();
        _src16b = src[0]->GetType() == TensorType16b;
        _dst16b = dst[0]->GetType() == TensorType16b;
        _size = src[0]->Size();

        if(src[0]->GetType() != TensorType32f && src[0]->GetType() != TensorType32i && src[0]->GetType() != TensorType16b)
            SYNET_ERROR("PowerLayer unsupported src[0]: " << Cpl::ToStr(src[0]->GetType()) << " type !");
        if(src[0]->GetType() != TensorType32f && _power != 1.0f)
            SYNET_ERROR("PowerLayer parameter 'power' must be 1.0 for non FP32 tensors!");

        if (!disable16b && (_src16b || _dst16b))
        {
            _scale16b.Init(1, _size, src[0]->GetType(), dst[0]->GetType(), TensorFormatNchw, true, true);
            if(!_scale16b.Enable())
                SYNET_ERROR("PowerLayer can't initialize Scale16b engine!");
        }

        if (src[0] != dst[0])
        {
            const Shape& shape = src[0]->Shape();
            TensorFormat format = src[0]->Format();
            const Strings& names = Param().src();
            if (TensorUsers(names[0]) == 1 && !src[0]->Const() && _src16b == _dst16b)
                dst[0]->ShareAs(*src[0], shape, format);
            else
            {
                if (disable16b)
                    dst[0]->Reshape(src[0]->GetType(), shape, format);
                else if (_dst16b)
                    dst[0]->Reshape(TensorType16b, shape, format);
                else
                    dst[0]->Reshape(TensorType32f, shape, format);
            }
        }
        if (src[0]->Const())
        {
            Forward(src, buf, dst, 0);
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

    int64_t PowerLayer::Flop() const
    {
        if (_const)
            return 0;
        if (_power == 1.0f)
            return _size * 2;
        else
            return _size * 41;
    }

    void PowerLayer::Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
        if (_scale16b.Enable())
        {
            _scale16b.Forward(src[0]->RawData(), &_scale, &_shift, dst[0]->RawData());
        }
        else if (src[0]->GetType() == TensorType32f)
        {
            const float* pSrc = src[0]->Data<float>();
            float* pDst = dst[0]->Data<float>();
            if (_power == 1.0f)
                ScaleForward32f(pSrc, &_scale, &_shift, 1, 1, _size, pDst, TensorFormatNchw, 0);
            else
                Power32f(pSrc, _size, _scale, _shift, _power, pDst);
        } 
        else if (src[0]->GetType() == TensorType32i)
        {
           Scale32i(src[0]->Data<int32_t>(), _size, (int32_t)_scale, (int32_t)_shift, dst[0]->Data<int32_t>());
        }
    }
}
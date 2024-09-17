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

#include "Synet/Layers/UnpackLayer.h"

namespace Synet
{
    UnpackLayer::UnpackLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    LowPrecisionType UnpackLayer::LowPrecision(TensorType type) const
    {
        if (type == TensorType8u)
            return LowPrecisionTypePassive;
        if (type == TensorType16b)
            return LowPrecisionTypePassive;
        return LowPrecisionTypeNone;
    }

    bool UnpackLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const UnpackParam & param = this->Param().unpack();
        _srcType = src[0]->GetType();
        if(_srcType != TensorType32f && _srcType != TensorType8u && _srcType != TensorType16b)
            SYNET_ERROR("Unsupported src type in UnpackLayer!");
        _axis = src[0]->Index(param.axis());
        _size = src[0]->Axis(_axis);
        _count = dst.size();
        _begins.resize(_count);
        _sizes.resize(_count);
        if (param.parts().empty())
        {
            size_t step = _size / _count;
            for (size_t i = 0; i < _count; ++i)
            {
                _begins[i] = i * step;
                _sizes[i] = step;
            }
        }
        else
        {
            if(param.parts().size() != _count)
                SYNET_ERROR("Check parameter unpack.parts in UnpackLayer!");
            for (size_t i = 0; i < _count; ++i)
            {
                _begins[i] = i ? _begins[i - 1] + param.parts()[i - 1] : 0;
                _sizes[i] = param.parts()[i];
            }
        }
        if (_begins.back() + _sizes.back() != _size)
            SYNET_ERROR("Incompartible output sizes in UnpackLayer!");
        _outer = src[0]->Size(0, _axis);
        _inner = src[0]->Size(_axis + 1);
        if (dst.size() > 1)
        {
            Shape shape = src[0]->Shape();
            for (size_t i = 0; i < _count; ++i)
            {
                shape[_axis] = _sizes[i];
                dst[i]->Reshape(_srcType, shape, src[0]->Format());
            }
            if (src[0]->Const())
            {
                ForwardCpu(src, buf, dst);
                for (size_t i = 0; i < _count; ++i)
                    dst[i]->SetConst(true);
                _const = true;
            }
            else
            {
                _const = false;
                if (Options().BFloat16Enable())
                    UsePerfStat(Cpl::ToStr(_srcType));
                else
                    UsePerfStat();
            }
        }
        else
        {
            dst[0]->Share(*src[0]);
            _const = true;
        }
        return true;
    }

    template <class T> void UnpackLayer::Unpack(const T* src, std::vector<T*> dst)
    {
        for (size_t o = 0; o < _outer; ++o)
        {
            for (size_t c = 0; c < _count; c += 1)
            {
                const T* pSrc = src + (o * _size + _begins[c]) * _inner;
                T* pDst = dst[c] + o * _sizes[c] * _inner;
                CpuCopy(pSrc, _sizes[c] * _inner, pDst);
            }
        }
    }

    void UnpackLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_srcType)
        {
        case TensorType32f:
        {
            std::vector<float*> pDst(dst.size());
            for (size_t i = 0; i < dst.size(); ++i)
                pDst[i] = dst[i]->Data<float>();
            Unpack(src[0]->Data<float>(), pDst);
            break;
        }
        case TensorType8u:
        {
            std::vector<uint8_t*> pDst(dst.size());
            for (size_t i = 0; i < dst.size(); ++i)
                pDst[i] = dst[i]->Data<uint8_t>();
            Unpack(src[0]->Data<uint8_t>(), pDst);
            break;
        }
        case TensorType16b:
        {
            std::vector<uint16_t*> pDst(dst.size());
            for (size_t i = 0; i < dst.size(); ++i)
                pDst[i] = dst[i]->Data<uint16_t>();
            Unpack(src[0]->Data<uint16_t>(), pDst);
            break;
        }
        default:
            assert(0);
        }
    }
}
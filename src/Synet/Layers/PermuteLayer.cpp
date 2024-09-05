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

#include "Synet/Layers/PermuteLayer.h"
#include "Synet/Utils/Permute.h"

namespace Synet
{
    static Shape Stride(const Shape& shape, const Shape& order)
    {
        Shape buf(shape.size(), 1), out(shape.size(), 1);
        for (ptrdiff_t i = shape.size() - 2; i >= 0; i--)
            buf[i] = buf[i + 1] * shape[i + 1];
        for (size_t i = 0; i < shape.size(); ++i)
            out[order[i]] = buf[i];
        return out;
    }

    static Shape CompactOrder(Shape order)
    {
        for (size_t i = 1; i < order.size();)
        {
            if (order[i] == order[i - 1] + 1)
            {
                order.erase(order.begin() + i);
                for (size_t j = 0; j < order.size(); ++j)
                    if (order[j] > order[i - 1])
                        order[j]--;
            }
            else
                ++i;
        }
        return order;
    }

    static Shape CompactShape(const Shape& shape, const Shape& order)
    {
        Shape compact;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            size_t dim = shape[order[i]];
            if (i && order[i] == order[i - 1] + 1)
                compact.back() *= dim;
            else
                compact.push_back(dim);
        }
        return compact;
    }

    //-----------------------------------------------------------------------------------------------------

    template <class T> void Permute(const uint8_t* src8, const Shape& shape, const Shape& stride, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        T* dst = (T*)dst8;
        switch (shape.size())
        {
        case 2:
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    *dst++ = src[i * stride[0] + j * stride[1]];
                }
            }
            break;
        case 3:
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2]];
                    }
                }
            }
            break;
        case 4:
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        for (size_t l = 0; l < shape[3]; ++l)
                        {
                            *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3]];
                        }
                    }
                }
            }
            break;
        case 5:
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    for (size_t k = 0; k < shape[2]; ++k)
                    {
                        for (size_t l = 0; l < shape[3]; ++l)
                        {
                            for (size_t m = 0; m < shape[4]; ++m)
                            {
                                *dst++ = src[i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3] + m * stride[4]];
                            }
                        }
                    }
                }
            }
            break;
        default:
            assert(0);
        }
    }

    PermuteLayer::PermutePtr GetPermute(TensorType type)
    {
        switch (type)
        {
        case TensorTypeBool:
        case TensorType8i:
        case TensorType8u: return Permute<uint8_t>;
        case TensorType16b:
        case TensorType16f: return Permute<uint16_t>;
        case TensorType32f:
        case TensorType32i: return Permute<uint32_t>;
        case TensorType64i:
        case TensorType64u: return Permute<uint64_t>;
        default: return NULL;
        }
    }

    //-----------------------------------------------------------------------------------------------------

    PermuteLayer::PermuteLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        _simdPermute = std::make_shared<SimdPermute>();
    }

    bool PermuteLayer::Can16b() const
    {
        return Options().BFloat16Enable();
    }

    bool PermuteLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("PermuteLayer supports only 1 input and 1 output!");
        if (src[0]->GetType() != dst[0]->GetType() && src[0]->GetType() != TensorType64i)
            SYNET_ERROR("PermuteLayer input and output must have the same type!");

        const PermuteParam & param = this->Param().permute();
        _dstOrder = param.order();
        _count = _dstOrder.size();
        _srcOrder.resize(_count);
        size_t is = 0, os = 0;
        Shape permute;
        for (size_t i = 0; i < _dstOrder.size(); ++i)
        {
            if (_dstOrder[i] != i)
                permute.push_back(i);
            is += i;
            os += _dstOrder[i];
            _srcOrder[_dstOrder[i]] = i;
        }
        if (is != os)
            SYNET_ERROR("PermuteLayer has wrong parameter permute.order!");
        bool nontrivial = permute.size() > 1;
        if (permute.size() == 2 && permute[0] + 1 == permute[1])
        {
            if (src[0]->Axis(permute[0]) == 1 || src[0]->Axis(permute[1]) == 1)
                nontrivial = false;
        }
        _srcShape = src[0]->Shape();
        if(_srcShape.size() != _count)
            SYNET_ERROR("PermuteLayer parameter permute.order incompatible with input shape!");
        _dstShape.clear();
        for (size_t i = 0; i < _count; ++i)
            _dstShape.push_back(_srcShape[_dstOrder[i]]);
        if (param.skip())
            nontrivial = false;
        if (nontrivial)
        {
            dst[0]->Reshape(src[0]->GetType(), _dstShape, param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
            _simdPermute->Init(src[0]->Shape(), param.order(), src[0]->GetType());
            if (!_simdPermute->Enable())
            {
                CompactShapes();
                if (_count < 2 || _count > 5)
                    SYNET_ERROR("PermuteLayer does not support " << _count << "D reordering!");
                _srcStride = Stride(_srcShape, _srcOrder);
                _dstStride = Stride(_dstShape, _dstOrder);
                _permute = GetPermute(src[0]->GetType());
                if(_permute == NULL)
                    SYNET_ERROR("PermuteLayer can't set permute worker!");
            }
            this->UsePerfStat(Cpl::ToStr(src[0]->GetType()));
            _const = false;
        }
        else
        {
            dst[0]->ShareAs(*src[0], _dstShape, param.format() == TensorFormatUnknown ? src[0]->Format() : param.format());
            _const = true;
        }
        return true;
    }

    void PermuteLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (_simdPermute->Enable())
            _simdPermute->Forward(src[0]->RawData(), dst[0]->RawData());
        else
            _permute(src[0]->RawData(), _dstShape, _srcStride, dst[0]->RawData());
    }

    void PermuteLayer::CompactShapes()
    {
        size_t count = 0;
        for (size_t i = 0; i < _count; ++i)
            if (i == 0 || _dstOrder[i] != _dstOrder[i - 1] + 1)
                count++;
        if (count == _count)
            return;
        _count = count;
        Shape dstShape = _dstShape;
        _dstShape = CompactShape(_srcShape, _dstOrder);
        _srcShape = CompactShape(dstShape, _srcOrder);
        _dstOrder = CompactOrder(_dstOrder);
        _srcOrder = CompactOrder(_srcOrder);
    }
}
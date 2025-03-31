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

#include "Synet/Layers/PadLayer.h"

namespace Synet
{
    template<class T> void PadConstant4(const Tensor<T> & src, const Shape& padB, const Shape& padE, Tensor<T> & dst)
    {
        assert(src.Count() == 4 && padB.size() == 4 && padE.size() == 4 && dst.Count() == 4);
        size_t size = src.Axis(3) * sizeof(T);
        for (size_t i0 = 0; i0 < src.Axis(0); ++i0)
        {
            for (size_t i1 = 0; i1 < src.Axis(1); ++i1)
            {
                for (size_t i2 = 0; i2 < src.Axis(2); ++i2)
                {
                    const T* pSrc = src.template Data<T>(Shp(i0, i1, i2, 0));
                    T* pDst = dst.template Data<T>(Shp(i0 + padB[0], i1 + padB[1], i2 + padB[2], padB[3]));
                    memcpy(pDst, pSrc, size);
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    PadLayer::PadLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool PadLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if ((src.size() != 1 && src.size() != 2) || dst.size() != 1)
            SYNET_ERROR("PadLayer supports only 1-2 inputs and 1 output!");

        const PadParam& pad = this->Param().pad();

        _dims = src[0]->Count();
        _padB.resize(_dims);
        _padE.resize(_dims);
        if (src.size() == 1)
        {
            if (src[0]->Count() * 2 != pad.pads().size())
                SYNET_ERROR("PadLayer parameter pad().pads() has wrong size: " << pad.pads().size() << " instead of " << src[0]->Count() * 2 << " !");
            const int64_t* raw = pad.pads().data();
            for (size_t i = 0; i < _dims; ++i)
            {
                _padB[i] = (size_t)raw[0 + i];
                _padE[i] = (size_t)raw[_dims + i];
            }
            if (src[0]->Format() == TensorFormatNhwc)
            {
                if (_dims == 4)
                {
                    _padB = Shp(_padB[0], _padB[2], _padB[3], _padB[1]);
                    _padE = Shp(_padB[0], _padE[2], _padE[3], _padE[1]);
                }
                else
                    SYNET_ERROR("PadLayer can process only 4D NHWC tensor!");
            }
        }
        else if (src[1]->GetType() == TensorType64i)
        {
            if (src[0]->Count() * 2 != src[1]->Size())
                SYNET_ERROR("PadLayer src[1] has wrong size: " << src[1]->Size() << " instead of " << src[0]->Count() * 2 << " !");
            const int64_t * raw = src[1]->Data<int64_t>();
            for (size_t i = 0; i < _dims; ++i)
            {
                _padB[i] = (size_t)raw[0 + i];
                _padE[i] = (size_t)raw[_dims + i];
            }
            if (src[0]->Format() == TensorFormatNhwc)
            {
                if (_dims == 4)
                {
                    _padB = Shp(_padB[0], _padB[2], _padB[3], _padB[1]);
                    _padE = Shp(_padB[0], _padE[2], _padE[3], _padE[1]);
                }
                else
                    SYNET_ERROR("PadLayer can process only 4D NHWC tensor!");
            }
        }
        else
            SYNET_ERROR("PadLayer does not support: " << Cpl::ToStr(_type) << " src[1] type!");
        Shape dstShape = src[0]->Shape();
        for (size_t i = 0; i < _dims; ++i)
            dstShape[i] += _padB[i] + _padE[i];
        _mode = pad.mode();
        if(_mode != PadModeConstant)
            SYNET_ERROR("PadLayer supports only: " << Cpl::ToStr(_mode) << " mode!");
        if (_dims != 4)
            SYNET_ERROR("PadLayer can process only 4D tensor!");
        if (dstShape == src[0]->Shape())
        {
            dst[0]->Share(*src[0]);
            _const = true;
        }
        else
        {
            _type = src[0]->GetType();
            switch (_type)
            {
            case TensorType32f:
                dst[0]->Reshape(_type, dstShape, src[0]->Format(), 0.0f);
                break;
            default:
                SYNET_ERROR("PadLayer does not support: " << Cpl::ToStr(_type) << " src[0] type!");
            }
            _const = false;
            this->UsePerfStat();
        }

        return true;
    }

    void PadLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        switch (_dims)
        {
        case 4:
            PadConstant4(*src[0], _padB, _padE, *dst[0]);
            break;
        default:
            assert(0);
        }
    }
}
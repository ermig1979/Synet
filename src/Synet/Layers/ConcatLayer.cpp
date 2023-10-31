/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layers/ConcatLayer.h"

#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T, size_t N> void Concat2N(const T * src0, const T * src1, size_t num, T * dst)
    {
        struct H { T a[N]; };
        for (size_t n = 0; n < num; ++n)
        {
            ((H*)dst)[0] = *(H*)src0;
            ((H*)dst)[1] = *(H*)src1;
            src0 += N;
            src1 += N;
            dst += 2*N;
        }
    }

    //-------------------------------------------------------------------------------------------------

    ConcatLayer::ConcatLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool ConcatLayer::Can8i() const
    {
        return this->Param().concat().can8i();
    }

    bool ConcatLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _concatAxis = src[0]->Index((int32_t)this->Param().concat().axis());
        _concatNum = src[0]->Size(0, _concatAxis);
        _concatInputSize = src[0]->Size(_concatAxis + 1);
        size_t srcSizeSum = src[0]->Size();
        Shape dstShape = src[0]->Shape();
        _srcConcatAxis.resize(src.size());
        _srcConcatAxis[0] = src[0]->Axis(_concatAxis);
        bool allConst = src[0]->Const();
        for (size_t i = 1; i < src.size(); ++i)
        {
            if (src[0]->GetType() != src[i]->GetType())
                SYNET_ERROR("Incompartible input types: src[0] " << Cpl::ToStr(src[0]->GetType()) << " and src[" << i << "] " << Cpl::ToStr(src[i]->GetType()) << " !");
            if(src[0]->Count() != src[i]->Count())
                SYNET_ERROR("Incompartible input shapes: src[0] " << Detail::DebugPrint(src[0]->Shape()) << " and src[" << i << "] " << Detail::DebugPrint(src[i]->Shape()) << " !");
            assert(src[0]->Count() == src[i]->Count());
            for (size_t j = 0; j < src[0]->Count(); ++j)
            {
                if (j == _concatAxis)
                    continue;
                if(dstShape[j] != src[i]->Axis(j))
                    SYNET_ERROR("Incompartible input shapes: src[0] " << Detail::DebugPrint(src[0]->Shape()) << " and src[" << i << "] " << Detail::DebugPrint(src[i]->Shape()) << " for axis " << _concatAxis << " !");
            }
            srcSizeSum += src[i]->Size();
            _srcConcatAxis[i] = src[i]->Axis(_concatAxis);
            dstShape[_concatAxis] += _srcConcatAxis[i];
            allConst = allConst && src[i]->Const();
        }
        if (src.size() == 1)
            dst[0]->Share(*src[0]);
        else
        {
            _srcType = src[0]->GetType();
            if(_srcType != TensorType32f && _srcType != TensorType8u && _srcType != TensorType8i)
                SYNET_ERROR("Unsupported input type: " << Cpl::ToStr(_srcType) << " !");
            dst[0]->Reshape(_srcType, dstShape, src[0]->Format());
            assert(srcSizeSum == dst[0]->Size());
        }
        _dstConcatAxis = dst[0]->Axis(_concatAxis);
        _special2N = (src.size() == 2 && _srcConcatAxis[0] == _srcConcatAxis[1]) ? _srcConcatAxis[0] : 0;
        if (allConst || this->Param().concat().fixed())
        {
            ForwardCpu(src, buf, dst);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            //ForwardCpu(src, buf, dst);
            this->UsePerfStat();
            _const = false;
        }
        return true;
    }

    void ConcatLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        if (src.size() == 1)
            return;
        switch (_srcType)
        {
        case TensorType32f:
        {
            std::vector<float*> pSrc(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                pSrc[i] = src[i]->Data<float>();
            Concat(pSrc, dst[0]->Data<float>());
            break;
        }
        case TensorType8u:
        {
            std::vector<uint8_t*> pSrc(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                pSrc[i] = src[i]->Data<uint8_t>();
            Concat(pSrc, dst[0]->Data<uint8_t>());
            break;
        }
        case TensorType8i:
        {
            std::vector<int8_t*> pSrc(src.size());
            for (size_t i = 0; i < src.size(); ++i)
                pSrc[i] = src[i]->Data<int8_t>();
            Concat(pSrc, dst[0]->Data<int8_t>());
            break;
        }
        default:
            assert(0);
        }
    }

    template <class T> void ConcatLayer::Concat(std::vector<T*> src, T * dst)
    {
        if (_concatInputSize == 1)
        {
            if (_special2N == 16)
                Concat2N<T, 16>(src[0], src[1], _concatNum, dst);
            else if (_special2N == 32)
                Concat2N<T, 32>(src[0], src[1], _concatNum, dst);
            else if (_special2N == 64)
                Concat2N<T, 64>(src[0], src[1], _concatNum, dst);
            else if (_special2N == 128)
                Concat2N<T, 128>(src[0], src[1], _concatNum, dst);
            else
            {
                for (size_t n = 0; n < _concatNum; ++n)
                {
                    for (size_t i = 0; i < src.size(); ++i)
                    {
                        size_t size = _srcConcatAxis[i];
                        CpuCopy(src[i], size, dst);
                        src[i] += size;
                        dst += size;
                    }
                }
            }
        }
        else
        {
            size_t concatAxisOffset = 0;
            for (size_t i = 0; i < src.size(); ++i)
            {
                for (size_t n = 0; n < _concatNum; ++n)
                    CpuCopy(src[i] + n * _srcConcatAxis[i] * _concatInputSize, _srcConcatAxis[i] * _concatInputSize,
                        dst + (n * _dstConcatAxis + concatAxisOffset) * _concatInputSize);
                concatAxisOffset += _srcConcatAxis[i];
            }
        }
    }
}

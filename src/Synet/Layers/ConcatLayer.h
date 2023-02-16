/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
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
    }

    template <class T> class ConcatLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ConcatLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Can8i() const
        {
            return this->Param().concat().can8i();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _concatAxis = this->Param().concat().axis();
            _fixed = this->Param().concat().fixed();
            _concatNum = src[0]->Size(0, _concatAxis);
            _concatInputSize = src[0]->Size(_concatAxis + 1);
            size_t srcSizeSum = src[0]->Size();
            Shape dstShape = src[0]->Shape();
            _srcConcatAxis.resize(src.size());
            _srcConcatAxis[0] = src[0]->Axis(_concatAxis);
            for (size_t i = 1; i < src.size(); ++i)
            {
                assert(src[0]->Count() == src[i]->Count());
                for (size_t j = 0; j < src[0]->Count(); ++j)
                {
                    if (j == _concatAxis)
                        continue;
                    assert(dstShape[j] == src[i]->Axis(j));
                }
                srcSizeSum += src[i]->Size();
                _srcConcatAxis[i] = src[i]->Axis(_concatAxis);
                dstShape[_concatAxis] += _srcConcatAxis[i];
            }
            if (src.size() == 1)
                dst[0]->Share(*src[0]);
            else
            {
                _type = src[0]->GetType();
                switch (_type)
                {
                case TensorType32f: dst[0]->As32f().Reshape(dstShape, src[0]->Format()); break;
                case TensorType8u: dst[0]->As8u().Reshape(dstShape, src[0]->Format()); break;
                case TensorType8i: dst[0]->As8i().Reshape(dstShape, src[0]->Format()); break;
                default:
                    assert(0);
                }
                assert(srcSizeSum == dst[0]->Size());
            }
            _dstConcatAxis = dst[0]->Axis(_concatAxis);
            _special2N = (src.size() == 2 && _srcConcatAxis[0] == _srcConcatAxis[1]) ? _srcConcatAxis[0] : 0;
            if (src.size() != 1)
                ForwardCpu(src, dst);
#if 0
            std::stringstream desc;
            desc << _concatNum << "x";
            for (size_t i = 0; i < src.size(); ++i)
                desc << (i ? "-" : "") << _srcConcatAxis[i];
            desc << "x" << _concatInputSize;
            this->UsePerfStat(desc.str(), dst[0]->Size());
#else
            this->UsePerfStat();
#endif
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (src.size() == 1 || _fixed)
                return;
            ForwardCpu(src, dst);
        }

        void ForwardCpu(const TensorPtrs& src, const TensorPtrs& dst)
        {
            switch (_type)
            {
            case TensorType32f:
            {
                std::vector<float*> pSrc(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    pSrc[i] = src[i]->As32f().CpuData();
                ForwardCpu(pSrc, dst[0]->As32f().CpuData());
                break;
            }
            case TensorType8u:
            {
                std::vector<uint8_t*> pSrc(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    pSrc[i] = src[i]->As8u().CpuData();
                ForwardCpu(pSrc, dst[0]->As8u().CpuData());
                break;
            }
            case TensorType8i:
            {
                std::vector<int8_t*> pSrc(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    pSrc[i] = src[i]->As8i().CpuData();
                ForwardCpu(pSrc, dst[0]->As8i().CpuData());
                break;
            }
            default:
                assert(0);
            }
        }

        template <class TT> void ForwardCpu(std::vector<TT*> src, TT * dst)
        {
            if (_concatInputSize == 1)
            {
                if (_special2N == 16)
                    Detail::Concat2N<TT, 16>(src[0], src[1], _concatNum, dst);
                else if (_special2N == 32)
                    Detail::Concat2N<TT, 32>(src[0], src[1], _concatNum, dst);
                else if (_special2N == 64)
                    Detail::Concat2N<TT, 64>(src[0], src[1], _concatNum, dst);
                else if (_special2N == 128)
                    Detail::Concat2N<TT, 128>(src[0], src[1], _concatNum, dst);
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

    private:
        typedef std::vector<Type*> Ptrs;
        size_t _concatNum, _concatInputSize, _concatAxis, _dstConcatAxis, _special2N;
        Index _srcConcatAxis;
        TensorType _type;
        bool _fixed;
    };
}

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

#pragma once

#include "Synet/Common.h"
#include "Synet/Layer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    template <class T> class StridedSliceLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        StridedSliceLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            StridedSliceParam param = this->Param().stridedSlice();

            _srcDims = src[0]->Shape();
            if (src.size() == 4 && param.beginDims().empty() && param.endDims().empty() && param.axes().empty())
            {
                param.beginDims() = Shp(src[1]->As64i().CpuData(), src[1]->Size());
                param.endDims() = Shp(src[2]->As64i().CpuData(), src[2]->Size());
                param.axes() = Shp(src[3]->As64i().CpuData(), src[3]->Size());
            }
            _axes = param.axes();
            if (param.beginDims().size())
            {
                if (_axes.size())
                {
                    assert(param.beginDims().size() == _axes.size());
                    _beginDims.clear();
                    for (size_t s = 0, b = 0; s < _srcDims.size(); ++s)
                    {
                        bool found = false;
                        for (size_t a = 0; a < _axes.size(); ++a)
                            if (s == _axes[a])
                                found = true;
                        _beginDims.push_back(found ? param.beginDims()[b++] : 0);
                    }
                }
                else
                    _beginDims = param.beginDims();
                assert(_beginDims.size() == _srcDims.size());
            }
            else
                _beginDims.resize(_srcDims.size(), 0);
            if (param.endDims().size())
            {
                if (_axes.size())
                {
                    assert(param.endDims().size() == _axes.size());
                    _endDims.clear();
                    for (size_t s = 0, e = 0; s < _srcDims.size(); ++s)
                    {
                        bool found = false;
                        for (size_t a = 0; a < _axes.size(); ++a)
                            if (s == _axes[a])
                                found = true;
                        _endDims.push_back(found ?  Min<size_t>(param.endDims()[e++], _srcDims[s]) : _srcDims[s]);
                    }
                }
                else
                    _endDims = param.endDims();
                assert(_endDims.size() == _srcDims.size());
            }
            else
                _endDims = _srcDims;
            if (param.strideDims().size())
            {
                if (_axes.size())
                {
                    assert(param.strideDims().size() == _axes.size());
                    _strideDims.clear();
                    for (size_t s = 0, sd = 0; s < _srcDims.size(); ++s)
                    {
                        bool found = false;
                        for (size_t a = 0; a < _axes.size(); ++a)
                            if (s == _axes[a])
                                found = true;
                        _strideDims.push_back(found ? param.strideDims()[sd++] : 1);
                    }
                }
                else
                    _strideDims = param.strideDims();
                assert(_beginDims.size() == _strideDims.size());
            }
            else
                _strideDims.resize(_srcDims.size(), 1);

            _dstDims.resize(_srcDims.size());
            for (size_t i = 0; i < _srcDims.size(); ++i)
            {
                size_t count = 0;
                if (_strideDims[i] > 0)
                {
                    size_t begin = _beginDims[i];
                    size_t end = _endDims[i] == 0 ? _srcDims[i] : Min(_endDims[i], _srcDims[i]);
                    for (; begin < end; begin += _strideDims[i])
                        count++;
                }
                else
                    assert(0);
                _dstDims[i] = count;
            }
            dst[0]->Reshape(_dstDims, src[0]->Format());

            _srcStrides.resize(_srcDims.size(), 1);
            for (size_t i = 0; i < _srcDims.size(); ++i)
                _srcStrides[i] = src[0]->Size(i);
            _dstStrides.resize(_dstDims.size(), 1);
            for (size_t i = 0; i < _dstDims.size(); ++i)
                _dstStrides[i] = dst[0]->Size(i);
            this->UsePerfStat();
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pSrc0 = src[0]->CpuData();
            Type * pDst0 = dst[0]->CpuData();
            switch (_srcDims.size())
            {
            case 1:
                for (size_t s0 = _beginDims[0], d0 = 0; d0 < _dstDims[0]; d0 += 1, s0 += _strideDims[0])
                    pDst0[d0] = pSrc0[s0];
                break;
            case 2:
                for (size_t s0 = _beginDims[0], d0 = 0; d0 < _dstDims[0]; d0 += 1, s0 += _strideDims[0])
                {
                    const Type * pSrc1 = pSrc0 + s0 * _srcStrides[1];
                    Type * pDst1 = pDst0 + d0 * _dstStrides[1];
                    for (size_t s1 = _beginDims[1], d1 = 0; d1 < _dstDims[1]; d1 += 1, s1 += _strideDims[1])
                        pDst1[d1] = pSrc1[s1];
                }
                break;
            case 3:
                for (size_t s0 = _beginDims[0], d0 = 0; d0 < _dstDims[0]; d0 += 1, s0 += _strideDims[0])
                {
                    const Type * pSrc1 = pSrc0 + s0 * _srcStrides[1];
                    Type * pDst1 = pDst0 + d0 * _dstStrides[1];
                    for (size_t s1 = _beginDims[1], d1 = 0; d1 < _dstDims[1]; d1 += 1, s1 += _strideDims[1])
                    {
                        const Type * pSrc2 = pSrc1 + s1 * _srcStrides[2];
                        Type * pDst2 = pDst1 + d1 * _dstStrides[2];
                        for (size_t s2 = _beginDims[2], d2 = 0; d2 < _dstDims[2]; d2 += 1, s2 += _strideDims[2])
                            pDst2[d2] = pSrc2[s2];
                    }
                }
                break;
            case 4:
                for (size_t s0 = _beginDims[0], d0 = 0; d0 < _dstDims[0]; d0 += 1, s0 += _strideDims[0])
                {
                    const Type * pSrc1 = pSrc0 + s0 * _srcStrides[1];
                    Type * pDst1 = pDst0 + d0 * _dstStrides[1];
                    for (size_t s1 = _beginDims[1], d1 = 0; d1 < _dstDims[1]; d1 += 1, s1 += _strideDims[1])
                    {
                        const Type * pSrc2 = pSrc1 + s1 * _srcStrides[2];
                        Type * pDst2 = pDst1 + d1 * _dstStrides[2];
                        for (size_t s2 = _beginDims[2], d2 = 0; d2 < _dstDims[2]; d2 += 1, s2 += _strideDims[2])
                        {
                            const Type * pSrc3 = pSrc2 + s2 * _srcStrides[3];
                            Type * pDst3 = pDst2 + d2 * _dstStrides[3];
                            for (size_t s3 = _beginDims[3], d3 = 0; d3 < _dstDims[3]; d3 += 1, s3 += _strideDims[3])
                                pDst3[d3] = pSrc3[s3];
                        }
                    }
                }
                break;
            case 5:
                for (size_t s0 = _beginDims[0], d0 = 0; d0 < _dstDims[0]; d0 += 1, s0 += _strideDims[0])
                {
                    const Type * pSrc1 = pSrc0 + s0 * _srcStrides[1];
                    Type * pDst1 = pDst0 + d0 * _dstStrides[1];
                    for (size_t s1 = _beginDims[1], d1 = 0; d1 < _dstDims[1]; d1 += 1, s1 += _strideDims[1])
                    {
                        const Type * pSrc2 = pSrc1 + s1 * _srcStrides[2];
                        Type * pDst2 = pDst1 + d1 * _dstStrides[2];
                        for (size_t s2 = _beginDims[2], d2 = 0; d2 < _dstDims[2]; d2 += 1, s2 += _strideDims[2])
                        {
                            const Type * pSrc3 = pSrc2 + s2 * _srcStrides[3];
                            Type * pDst3 = pDst2 + d2 * _dstStrides[3];
                            for (size_t s3 = _beginDims[3], d3 = 0; d3 < _dstDims[3]; d3 += 1, s3 += _strideDims[3])
                            {
                                const Type * pSrc4 = pSrc3 + s3 * _srcStrides[4];
                                Type * pDst4 = pDst3 + d3 * _dstStrides[4];
                                for (size_t s4 = _beginDims[4], d4 = 0; d4 < _dstDims[4]; d4 += 1, s4 += _strideDims[4])
                                    pDst4[d4] = pSrc4[s4];
                            }
                        }
                    }
                }
                break;
            default:
                assert(0);
            }
        }

    private:
        Shape _axes, _beginDims, _endDims, _strideDims, _srcDims, _dstDims;
        Shape _srcStrides, _dstStrides;
    };
}
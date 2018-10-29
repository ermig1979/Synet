/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
    template <class T> class NormalizeLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        NormalizeLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const NormalizeParam & param = this->Param().normalize();
            _acrossSpatial = param.acrossSpatial();
            _channelShared = param.channelShared();
            _eps = param.eps();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0]->Count() >= 3);
            dst[0]->Reshape(src[0]->Shape());
            _buffer.Reshape({ 1, src[0]->Axis(-3), src[0]->Axis(-2), src[0]->Axis(-1) });
            if (!_acrossSpatial)
                _norm.Reshape({ src[0]->Axis(-4), 1, src[0]->Axis(-2), src[0]->Axis(-1) });
            size_t spatialDim = src[0]->Size(-2);
            _sumChannelMultiplier.Reshape({ 1, src[0]->Axis(-3), 1, 1 }, Type(1));
            if (spatialDim != _sumSpatialMultiplier.Size()) 
            {
                _sumSpatialMultiplier.Reshape({ 1, 1, src[0]->Axis(-2), src[0]->Axis(-1) });
                CpuSet(spatialDim, Type(1), _sumSpatialMultiplier.CpuData());
                _bufferSpatial.Reshape({ 1, 1, src[0]->Axis(-2), src[0]->Axis(-1) });
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            const Type * scale = this->Weight()[0].CpuData();
            Type * pBuffer = _buffer.CpuData();
            Type * pNorm = _norm.CpuData();
            CpuSet(_norm.Size(), Type(_eps), pNorm);
            const Type * sumChannelMultiplier = _sumChannelMultiplier.CpuData();
            const Type * sumSpatialMultiplier = _sumSpatialMultiplier.CpuData();
            size_t num = src[0]->Axis(0);
            size_t dim = src[0]->Size() / num;
            size_t spatialDim = src[0]->Size(-2);
            size_t channels = src[0]->Axis(1);
            for (size_t n = 0; n < num; ++n) 
            {
                CpuSqr(pSrc, dim, pBuffer);
                if (_acrossSpatial) 
                {
                    pNorm[n] = ::sqrt(CpuAbsSum(pBuffer, dim) + _eps);
                    CpuScale(pSrc, dim, Type(1) / pNorm[n], pDst);
                }
                else 
                {
                    CpuGemv(CblasTrans, channels, spatialDim, Type(1), pBuffer, sumChannelMultiplier, Type(1), pNorm);
                    CpuPow(pNorm, spatialDim, Type(0.5), pNorm);
                    CpuGemm(CblasNoTrans, CblasNoTrans, channels, spatialDim, 1, Type(1), sumChannelMultiplier, pNorm, Type(0), pBuffer);
                    CpuDiv(pSrc, pBuffer, dim,  pDst);
                    pNorm += spatialDim;
                }
                if (_channelShared)
                {
                    CpuScale(pDst, dim, scale[0], pDst);
                }
                else 
                {
                    CpuGemm(CblasNoTrans, CblasNoTrans, channels, spatialDim, 1, Type(1), scale, sumSpatialMultiplier, Type(0), pBuffer);
                    CpuMul(pDst, pBuffer, dim, pDst);
                }
                pSrc += dim;
                pDst += dim;
            }
        }

    private:
        typedef typename Base::Tensor Tensor;

        Tensor _buffer, _norm, _sumSpatialMultiplier, _bufferSpatial, _sumChannelMultiplier;
        bool _acrossSpatial, _channelShared;
        Type _eps;
    };
}
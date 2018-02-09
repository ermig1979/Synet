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
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> class BatchNormLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        BatchNormLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            const BatchNormParam & param = this->Param().batchNorm();
            _movingAverageFraction = param.movingAverageFraction();
            _eps = param.eps();
            _useGlobalStats = param.useGlobalStats();
            if (src[0]->Count() == 1)
                _channels = 1;
            else
                _channels = src[0]->Axis(1);
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            if (src[0]->Count() >= 1)
                assert(src[0]->Axis(1) == _channels);
            dst[0]->Reshape(src[0]->Shape());

            _mean.Reshape({_channels});
            _variance.Reshape({ _channels });
            _temp.Reshape(src[0]->Shape());
            _xNorm.Reshape(src[0]->Shape());
            _batchSumMultiplier.Reshape({src[0]->Axis(0)});

            Shape spatialDim = { src[0]->Size() / (_channels*src[0]->Axis(0)) };
            if (_spatialSumMultiplier.Shape() != spatialDim)
            {
                _spatialSumMultiplier.Reshape(spatialDim);
                CpuSet(_spatialSumMultiplier.Size(), Type(1), _spatialSumMultiplier.Data());
            }

            Shape numByChans = { _channels*src[0]->Axis(0) };
            if (_numByChans.Shape() != numByChans)
            {
                _numByChans.Reshape(numByChans);
                CpuSet(_batchSumMultiplier.Size(), Type(1), _batchSumMultiplier.Data());
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pSrc = src[0]->Data();
            Type * pDst = dst[0]->Data();
            size_t num = src[0]->Axis(0);
            size_t spatialDim = src[0]->Size() / (num*_channels);

            if (src[0] != dst[0])
                CpuCopy(pSrc, src[0]->Size(), pDst);

            if (_useGlobalStats)
            {
                const Type scaleFactor = this->Weight()[2].Data()[0] == 0 ? 0 : 1 / this->Weight()[2].Data()[0];
                CpuScale(this->Weight()[0].Data(), _mean.Size(), scaleFactor, _mean.Data());
                CpuScale(this->Weight()[1].Data(), _variance.Size(), scaleFactor, _variance.Data());
            }
            else 
            {
                CpuGemv<Type>(CblasNoTrans, _channels * num, spatialDim, Type(1) / (num * spatialDim), pSrc, _spatialSumMultiplier.Data(), Type(0), _numByChans.Data());
                CpuGemv<Type>(CblasTrans, num, _channels, Type(1), _numByChans.Data(), _batchSumMultiplier.Data(), Type(0), _mean.Data());
            }

            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, num, _channels, 1, 1, _batchSumMultiplier.Data(), _mean.Data(), Type(0), _numByChans.Data());
            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _channels * num, spatialDim, 1, -1, _numByChans.Data(), _spatialSumMultiplier.Data(), Type(1), pDst);

            if (!_useGlobalStats) 
            {
                CpuPow(pDst, dst[0]->Size(), Type(2), _temp.Data());
                CpuGemv<Type>(CblasNoTrans, _channels * num, spatialDim, Type(1) / (num * spatialDim), _temp.Data(), _spatialSumMultiplier.Data(), Type(0), _numByChans.Data());
                CpuGemv<Type>(CblasTrans, num, _channels, Type(1), _numByChans.Data(), _batchSumMultiplier.Data(), Type(0), _variance.Data());

                Tensors & weights = (Tensors&)this->Weight();
                weights[2].Data()[0] *= _movingAverageFraction;
                weights[2].Data()[0] += Type(1);
                size_t m = src[0]->Size() / _channels;
                Type biasCorrectionFactor = m > 1 ? Type(m) / (m - 1) : Type(1);
                CpuAxpby(_variance.Size(), biasCorrectionFactor, _variance.Data(), _movingAverageFraction, weights[1].Data());
            }

            CpuAdd(_eps, _variance.Data(), _variance.Size());
            CpuPow(_variance.Data(), _variance.Size(), Type(0.5), _variance.Data());

            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, num, _channels, 1, 1, _batchSumMultiplier.Data(), _variance.Data(), Type(0), _numByChans.Data());
            CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _channels * num, spatialDim, 1, Type(1), _numByChans.Data(), _spatialSumMultiplier.Data(), Type(0), _temp.Data());
            CpuDiv(pDst, _temp.Data(), _temp.Size(), pDst);
            CpuCopy(pDst, _xNorm.Size(), _xNorm.Data());
        }

    private:
        size_t _channels;
        bool _useGlobalStats;
        Type _movingAverageFraction, _eps;
        Tensor _mean, _variance, _temp, _xNorm;
        Tensor _batchSumMultiplier, _numByChans, _spatialSumMultiplier;
    };
}
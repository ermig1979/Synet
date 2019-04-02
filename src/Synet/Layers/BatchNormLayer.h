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
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Utils/Math.h"
#include "Synet/Utils/Gemm.h"

namespace Synet
{
    template <class T> class BatchNormLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        BatchNormLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const BatchNormParam & param = this->Param().batchNorm();
            _movingAverageFraction = param.movingAverageFraction();
            _eps = param.eps();
            _useGlobalStats = param.useGlobalStats();
            _yoloCompatible = param.yoloCompatible();
            if (src[0]->Count() == 1)
                _channels = 1;
            else
                _channels = src[0]->Axis(1);            
            
            if (src[0]->Count() >= 1)
                assert(src[0]->Axis(1) == _channels);
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            if (_useGlobalStats)
            {
                Type scaleFactor = Type(1);
                if(this->Weight().size() > 2)
                    scaleFactor = this->Weight()[2].CpuData()[0] == 0 ? Type(0) : Type(1) / this->Weight()[2].CpuData()[0];
                _scale.Reshape({ _channels });
                _bias.Reshape({ _channels });
                for (size_t i = 0; i < _channels; ++i)
                {
                    if(_yoloCompatible)
                        _scale.CpuData()[i] = Type(1) / (::sqrt(this->Weight()[1].CpuData()[i]) + _eps);
                    else
                        _scale.CpuData()[i] = Type(1) / ::sqrt(_eps + this->Weight()[1].CpuData()[i] * scaleFactor);
                    _bias.CpuData()[i] = -this->Weight()[0].CpuData()[i] * scaleFactor * _scale.CpuData()[i];
                }
            }
            else
            {
                _mean.Reshape({ _channels });
                _variance.Reshape({ _channels });
                _temp.Reshape(src[0]->Shape());
                _batchSumMultiplier.Reshape({ src[0]->Axis(0) });

                Shape spatialDim = { src[0]->Size() / (_channels*src[0]->Axis(0)) };
                if (_spatialSumMultiplier.Shape() != spatialDim)
                {
                    _spatialSumMultiplier.Reshape(spatialDim);
                    CpuSet(_spatialSumMultiplier.Size(), Type(1), _spatialSumMultiplier.CpuData());
                }

                Shape numByChans = { _channels*src[0]->Axis(0) };
                if (_numByChans.Shape() != numByChans)
                {
                    _numByChans.Reshape(numByChans);
                    CpuSet(_batchSumMultiplier.Size(), Type(1), _batchSumMultiplier.CpuData());
                }
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            const Type * pSrc = src[0]->CpuData();
            Type * pDst = dst[0]->CpuData();
            size_t num = src[0]->Axis(0);
            size_t spatialDim = src[0]->Size() / (num*_channels);

            if (_useGlobalStats)
            {
                size_t size = src[0]->Size(1);
                for (size_t i = 0; i < num; ++i)
                {
                    Detail::ScaleLayerForwardCpu(pSrc, _scale.CpuData(), _bias.CpuData(), _channels, spatialDim, pDst, src[0]->Format() == TensorFormatNhwc);
                    pSrc += size;
                    pDst += size;
                }
            }
            else
            {
                if (src[0] != dst[0])
                    CpuCopy(pSrc, src[0]->Size(), pDst);

                CpuGemv<Type>(CblasNoTrans, _channels * num, spatialDim, Type(1) / (num * spatialDim), pSrc, _spatialSumMultiplier.CpuData(), Type(0), _numByChans.CpuData());
                CpuGemv<Type>(CblasTrans, num, _channels, Type(1), _numByChans.CpuData(), _batchSumMultiplier.CpuData(), Type(0), _mean.CpuData());

                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, num, _channels, 1, Type(1), _batchSumMultiplier.CpuData(), 1, _mean.CpuData(), _channels, Type(0), _numByChans.CpuData(), _channels);
                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _channels * num, spatialDim, 1, Type(-1), _numByChans.CpuData(), 1, _spatialSumMultiplier.CpuData(), spatialDim, Type(1), pDst, spatialDim);

                CpuPow(pDst, dst[0]->Size(), Type(2), _temp.CpuData());
                CpuGemv<Type>(CblasNoTrans, _channels * num, spatialDim, Type(1) / (num * spatialDim), _temp.CpuData(), _spatialSumMultiplier.CpuData(), Type(0), _numByChans.CpuData());
                CpuGemv<Type>(CblasTrans, num, _channels, Type(1), _numByChans.CpuData(), _batchSumMultiplier.CpuData(), Type(0), _variance.CpuData());

                Tensors & weights = (Tensors&)this->Weight();
                weights[2].CpuData()[0] *= _movingAverageFraction;
                weights[2].CpuData()[0] += Type(1);
                size_t m = src[0]->Size() / _channels;
                Type biasCorrectionFactor = m > 1 ? Type(m) / (m - 1) : Type(1);
                CpuAxpby(_variance.Size(), biasCorrectionFactor, _variance.CpuData(), _movingAverageFraction, weights[1].CpuData());
                CpuAdd(_eps, _variance.CpuData(), _variance.Size());
                CpuPow(_variance.CpuData(), _variance.Size(), Type(0.5), _variance.CpuData());

                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, num, _channels, 1, 1, _batchSumMultiplier.CpuData(), 1, _variance.CpuData(), _channels, Type(0), _numByChans.CpuData(), _channels);
                CpuGemm<Type>(CblasNoTrans, CblasNoTrans, _channels * num, spatialDim, 1, Type(1), _numByChans.CpuData(), 1, _spatialSumMultiplier.CpuData(), spatialDim, Type(0), _temp.CpuData(), spatialDim);
                CpuDiv(pDst, _temp.CpuData(), _temp.Size(), pDst);
            }
        }

    private:
        size_t _channels;
        bool _useGlobalStats, _yoloCompatible;
        Type _movingAverageFraction, _eps;
        Tensor _mean, _variance, _temp;
        Tensor _batchSumMultiplier, _numByChans, _spatialSumMultiplier;
        Tensor _scale, _bias;
    };
}
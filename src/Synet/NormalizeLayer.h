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
    template <class T, template<class> class A> class NormalizeLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
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
            if (spatialDim != _sumSpatialMultiplier.Size()) 
            {
                _sumSpatialMultiplier.Reshape({ 1, 1, src[0]->Axis(-2), src[0]->Axis(-1) });
                CpuSet(spatialDim, Type(1), _sumSpatialMultiplier.Data());
                _bufferSpatial.Reshape({ 1, 1, src[0]->Axis(-2), src[0]->Axis(-1) });
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            //const Type * pSrc = src[0]->Data();
            //Type * pDst = dst[0]->Data();
            //const Type * scale = this->Weight()[0]->Data();
            //Type * pBuffer = _buffer.Data();
            //Type * pNorm = _norm.Data();
            //CpuSet(_norm.Size(), Type(_eps), pNorm);
            //const Type * sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
            //const Type * sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
            //size_t num = src[0]->Axis(0);
            //size_t dim = src[0]->Size() / num;
            //size_t spatialDim = src[0]->Size(-2);
            //int channels = src[0]->channels();
            //for (size_t n = 0; n < num; ++n) 
            //{
            //    caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
            //    if (_acrossSpatial) 
            //    {
            //        // add eps to avoid overflow
            //        norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data) + eps_, Dtype(0.5));
            //        caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data, top_data);
            //    }
            //    else 
            //    {
            //        CpuGemv(CblasTrans, channels, spatialDim, Type(1), pBuffer, sum_channel_multiplier, Type(1), pNorm);
            //        // compute norm
            //        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
            //        // scale the layer
            //        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
            //            1, Dtype(1), sum_channel_multiplier, norm_data,
            //            Dtype(0), buffer_data);
            //        caffe_div<Dtype>(dim, bottom_data, buffer_data, top_data);
            //        norm_data += spatial_dim;
            //    }
            //    if (_channelShared)
            //    {
            //        caffe_scal<Dtype>(dim, scale[0], top_data);
            //    }
            //    else 
            //    {
            //        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
            //            1, Dtype(1), scale, sum_spatial_multiplier,
            //            Dtype(0),
            //            buffer_data);
            //        caffe_mul<Dtype>(dim, top_data, buffer_data, top_data);
            //    }
            //    bottom_data += dim;
            //    top_data += dim;
            //}
        }

    private:
        typedef typename Base::Tensor Tensor;

        //Blob<Dtype> norm_;
        //Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
        //Blob<Dtype> buffer_, buffer_channel_, buffer_spatial_;
        Tensor _buffer, _norm, _sumSpatialMultiplier, _bufferSpatial;
        bool _acrossSpatial, _channelShared;
        Type _eps;
    };
}
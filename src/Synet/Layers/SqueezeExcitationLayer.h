/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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
#include "Synet/Layers/ScaleLayer.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    namespace Detail
    {
        template <class T, class S> void ChannelSum(const T* src, size_t channels, size_t height, size_t width, TensorFormat format, S * sum)
        {
            if (format == TensorFormatNhwc)
            {
                for (size_t c = 0; c < channels; ++c)
                    sum[c] = S(0);
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            sum[c] += src[c];
                        src += channels;
                    }
                }
            }
            else if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    sum[c] = S(0);
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                            sum[c] += src[w];
                        src += width;
                    }
                }
            }
            else
                assert(0);
        }
    }

    template <class T> class SqueezeExcitationLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        SqueezeExcitationLayer(const LayerParam& param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const Tensors& weight = this->Weight();
            assert(weight[0].Count() == 4 && weight[1].Count() == 4);
            _type = src[0]->GetType();
            _format = src[0]->Format();
            _batch = src[0]->Axis(0);
            if (_format == TensorFormatNchw)
            {
                _channels = src[0]->Axis(1);
                _height = src[0]->Axis(2);
                _width = src[0]->Axis(3);
                _squeeze = weight[0].Axis(0);
                assert(weight[1].Axis(0) == _channels);
            }
            else if (_format == TensorFormatNhwc)
            {
                _height = src[0]->Axis(1);
                _width = src[0]->Axis(2);
                _channels = src[0]->Axis(3);
                _squeeze = weight[0].Axis(3);
                assert(weight[1].Axis(3) == _channels);
            }
            else
                assert(0);
            _size = _channels * _height * _width;
            _kA = 1.0f / (_height * _width);
            _norm[0].Reshape(Shp(_channels));
            _norm[1].Reshape(Shp(_squeeze));

            if (_type == TensorType32f)
            {
                _sum.As32f().Reshape(Shp(_channels));
            }
            else if (_type == TensorType8u)
            {
                _sum.As32i().Reshape(Shp(_channels));
            }
            else
                assert(0);

            if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), _format);
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            if (_type == TensorType32f)
                ForwardCpu(src[0]->As32f().CpuData(), dst[0]->As32f().CpuData());
            else if (_type == TensorType8u)
                ForwardCpu(src[0]->As8u().CpuData(), dst[0]->As8u().CpuData());
            else
                assert(0);
        }

        void ForwardCpu(const float * src, float * dst)
        {
            float * sum = _sum.As32f().CpuData();
            float * norm0 = _norm[0].CpuData();
            float * norm1 = _norm[1].CpuData();
            const float * wgt0 = this->Weight()[0].CpuData();
            const float * wgt1 = this->Weight()[1].CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ChannelSum(src, _channels, _height, _width, _format, sum);
                CpuScale(sum, _channels, _kA, norm0);
                Product(_channels, _squeeze, norm0, wgt0, norm1);
                CpuRelu(norm1, _squeeze, 0.0f, norm1);
                Product(_squeeze, _channels, norm1, wgt1, norm0);
                CpuSigmoid(norm0, _channels, norm0);
                Detail::ScaleLayerForwardCpu<float>(src, norm0, NULL, _channels, _height, _width, dst, _format, 0);
                src += _size, dst += _size;
            }
        }

        void Product(size_t C, size_t D, const float* src, const float* weight, float* dst)
        {
            if (_format == TensorFormatNchw)
                CpuGemm(CblasNoTrans, CblasNoTrans, D, 1, C, 1.0f, weight, C, src, 1, 0.0f, dst, 1);
            else if (_format == TensorFormatNhwc)
                CpuGemm(CblasNoTrans, CblasNoTrans, 1, D, C, 1.0f, src, C, weight, D, 0.0f, dst, D);
        }

        void ForwardCpu(const uint8_t* src, uint8_t* dst)
        {
            int32_t * sum = _sum.As32i().CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ChannelSum(src, _channels, _height, _width, _format, sum);

                src += _size, dst += _size;
            }
        }

    private:
        TensorType _type;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width, _size, _squeeze; 
        Tensor _sum, _norm[2];
        float _kA;
    };
}
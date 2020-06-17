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
#include "Synet/Quantization/Convert.h"

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

        SqueezeExcitationLayer(const LayerParam& param, QuantizationMethod method)
            : Base(param)
            , _method(method)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const Tensors& weight = this->Weight();
            assert(weight[0].Count() == 4 && weight[1].Count() == 4);
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
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
            _kAvg = 1.0f / (_height * _width);

            if (_src8u)
            {
                Base::Extend32i(buf, 0, Shp(_channels), _format);
                Init8i();
            }
            else
                Base::Extend32f(buf, 0, Shp(_channels), _format);
            Base::Extend32f(buf, 1, Shp(_channels + _squeeze), _format);

            if (src[0] != dst[0])
            {
                if(_dst8u)
                    dst[0]->As8u().Reshape(src[0]->Shape(), _format);
                else
                    dst[0]->As32f().Reshape(src[0]->Shape(), _format);
            }
            this->UsePerfStat();
        }

        virtual bool Can8i() const
        {
            return _method != QuantizationMethodUnknown;
        }

        virtual bool Is8i() const
        {
            return _method != QuantizationMethodUnknown;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            float * norm0 = Base::Buf32f(buf, 1);
            float * norm1 = norm0 + _channels;
            if (_src8u)
            {
                if(_dst8u)
                    ForwardCpu(src[0]->As8u().CpuData(), Base::Buf32i(buf, 0), norm0, norm1, dst[0]->As8u().CpuData(), NULL);
                else
                    ForwardCpu(src[0]->As8u().CpuData(), Base::Buf32i(buf, 0), norm0, norm1, NULL, dst[0]->As32f().CpuData());
            }
            else
                ForwardCpu(src[0]->As32f().CpuData(), Base::Buf32f(buf, 0), norm0, norm1, dst[0]->As32f().CpuData());
        }

        void ForwardCpu(const float * src, float * sum, float * norm0, float * norm1, float * dst)
        {
            const float* wgt0 = this->Weight()[0].CpuData();
            const float* wgt1 = this->Weight()[1].CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ChannelSum(src, _channels, _height, _width, _format, sum);
                CpuScale(sum, _channels, _kAvg, norm0);
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

        void ForwardCpu(const uint8_t* src, int32_t * sum, float* norm0, float* norm1, uint8_t* dst8u, float* dst32f)
        {
            const float* wgt0 = this->Weight()[0].CpuData();
            const float* wgt1 = this->Weight()[1].CpuData();
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ChannelSum(src, _channels, _height, _width, _format, sum);
                Detail::Convert<int32_t, float, float>(sum, 1, _channels, 1, 1, TensorFormatNhwc, 
                    _sumScale.data(), _sumShift.data(), INT_MIN, INT_MAX, norm0);
                Product(_channels, _squeeze, norm0, wgt0, norm1);
                CpuRelu(norm1, _squeeze, 0.0f, norm1);
                Product(_squeeze, _channels, norm1, wgt1, norm0);
                CpuSigmoid(norm0, _channels, norm0);
                if(_dst8u)
                    Scale(src, norm0, dst8u, NULL), dst8u += _size;
                else
                    Scale(src, norm0, NULL, dst32f), dst32f += _size;
                src += _size;
            }
        }

        void Init8i()
        {
            _sumScale.resize(_channels);
            _sumShift.resize(_channels);
            Stat& statS = *this->Stats(0)[0];
            Stat& statD = *this->Stats(2)[0];
            statS.Init8u(_method);
            statD.Init8u(_method);
            for (size_t c = 0; c < _channels; c++)
            {
                _sumScale[c] = statS.scale8uTo32f[c] * _kAvg;
                _sumShift[c] = statS.shift8uTo32f[c];
            }
        }

        void Scale(const uint8_t* src, float* norm, uint8_t* dst8u, float *dst32f)
        {
            int lower, upper;
            if (_method == QuantizationMethodIECompatible)
                lower = QUANT_IE_COMP_SRC_U8_MIN, upper = QUANT_IE_COMP_SRC_U8_MAX;
            else if (_method == QuantizationMethodSymmetricNarrowed)
                lower = QUANT_SYMM_NARR_SRC_U8_MIN, upper = QUANT_SYMM_NARR_SRC_U8_MAX;
            const float* srcScale = this->Stats(0)[0]->scale8uTo32f.data();
            const float* srcShift = this->Stats(0)[0]->shift8uTo32f.data();
            const float* dstScale = this->Stats(2)[0]->scale32fTo8u.data();
            const float* dstShift = this->Stats(2)[0]->shift32fTo8u.data();
            if (_format == TensorFormatNchw)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        if (_dst8u)
                        {
                            for (size_t w = 0; w < _width; ++w)
                            {
                                float value = Detail::Convert<uint8_t, float, float>(src[w], srcScale[c], srcShift[c], INT_MIN, INT_MAX);
                                dst8u[w] = Detail::Convert<float, uint8_t, float>(value*norm[c], dstScale[c], dstShift[c], lower, upper);
                            }
                            dst8u += _width; 
                        }
                        else
                        {
                            for (size_t w = 0; w < _width; ++w)
                                dst32f[w] = Detail::Convert<uint8_t, float, float>(src[w], srcScale[c], srcShift[c], INT_MIN, INT_MAX) * norm[c];
                            dst32f += _width;
                        }
                        src += _width;
                    }
                }
            }
            else if (_format == TensorFormatNhwc)
            {
                for (size_t h = 0; h < _height; ++h)
                {
                    for (size_t w = 0; w < _width; ++w)
                    {
                        if (_dst8u)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                            {
                                float value = Detail::Convert<uint8_t, float, float>(src[c], srcScale[c], srcShift[c], INT_MIN, INT_MAX);
                                dst8u[c] = Detail::Convert<float, uint8_t, float>(value * norm[c], dstScale[c], dstShift[c], lower, upper);
                            }
                            dst8u += _channels;
                        }
                        else
                        {
                            for (size_t c = 0; c < _channels; ++c)
                                dst32f[c] = Detail::Convert<uint8_t, float, float>(src[c], srcScale[c], srcShift[c], INT_MIN, INT_MAX) * norm[c];
                            dst32f += _channels;
                        }
                        src += _channels;
                    }
                }
            }
            else
                assert(0);
        }

    private:
        bool _src8u, _dst8u;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width, _size, _squeeze; 
        float _kAvg;
        QuantizationMethod _method;
        Floats _sumScale, _sumShift;
    };
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar,
*               2019-2019 Artur Voronkov.
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
#include "Synet/Quantization/Convert.h"

namespace Synet
{
    template <class T> class AddLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        AddLayer(const LayerParam & param, Context* context, QuantizationMethod method)
            : Base(param, context)
            , _method(method)
        {
        }

        virtual bool Can8i() const
        {
            return _method != QuantizationMethodUnknown;
        }

        virtual bool Is8i() const
        {
            return _method != QuantizationMethodUnknown;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Shape() == src[1]->Shape() && src[0]->GetType() == src[1]->GetType() && src[0]->Count() == 4);
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
            _format = src[0]->Format();
            _batch = src[0]->Axis(0);
            if (_format == TensorFormatNchw)
            {
                _channels = src[0]->Axis(1);
                _height = src[0]->Axis(2);
                _width = src[0]->Axis(3);
            }
            else if (_format == TensorFormatNhwc)
            {
                _height = src[0]->Axis(1);
                _width = src[0]->Axis(2);
                _channels = src[0]->Axis(3);
            }
            else
                assert(0);

            if (_src8u || _dst8u)
            {
                this->Stats(0)[0]->Init8u(_method);
                this->Stats(0)[1]->Init8u(_method);
                this->Stats(2)[0]->Init8u(_method);
            }

            if (src[0] != dst[0])
            {
                if (_dst8u)
                    dst[0]->As8u().Reshape(src[0]->Shape(), _format);
                else
                    dst[0]->As32f().Reshape(src[0]->Shape(), _format);
            }
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_src8u)
            {
                if (_dst8u)
                    Add8i(src[0]->As8u().CpuData(), src[1]->As8u().CpuData(), dst[0]->As8u().CpuData());
                else
                    Add8i(src[0]->As8u().CpuData(), src[1]->As8u().CpuData(), dst[0]->As32f().CpuData());
            }
            else
            {
                if (_dst8u)
                    Add8i(src[0]->As32f().CpuData(), src[1]->As32f().CpuData(), dst[0]->As8u().CpuData());
                else
                    CpuAdd(src[0]->As32f().CpuData(), src[1]->As32f().CpuData(), src[0]->Size(), dst[0]->As32f().CpuData());
            }
        }
        
        void Add8i(const uint8_t* src0, const uint8_t* src1, uint8_t * dst)
        {
            const float* scaleSrc0 = this->Stats(0)[0]->scale8uTo32f.data();
            const float* shiftSrc0 = this->Stats(0)[0]->shift8uTo32f.data();
            const float* scaleSrc1 = this->Stats(0)[1]->scale8uTo32f.data();
            const float* shiftSrc1 = this->Stats(0)[1]->shift8uTo32f.data();
            const float* scaleDst = this->Stats(2)[0]->scale32fTo8u.data();
            const float* shiftDst = this->Stats(2)[0]->shift32fTo8u.data();
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
            {
                SimdSynetCompatibilityType compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
                ::SimdSynetAdd8i(src0, scaleSrc0, shiftSrc0, src1, scaleSrc1, shiftSrc1, dst, scaleDst, shiftDst, 
                    _batch, _channels, _height * _width, (SimdTensorFormatType)_format, compatibility);
                return;
            }
#endif
            int upper = ((_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ? 
                QUANT_SYMM_NARR_SRC_U8_MAX : QUANT_IE_COMP_SRC_U8_MAX);
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        for (size_t h = 0; h < _height; ++h)
                        {
                            for (size_t w = 0; w < _width; ++w)
                            {
                                float s0 = Detail::Convert<uint8_t, float, float>(src0[w], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                                float s1 = Detail::Convert<uint8_t, float, float>(src1[w], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                                dst[w] = Detail::Convert<float, uint8_t, float>(s0 + s1, scaleDst[c], shiftDst[c], 0, upper);
                            }
                            dst += _width, src0 += _width, src1 += _width;
                        }
                    }
                }
                else if (_format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        for (size_t w = 0; w < _width; ++w)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                            {
                                float s0 = Detail::Convert<uint8_t, float, float>(src0[c], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                                float s1 = Detail::Convert<uint8_t, float, float>(src1[c], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                                dst[c] = Detail::Convert<float, uint8_t, float>(s0 + s1, scaleDst[c], shiftDst[c], 0, upper);
                            }
                            dst += _channels, src0 += _channels, src1 += _channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        void Add8i(const uint8_t* src0, const uint8_t* src1, float* dst)
        {
            const float* scaleSrc0 = this->Stats(0)[0]->scale8uTo32f.data();
            const float* shiftSrc0 = this->Stats(0)[0]->shift8uTo32f.data();
            const float* scaleSrc1 = this->Stats(0)[1]->scale8uTo32f.data();
            const float* shiftSrc1 = this->Stats(0)[1]->shift8uTo32f.data();
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        for (size_t h = 0; h < _height; ++h)
                        {
                            for (size_t w = 0; w < _width; ++w)
                            {
                                float s0 = Detail::Convert<uint8_t, float, float>(src0[w], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                                float s1 = Detail::Convert<uint8_t, float, float>(src1[w], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                                dst[w] = s0 + s1;
                            }
                            dst += _width, src0 += _width, src1 += _width;
                        }
                    }
                }
                else if (_format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        for (size_t w = 0; w < _width; ++w)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                            {
                                float s0 = Detail::Convert<uint8_t, float, float>(src0[c], scaleSrc0[c], shiftSrc0[c], INT_MIN, INT_MAX);
                                float s1 = Detail::Convert<uint8_t, float, float>(src1[c], scaleSrc1[c], shiftSrc1[c], INT_MIN, INT_MAX);
                                dst[c] = s0 + s1;
                            }
                            dst += _channels, src0 += _channels, src1 += _channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        void Add8i(const float* src0, const float* src1, uint8_t * dst)
        {
            const float* scaleDst = this->Stats(2)[0]->scale32fTo8u.data();
            const float* shiftDst = this->Stats(2)[0]->shift32fTo8u.data();
            int upper = ((_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed) ?
                QUANT_SYMM_NARR_SRC_U8_MAX : QUANT_IE_COMP_SRC_U8_MAX);
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < _channels; ++c)
                    {
                        for (size_t h = 0; h < _height; ++h)
                        {
                            for (size_t w = 0; w < _width; ++w)
                                dst[w] = Detail::Convert<float, uint8_t, float>(src0[w] + src1[w], scaleDst[c], shiftDst[c], 0, upper);
                            dst += _width, src0 += _width, src1 += _width;
                        }
                    }
                }
                else if (_format == TensorFormatNhwc)
                {
                    for (size_t h = 0; h < _height; ++h)
                    {
                        for (size_t w = 0; w < _width; ++w)
                        {
                            for (size_t c = 0; c < _channels; ++c)
                                dst[c] = Detail::Convert<float, uint8_t, float>(src0[c] + src1[c], scaleDst[c], shiftDst[c], 0, upper);
                            dst += _channels, src0 += _channels, src1 += _channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width;
    };
}
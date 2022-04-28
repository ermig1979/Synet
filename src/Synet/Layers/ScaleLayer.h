/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
        template <typename T> void ScaleLayerForwardCpu(const T * src, const T * scale, const T * bias, size_t channels, size_t height, size_t width, T * dst, TensorFormat format, int compatibility)
        {
            if (format == TensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    const T s = scale[c];
                    const T b = bias ? bias[c] : 0;
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            dst[w] = src[w] * s + b;
                        }
                        src += width;
                        dst += width;
                    }
                }
            }
            else if (format == TensorFormatNhwc)
            {
                if (bias)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c] + bias[c];
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = src[c] * scale[c];
                            src += channels;
                            dst += channels;
                        }
                    }
                }
            }
            else
                assert(0);
        }

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        template <> SYNET_INLINE void ScaleLayerForwardCpu<float>(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float* dst, TensorFormat format, int compatibility)
        {
            ::SimdSynetScaleLayerForward(src, scale, bias, channels, height, width, dst, (::SimdTensorFormatType)format, (::SimdSynetCompatibilityType)compatibility);
        }
#endif

        //-------------------------------------------------------------------------

        template<class S, class D> D Scale8i(S value, float scale, float shift, int lower, int upper);

        template<> SYNET_INLINE float Scale8i<uint8_t, float>(uint8_t value, float scale, float shift, int lower, int upper)
        {
            return float(value) * scale + shift;
        }

        template<> SYNET_INLINE uint8_t Scale8i<uint8_t, uint8_t>(uint8_t value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Synet::RestrictRange(Round(float(value) * scale + shift), lower, upper);
        }

        template<> SYNET_INLINE uint8_t Scale8i<float, uint8_t>(float value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Synet::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<> SYNET_INLINE float Scale8i<float, float>(float value, float scale, float shift, int lower, int upper)
        {
            return value * scale + shift;
        }

        template<class S, class D> void Scale8i(const S* src, size_t batch, size_t channels, size_t spatial,
            TensorFormat format, const float* scale, const float* shift, int lower, int upper, D* dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == TensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float _scale = scale[c];
                        float _shift = shift[c];
                        for (size_t s = 0; s < spatial; ++s)
                            dst[s] = Scale8i<S, D>(src[s], _scale, _shift, lower, upper);
                        src += spatial;
                        dst += spatial;
                    }
                }
                else if (format == TensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = Scale8i<S, D>(src[c], scale[c], shift[c], lower, upper);
                        src += channels;
                        dst += channels;
                    }
                }
                else
                    assert(0);
            }
        }
    }

    //-------------------------------------------------------------------------

    class Scale8i
    {
    public:
        Scale8i()
            : _context(NULL)
            , _batch(0)
            , _spatial(0)
        {
        }

        virtual ~Scale8i()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(size_t batch, size_t channels, size_t spatial, TensorType srcType, TensorType dstType, TensorFormat format, QuantizationMethod method)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_batch != batch || _spatial != spatial)
            {
                _batch = batch, _spatial = spatial;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                SimdSynetCompatibilityType compatibility;
                if (method == QuantizationMethodSymmetricNarrowed || method == QuantizationMethodUnifiedNarrowed)
                    compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
                else
                    return;
                _context = ::SimdSynetScale8iInit(batch, channels, spatial, (SimdTensorDataType)srcType, (SimdTensorDataType)dstType, (SimdTensorFormatType)format, compatibility);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE size_t InternalBufferSize() const
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return _context ? ::SimdSynetScale8iInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        SYNET_INLINE void SetParams(const float* weight, const float* bias, const float* const* stats)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetScale8iSetParams(_context, weight, bias, stats);
#endif
        }

        SYNET_INLINE void Forward(const uint8_t* src, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetScale8iForward(_context, src, dst);
#endif
        }

    private:
        void* _context;
        size_t _batch, _spatial;
    };

    //-------------------------------------------------------------------------

    template <class T> class ScaleLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        ScaleLayer(const LayerParam & param, Context* context, QuantizationMethod method)
            : Base(param, context)
            , _method(method)
        {
            _is8i = _method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed;
        }

        virtual bool Is8i() const
        {
            return _is8i;
        }

        virtual bool Can8i() const
        {
            return _is8i;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const ScaleParam & param = this->Param().scale();
            _axis = param.axis();
            _biasTerm = param.biasTerm();
            assert(this->Weight().size());
            if (_biasTerm)
            {
                assert(this->Weight().size() > 1);
                assert(this->Weight()[0].Shape() == this->Weight()[1].Shape());
            }
            const Tensor & scale = this->Weight()[0];
            _channels = scale.Size();
            _src8u = src[0]->GetType() == TensorType8u;
            _dst8u = dst[0]->GetType() == TensorType8u;
            _format = src[0]->Format();
            if (scale.Size() == src[0]->Size())
            {
                _batch = 1;
                _height = 1;
                _width = 1;
                if (_format == TensorFormatUnknown)
                    _format = TensorFormatNchw;
            }
            else
            {
                _batch = src[0]->Size(0, _axis);
                if (src[0]->Count() < 4)
                {
                    _height = 1;
                    _width = src[0]->Size() / _batch / _channels;
                }
                else
                {
                    _height = _format == TensorFormatNhwc ? src[0]->Axis(1) : src[0]->Axis(2);
                    _width = _format == TensorFormatNhwc ? src[0]->Axis(2) : src[0]->Axis(3);
                }
            }
            assert(src[0]->Size() == _batch*_channels*_height*_width);
            if (_is8i)
            {
                _scale8i.Init(_batch, _channels, _height * _width, src[0]->GetType(), dst[0]->GetType(), _format, _method);
                if (_scale8i.Enable())
                {
                    const float* bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
                    const float* stats[4] = {
                        this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                        this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                        this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                        this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data() };
                    _scale8i.SetParams(this->Weight()[0].CpuData(), bias, stats);
                }
                else
                    Init8i();
                if (_dst8u)
                    dst[0]->As8u().Reshape(src[0]->Shape(), _format);
                else
                    dst[0]->As32f().Reshape(src[0]->Shape(), _format);
            }
            else if (src[0] != dst[0])
                dst[0]->Reshape(src[0]->Shape(), _format);
            this->UsePerfStat();
            _compatibility = 1;
        }

        virtual size_t MemoryUsage() const
        { 
            return Base::MemoryUsage() + _scale.MemoryUsage() + _shift.MemoryUsage() + _scale8i.InternalBufferSize();
        }

        virtual void CompactWeight()
        {
            if (_is8i)
            {
                ((Tensor&)this->Weight()[0]).Clear();
                if(_biasTerm)
                    ((Tensor&)this->Weight()[1]).Clear();
            }
        }

        virtual int64_t Flop() const
        {
            return _batch * _channels * _height * _width * 2;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_is8i)
            {
                if (_scale8i.Enable())
                    _scale8i.Forward(src[0]->RawCpuData(), dst[0]->RawCpuData());
                else
                {
                    const float* scale = _scale.CpuData();
                    const float* shift = _shift.CpuData();
                    if (_src8u && _dst8u)
                        Detail::Scale8i(src[0]->As8u().CpuData(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->As8u().CpuData());
                    else if (!_src8u && _dst8u)
                        Detail::Scale8i(src[0]->As32f().CpuData(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->As8u().CpuData());
                    else if (_src8u && !_dst8u)
                        Detail::Scale8i(src[0]->As8u().CpuData(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->As32f().CpuData());
                    else
                        Detail::Scale8i(src[0]->As32f().CpuData(), _batch, _channels, _height * _width, _format, scale, shift, _lower, _upper, dst[0]->As32f().CpuData());
                }
            }
            else
            {
                const float * scale = this->Weight()[0].CpuData();
                const float * bias = _biasTerm ? this->Weight()[1].CpuData() : NULL;
                Scale32f(src[0]->CpuData(), scale, bias, dst[0]->CpuData());
            }
        }

        void Scale32f(const float * src, const float* scale, const float * bias, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                Detail::ScaleLayerForwardCpu(src, scale, bias, _channels, _height, _width, dst, _format, _compatibility);
                src += _channels * _height * _width;
                dst += _channels * _height * _width;
            }
        }

        void Init8i()
        {
            Stat& statS = *this->Stats(0)[0];
            Stat& statD = *this->Stats(2)[0];
            statS.Init8u(_method);
            statD.Init8u(_method);
            _scale.Reshape(Shp(_channels), 1.0f);
            _shift.Reshape(Shp(_channels), 0.0f);
            if (_src8u)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    _scale.CpuData()[c] = statS.scale8uTo32f[c];
                    _shift.CpuData()[c] = statS.shift8uTo32f[c];
                }
            }
            const float* scale = this->Weight()[0].CpuData();            
            if (_biasTerm)
            {

                const float* bias = this->Weight()[1].CpuData();
                for (size_t c = 0; c < _channels; ++c)
                {
                    _scale.CpuData()[c] = _scale.CpuData()[c] * scale[c];
                    _shift.CpuData()[c] = _shift.CpuData()[c] * scale[c] + bias[c];
                }
            }
            else
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    _scale.CpuData()[c] = _scale.CpuData()[c] * scale[c];
                    _shift.CpuData()[c] = _shift.CpuData()[c] * scale[c];
                }
            }
            if (_dst8u)
            {
                for (size_t c = 0; c < _channels; ++c)
                {
                    _scale.CpuData()[c] = _scale.CpuData()[c] * statD.scale32fTo8u[c];
                    _shift.CpuData()[c] = _shift.CpuData()[c] * statD.scale32fTo8u[c] + statD.shift32fTo8u[c];
                }
            }
            if (_method == QuantizationMethodIECompatible)
                _lower = QUANT_IE_COMP_SRC_U8_MIN, _upper = QUANT_IE_COMP_SRC_U8_MAX;
            else if (_method == QuantizationMethodSymmetricNarrowed || _method == QuantizationMethodUnifiedNarrowed)
                _lower = QUANT_SYMM_NARR_SRC_U8_MIN, _upper = QUANT_SYMM_NARR_SRC_U8_MAX;
        }

    private:
        QuantizationMethod _method;
        TensorFormat _format;
        size_t _axis, _batch, _channels, _height, _width;
        int _compatibility, _lower, _upper;
        bool _biasTerm, _src8u, _dst8u, _is8i;
        Tensor _scale, _shift;
        Scale8i _scale8i;
    };
}
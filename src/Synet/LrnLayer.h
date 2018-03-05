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
    namespace Detail
    {
        template<class T> void LrnLayerCrossChannelsCpu(const T * src, size_t channels, size_t size, size_t inner, T alpha, T beta, T k, T * buffer, T * dst)
        {
            size_t prePad = (size - 1) / 2;
            size_t paddedSize = (channels + size - 1)*inner;
            size_t scaleSize = channels*inner;
            T * padded = buffer;
            T * scale = buffer + paddedSize;
            CpuSet(scaleSize, k, scale);
            CpuSet(paddedSize, T(0), padded);
            CpuSqr(src, scaleSize, padded + inner*prePad);
            for (size_t c = 0; c < size; ++c)
                CpuAxpy(padded + c*inner, inner, alpha, scale);
            for (size_t c = 1; c < channels; ++c)
            {
                CpuCopy(scale + (c - 1)*inner, inner, scale + c*inner);
                CpuAxpy(padded + (c + size - 1)*inner, inner, alpha, scale + c*inner);
                CpuAxpy(padded + (c - 1)*inner, inner, -alpha, scale + c*inner);
            }
            CpuPow(scale, scaleSize, -beta, dst);
            CpuMul(src, dst, scaleSize, dst);
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void LrnLayerCrossChannelsCpu<float>(const float * src, size_t channels, size_t size, size_t inner, float alpha, float beta, float k, float * buffer, float * dst)
        {
            float _k[3] = { k, alpha, -beta };
            ::SimdSynetLrnLayerCrossChannels(src, (size - 1)/2, channels, inner, _k, dst);
        }
#endif
    }

    template <class T, template<class> class A> class LrnLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        LrnLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            _normRegion = this->Param().lrn().normRegion();
            _size = this->Param().lrn().localSize();
            assert(_size % 2 == 1);
            _prePad = (_size - 1) / 2;
            _alpha = this->Param().lrn().alpha();
            _beta = this->Param().lrn().beta();
            _k = this->Param().lrn().k();
            if (_normRegion == NormRegionTypeWithinChannel)
            {
                assert(0);
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0]->Count() == 4);
            _num = src[0]->Axis(0);
            _channels = src[0]->Axis(1);
            _height = src[0]->Axis(2);
            _width = src[0]->Axis(3);
            switch (_normRegion)
            {
            case NormRegionTypeAcrossChannels:
                dst[0]->Reshape({ _num, _channels, _height, _width });
                _buffer.Reshape({ 1, _channels*2 + _size - 1, _height, _width });
                break;
            case NormRegionTypeWithinChannel:
                assert(0);
                break;
            default:
                assert(0);
                break;
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            switch (_normRegion)
            {
            case NormRegionTypeAcrossChannels:
                CrossChannelsCpu(src, buf, dst);
                break;
            case NormRegionTypeWithinChannel:
                assert(0);
                break;
            default:
                assert(0);
                break;
            }
        }

        virtual void CrossChannelsCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            Type alpha = _alpha / _size;
            for (size_t n = 0; n < _num; ++n)
            {
                const Type * pSrc = src[0]->Data({ n, 0, 0, 0 });
                Type * pDst = dst[0]->Data({ n, 0, 0, 0 });
                Detail::LrnLayerCrossChannelsCpu(pSrc, _channels, _size, _width*_height, alpha, _beta, _k, _buffer.Data(), pDst);
            }
        }
    
    private:

        NormRegionType _normRegion;
        size_t _size, _prePad, _num, _channels, _width, _height;
        Type _alpha, _beta, _k;
        Tensor _buffer;
    };
}
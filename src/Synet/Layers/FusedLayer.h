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

namespace Synet
{
    namespace Detail
    {
        template <class T> void FusedLayerForwardCpu0(const T * src, const T * bias, const T * scale, size_t count, size_t size, T * dst)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const T b = bias[i];
                const T s = scale[i];
                for (size_t j = 0; j < size; ++j)
                {
                    T x = src[j] + b;
                    dst[j] = (x - ::abs(x))*s + std::max(T(0), x);
                }
                src += size;
                dst += size;
            }
        }

        template <class T> void FusedLayerForwardCpu1(const T * src, const T * bias0, const T * scale1, const T * bias1, size_t count, size_t size, T * dst)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const T b0 = bias0[i];
                const T s1 = scale1[i];
                const T b1 = bias1[i];
                for (size_t j = 0; j < size; ++j)
                {
                    T x = src[j] + b0;
                    dst[j] = std::max(T(0), -x)*s1 + b1 + std::max(T(0), x);
                }
                src += size;
                dst += size;
            }
        }

        template <class T> void FusedLayerForwardCpu2(const T * src, const T * scale, const T * bias, size_t count, size_t size, T slope, T * dst)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const T s = scale[i];
                const T b = bias[i];
                for (size_t j = 0; j < size; ++j)
                {
                    T x = src[j]*s + b;
                    dst[j] = std::max(x, T(0)) + slope * std::min(x, T(0));
                }
                src += size;
                dst += size;
            }
        }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        template <> SYNET_INLINE void FusedLayerForwardCpu0<float>(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst)
        {
            ::SimdSynetFusedLayerForward0(src, bias, scale, count, size, dst);
        }

        template <> SYNET_INLINE void FusedLayerForwardCpu1<float>(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst)
        {
            ::SimdSynetFusedLayerForward1(src, bias0, scale1, bias1, count, size, dst);
        }

        template <> SYNET_INLINE void FusedLayerForwardCpu2<float>(const float * src, const float * scale, const float * bias, size_t count, size_t size, float slope, float * dst)
        {
            ::SimdSynetFusedLayerForward2(src, scale, bias, count, size, &slope, dst);
        }
#endif
    }

    template <class T> class FusedLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        FusedLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const FusedParam & fused = this->Param().fused();
            _type = fused.type();
            const Tensors & weight = this->Weight();
            switch (_type)
            {
            case 0:
            {
                assert(weight.size() == 3);
                _t0.bias.Share(weight[0]);
                _t0.count = _t0.bias.Size();
                _t0.scale.Reshape(_t0.bias.Shape());
                assert(weight[1].Size() == _t0.count || weight[1].Size() == 1);
                if (weight[1].Size() == _t0.count)
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] = weight[1].CpuData()[i];
                }
                else
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] = weight[1].CpuData()[0];
                }
                assert(weight[2].Size() == _t0.count || weight[2].Size() == 1);
                if (weight[2].Size() == _t0.count)
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] *= weight[2].CpuData()[i];
                }
                else
                {
                    for (size_t i = 0; i < _t0.count; ++i)
                        _t0.scale.CpuData()[i] *= weight[2].CpuData()[0];
                }
                break;
            }
            case 1:
            {
                assert(weight.size() == 5);
                _t1.bias0.Share(weight[0]);
                _t1.count = _t1.bias0.Size();
                assert(weight[1].Size() == 1 && weight[1].CpuData()[0] == -1.0f);
                assert(weight[2].Size() == 1 && weight[2].CpuData()[0] == 0.0f);
                _t1.scale1.Share(weight[3]);
                _t1.bias1.Share(weight[4]);
                break;
            }
            case 2:
                assert(weight.size() == 4 && weight[0].Count() == 1);
                assert(weight[0].Shape() == weight[1].Shape() && weight[0].Shape() == weight[2].Shape() && weight[0].Shape() == weight[3].Shape());
                assert(fused.floats().size() == 2);
                _t2.scale.Reshape(weight[0].Shape());
                _t2.bias.Reshape(weight[0].Shape());
                _t2.count = _t2.scale.Size();
                for (size_t i = 0; i < _t2.count; ++i)
                {
                    Type eps = fused.floats()[0];
                    Type scale = Type(1) / (::sqrt(weight[1].CpuData()[i]) + eps);
                    Type bias = - weight[0].CpuData()[i] * scale;
                    _t2.scale.CpuData()[i] = scale*weight[2].CpuData()[i];
                    _t2.bias.CpuData()[i] = bias*weight[2].CpuData()[i] + weight[3].CpuData()[i];
                }
                _t2.slope = fused.floats()[1];
                break;
            default:
                assert(0);
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            switch (_type)
            {
            case 0:
            {
                _t0.size = src[0]->Size() / _t0.count;
                assert(_t0.count == src[0]->Axis(1));
                assert(_t0.size*_t0.count == src[0]->Size());
                dst[0]->Reshape(src[0]->Shape());
                break;
            }
            case 1:
            {
                _t1.size = src[0]->Size() / _t1.count;
                assert(_t1.count == src[0]->Axis(1));
                assert(_t1.size*_t1.count == src[0]->Size());
                dst[0]->Reshape(src[0]->Shape());
                break;
            }
            case 2:
            {
                _t2.size = src[0]->Size() / _t2.count;
                assert(_t2.count == src[0]->Axis(1));
                assert(_t2.size*_t2.count == src[0]->Size());
                dst[0]->Reshape(src[0]->Shape());
                break;
            }
            default: 
                assert(0);
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();
            ForwardCpu(src[0]->CpuData(), dst[0]->CpuData());
        }

        void ForwardCpu(const Type * src, Type * dst)
        {
#ifdef SYNET_SIZE_STATISTIC
            std::stringstream ss;
            ss << " t=" << _type;
            ss << " c=" << (_type == 0 ? _t0.count : (_type == 1 ? _t1.count : _t2.count));
            ss << " s=" << (_type == 0 ? _t0.size : (_type == 1 ? _t1.size : _t2.size));
            SYNET_PERF_BLOCK(ss.str().c_str());
#else
            SYNET_PERF_FUNC();
#endif
            switch (_type)
            {
            case 0:
                Detail::FusedLayerForwardCpu0(src, _t0.bias.CpuData(), _t0.scale.CpuData(), _t0.count, _t0.size, dst);
                break;
            case 1:
                Detail::FusedLayerForwardCpu1(src, _t1.bias0.CpuData(), _t1.scale1.CpuData(), _t1.bias1.CpuData(), _t1.count, _t1.size, dst);
                break;
            case 2:
                Detail::FusedLayerForwardCpu2(src, _t2.scale.CpuData(), _t2.bias.CpuData(), _t2.count, _t2.size, _t2.slope, dst);
                break;
            default:
                assert(0);
            }
        }

    private:
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        int _type;

        struct T0
        {
            size_t count, size;
            Tensor bias, scale;
        } _t0;

        struct T1
        {
            size_t count, size;
            Tensor bias0, scale1, bias1;
        } _t1;

        struct T2
        {
            size_t count, size;
            Tensor scale, bias;
            Type slope;
        } _t2;
    };
}
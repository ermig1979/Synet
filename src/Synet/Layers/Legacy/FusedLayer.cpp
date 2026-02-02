/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layers/Legacy/FusedLayer.h"

#include "Synet/Layers/Math/ScaleLayer.h"

namespace Synet
{
    template <class T> SYNET_INLINE T FusedLayerForward0(T x, T s)
    {
        return (x - (T)Abs(x))*s + Max((T)0, x);
    }

    template <class T> void FusedLayerForwardCpu0(const T * src, const T * bias, const T * scale, size_t count, size_t size, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = FusedLayerForward0(src[i] + bias[i], scale[i]);
                src += count;
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    dst[j] = FusedLayerForward0(src[j] + bias[i], scale[i]);
                src += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T FusedLayerForward1(T x, T s, T b)
    {
        return Max((T)0, -x)*s + b + Max((T)0, x);
    }

    template <class T> void FusedLayerForwardCpu1(const T * src, const T * bias0, const T * scale1, const T * bias1, size_t count, size_t size, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = FusedLayerForward1(src[i] + bias0[i], scale1[i], bias1[i]);
                src += count;
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    dst[j] = FusedLayerForward1(src[j] + bias0[i], scale1[i], bias1[i]);
                src += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T FusedLayerForward2(T src, T scale, T bias, T slope)
    {
        T x = src * scale + bias;
        return Max((T)0, x) + Min((T)0, x)*slope;
    }

    template <class T> void FusedLayerForwardCpu2(const T * src, const T * scale, const T * bias, size_t count, size_t size, T slope, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = FusedLayerForward2(src[i], scale[i], bias[i], slope);
                src += count;
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    dst[j] = FusedLayerForward2(src[j], scale[i], bias[i], slope);
                src += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------
    
    template <class T> SYNET_INLINE T FusedLayerForward3(T x, T s)
    {
        return Max((T)0, x) + Min(x, (T)0) * s;
    }        
        
    template <class T> void FusedLayerForwardCpu3(const T * src, const T * bias, const T * scale, size_t count, size_t size, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = FusedLayerForward3(src[i] + bias[i], scale[i]);
                src += count;
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    dst[j] = FusedLayerForward3(src[j] + bias[i], scale[i]);
                src += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> void FusedLayerForwardCpu4(const T * src, const T * bias0, const T * scale1, const T * bias1, size_t count, size_t size, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    T x = src[i] + bias0[i];
                    dst[i] = Max((T)0, x);
                    dst[i + count] = Max((T)0, x*scale1[0] + bias1[0]);
                }
                src += count;
                dst += 2 * count;
            }
        }
        else
        {
            T * dst0 = dst, * dst1 = dst + count*size;
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    T x = src[j] + bias0[i];
                    dst0[j] = Max((T)0, x);
                    dst1[j] = Max((T)0, x*scale1[0] + bias1[0]);
                }
                src += size;
                dst0 += size;
                dst1 += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T FusedLayerForward8(T x0, T x1, T x2)
    {
        return x0 + x1*x2;
    }

    template <class T> void FusedLayerForwardCpu8(const T * src0, const T * src1, const T * src2, size_t count, size_t size, T * dst, int trans)
    {
        if ((trans || size == 1) && count != 1)
        {
            for (size_t j = 0; j < size; ++j)
            {
                for (size_t i = 0; i < count; ++i)
                    dst[i] = FusedLayerForward8(src0[i], src1[i], src2[i]);
                src0 += count;
                src1 += count;
                dst += count;
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                for (size_t j = 0; j < size; ++j)
                    dst[j] = FusedLayerForward8(src0[j], src1[j], src2[i]);
                src0 += size;
                src1 += size;
                dst += size;
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T FusedLayerForward9(T src, T scale, T bias)
    {
        return Max((T)0, src * scale + bias);
    }

    template <class T> void FusedLayerForwardCpu9(const T * src0, const T * src1, const T * scale0, const T * bias0, size_t count0, size_t count1, size_t size, T * dst0, T * dst1, int trans)
    {
        const T * scale1 = scale0 + count0;
        const T * bias1 = bias0 + count0;
        if (trans || size == 1)
        {
            if (dst1)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    for (size_t i = 0; i < count0; ++i)
                        dst0[i] = FusedLayerForward9(src0[i], scale0[i], bias0[i]), dst1[i] = src0[i];
                    src0 += count0, dst0 += count0, dst1 += count0;
                    for (size_t i = 0; i < count1; ++i)
                        dst0[i] = FusedLayerForward9(src1[i], scale1[i], bias1[i]), dst1[i] = src1[i];
                    src1 += count1, dst0 += count1, dst1 += count1;
                }
            }
            else
            {
                for (size_t j = 0; j < size; ++j)
                {
                    for (size_t i = 0; i < count0; ++i)
                        dst0[i] = FusedLayerForward9(src0[i], scale0[i], bias0[i]);
                    src0 += count0, dst0 += count0;
                    for (size_t i = 0; i < count1; ++i)
                        dst0[i] = FusedLayerForward9(src1[i], scale1[i], bias1[i]);
                    src1 += count1, dst0 += count1;
                }
            }
        }
        else
        {
            if (dst1)
            {
                for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size, dst1 += size)
                    for (size_t j = 0; j < size; ++j)
                        dst0[j] = FusedLayerForward9(src0[j], scale0[i], bias0[i]), dst1[j] = src0[j];
                for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size, dst1 += size)
                    for (size_t j = 0; j < size; ++j)
                        dst0[j] = FusedLayerForward9(src1[j], scale1[i], bias1[i]), dst1[j] = src1[j];
            }
            else
            {
                for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size)
                    for (size_t j = 0; j < size; ++j)
                        dst0[j] = FusedLayerForward9(src0[j], scale0[i], bias0[i]);
                for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size)
                    for (size_t j = 0; j < size; ++j)
                        dst0[j] = FusedLayerForward9(src1[j], scale1[i], bias1[i]);
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> SYNET_INLINE T FusedLayerForward11(T src, T shift, T lower, T upper, T scale)
    {
        return Max(lower, Min(src + shift, upper))*scale*src;
    }

    template <class T> void FusedLayerForwardCpu11(const T * src, size_t size, const T * params, T * dst)
    {
        for (size_t i = 0; i < size; ++i)
            dst[i] = FusedLayerForward11(src[i], params[0], params[1], params[2], params[3]);
    }

    //-------------------------------------------------------------------------------------------------

    FusedLayer::FusedLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool FusedLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        for (size_t i = 0; i < src.size(); ++i)
            if (src[i]->GetType() != TensorType32f && src[i]->GetType() != TensorType64i)
                SYNET_ERROR("FusedLayer: all inputs must be FP32 type!");
        const FusedParam & fused = this->Param().fused();
        _type = fused.type();
        const Tensors & weight = this->Weight();
        _format = src[0]->Format();
        _trans = _format == TensorFormatNhwc;
        switch (_type)
        {
        case 0:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V0 supports 1 input and 1 output!");
            if (weight.size() != 3)
                SYNET_ERROR("FusedLayer V0 needs 3 weights!");
            _t0.bias.Share(weight[0]);
            _count = _t0.bias.Size();
            _t0.scale.Reshape(TensorType32f, _t0.bias.Shape());
            if (weight[1].Size() != _count && weight[1].Size() != 1)
                SYNET_ERROR("FusedLayer V0: chaeck weight[1] size!");
            if (weight[1].Size() == _count)
            {
                for (size_t i = 0; i < _count; ++i)
                    _t0.scale.Data<float>()[i] = weight[1].Data<float>()[i];
            }
            else
            {
                for (size_t i = 0; i < _count; ++i)
                    _t0.scale.Data<float>()[i] = weight[1].Data<float>()[0];
            }
            if (weight[2].Size() != _count && weight[2].Size() != 1)
                SYNET_ERROR("FusedLayer V0: chaeck weight[2] size!");
            if (weight[2].Size() == _count)
            {
                for (size_t i = 0; i < _count; ++i)
                    _t0.scale.Data<float>()[i] *= weight[2].Data<float>()[i];
            }
            else
            {
                for (size_t i = 0; i < _count; ++i)
                    _t0.scale.Data<float>()[i] *= weight[2].Data<float>()[0];
            }
            break;
        }
        case 1:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V1 supports 1 input and 1 output!");
            if (weight.size() != 5)
                SYNET_ERROR("FusedLayer V1 needs 5 weights!");
            _t1.bias0.Share(weight[0]);
            _count = _t1.bias0.Size();
            assert(weight[1].Size() == 1 && weight[1].Data<float>()[0] == -1.0f);
            assert(weight[2].Size() == 1 && weight[2].Data<float>()[0] == 0.0f);
            _t1.scale1.Share(weight[3]);
            _t1.bias1.Share(weight[4]);
            break;
        }
        case 2:
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V0 supports 2 input and 1 output!");
            assert(weight.size() == 4 && weight[0].Count() == 1);
            assert(weight[0].Shape() == weight[1].Shape() && weight[0].Shape() == weight[2].Shape() && weight[0].Shape() == weight[3].Shape());
            assert(fused.floats().size() == 2);
            _t2.scale.Reshape(TensorType32f, weight[0].Shape());
            _t2.bias.Reshape(TensorType32f, weight[0].Shape());
            _count = _t2.scale.Size();
            for (size_t i = 0; i < _count; ++i)
            {
                float eps = fused.floats()[0];
                float scale = 1.0f / (::sqrt(weight[1].Data<float>()[i]) + eps);
                float bias = -weight[0].Data<float>()[i] * scale;
                _t2.scale.Data<float>()[i] = scale*weight[2].Data<float>()[i];
                _t2.bias.Data<float>()[i] = bias*weight[2].Data<float>()[i] + weight[3].Data<float>()[i];
            }
            _t2.slope = fused.floats()[1];
            break;
        case 3:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V3 supports 1 input and 1 output!");
            assert(weight.size() == 2 && weight[0].Size() == weight[1].Size());
            _t3.bias.Share(weight[0]);
            _count = _t3.bias.Size();
            _t3.scale.Share(weight[1]);
            break;
        }
        case 4:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V4 supports 1 input and 1 output!");
            assert(weight.size() == 1 && fused.floats().size() == 2);
            _t4.bias0.Share(weight[0]);
            _count = _t4.bias0.Size();
            _t4.scale1 = fused.floats()[0];
            _t4.bias1 = fused.floats()[1];
            break;
        }
        case 5:
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V5 supports 1 input and 1 output!");
            assert(weight.size() == 4);
            assert(weight[0].Shape() == weight[1].Shape() && weight[0].Shape() == weight[2].Shape() && weight[0].Shape() == weight[3].Shape());
            _t2.scale.Reshape(TensorType32f, weight[0].Shape());
            _t2.bias.Reshape(TensorType32f, weight[0].Shape());
            _count = _t2.scale.Size();
            for (size_t i = 0; i < _count; ++i)
            {
                _t2.scale.Data<float>()[i] = weight[0].Data<float>()[i] * weight[2].Data<float>()[i];
                _t2.bias.Data<float>()[i] = weight[1].Data<float>()[i] * weight[2].Data<float>()[i] + weight[3].Data<float>()[i];
            }
            _t2.slope = 0.0f;
            break;
        case 6:
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V6 supports 1 input and 1 output!");
            assert(weight.size() == 2);
            assert(weight[0].Shape() == weight[1].Shape());
            _t2.scale.Share(weight[0]);
            _t2.bias.Share(weight[1]);
            _count = _t2.scale.Size();
            _t2.slope = 0.0f;
            break;
        case 7:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V7 supports 1 input and 1 output!");
            assert(weight.size() == 3);
            _t1.bias0.Share(weight[0]);
            _count = _t1.bias0.Size();
            _t1.scale1.Share(weight[1]);
            _t1.bias1.Share(weight[2]);
            break;
        }
        case 8:
        {
            if (src.size() != 3 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V8 supports 3 inputs and 1 output!");
            assert(src[0]->Shape() == src[1]->Shape());
            _count = src[2]->Size(fused.axis());
            break;
        }
        case 9:
        {
            if (src.size() != 2 || (dst.size() != 1 && dst.size() != 2))
                SYNET_ERROR("FusedLayer V9 supports 2 inputs and 1-2 outputs!");
            assert(weight.size() == 2);
            _t5.scale.Share(weight[0]);
            _t5.bias.Share(weight[1]);
            _t5.count0 = src[0]->Axis(_trans ? 3 : 1);
            _t5.count1 = src[1]->Axis(_trans ? 3 : 1);
            _count = _t5.count0;
            break;
        }
        case 10:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V10 supports 1 input and 1 output!");
            assert(weight.size() == 2);
            assert(weight[0].Shape() == weight[1].Shape());
            _t0.scale.Reshape(TensorType32f, weight[0].Shape());
            _t0.bias.Reshape(TensorType32f, weight[0].Shape());
            _count = _t0.scale.Size();
            assert(fused.floats().size() == 4);
            float preScale = fused.floats()[0];
            float preBias = fused.floats()[1];
            const float * scale = weight[0].Data<float>();
            const float * bias = weight[1].Data<float>();
            float postScale = fused.floats()[2];
            float postBias = fused.floats()[3];
            for (size_t i = 0; i < _count; ++i)
            {
                _t0.scale.Data<float>()[i] = preScale * scale[i] * postScale;
                _t0.bias.Data<float>()[i] = (preBias * scale[i] + bias[i]) * postScale + postBias;
            }
            break;
        }
        case 11:
        {
            if (src.size() != 1 || dst.size() != 1)
                SYNET_ERROR("FusedLayer V11 supports 1 input and 1 output!");
            assert(fused.floats().size() == 4);
            for(size_t i = 0; i < 4; ++i)
                _t11.params[i] = fused.floats()[i];
            _count = 1;
            break;
        }
        default:
            SYNET_ERROR("FusedLayer unkown type!");
        }

        _num = src[0]->Size(0, fused.axis());
        _size = src[0]->Size() / _count / _num;
        assert(_num*_size*_count == src[0]->Size());
        Shape shape = src[0]->Shape();
        if (_type == 4)
            shape[_trans ? 3 : 1] *= 2;
        if (_type == 9)
            shape[_trans ? 3 : 1] += src[1]->Shape()[_trans ? 3 : 1];
        dst[0]->Reshape(TensorType32f, shape, src[0]->Format());
        if (_type == 9 && dst.size() == 2)
            dst[1]->Reshape(TensorType32f, shape, src[0]->Format());
        _srcStride = src[0]->Size(fused.axis());
        _dstStride = dst[0]->Size(fused.axis());
        std::stringstream desc;
        desc << src.size() << "->" << dst.size() << " t=" << _type << " c=" << _count << " s=" << _size;
        this->UsePerfStat(desc.str());
        return true;
    }

    void FusedLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        if (src.size() == 1 && dst.size() == 1)
            ForwardCpu11(src[0]->Data<float>(), dst[0]->Data<float>());
        else if (src.size() == 2 && (dst.size() == 2 || dst.size() == 1))
            ForwardCpu22(src[0]->Data<float>(), src[1]->Data<float>(), dst[0]->Data<float>(), dst.size() == 2 ? dst[1]->Data<float>() : NULL);
        else if (src.size() == 3 && dst.size() == 1)
            ForwardCpu31(src[0]->Data<float>(), src[1]->Data<float>(), src[2]->Data<float>(), dst[0]->Data<float>());
        else
            assert(0);
    }

    void FusedLayer::ForwardCpu11(const float * src, float* dst)
    {
        for (size_t i = 0; i < _num; ++i)
        {
            switch (_type)
            {
            case 0:
                FusedLayerForwardCpu0(src, _t0.bias.Data<float>(), _t0.scale.Data<float>(), _count, _size, dst, _trans);
                break;
            case 1:
            case 7:
                FusedLayerForwardCpu1(src, _t1.bias0.Data<float>(), _t1.scale1.Data<float>(), _t1.bias1.Data<float>(), _count, _size, dst, _trans);
                break;
            case 2:
            case 5:
            case 6:
                FusedLayerForwardCpu2(src, _t2.scale.Data<float>(), _t2.bias.Data<float>(), _count, _size, _t2.slope, dst, _trans);
                break;
            case 3:
                FusedLayerForwardCpu3(src, _t3.bias.Data<float>(), _t3.scale.Data<float>(), _count, _size, dst, _trans);
                break;
            case 4:
                FusedLayerForwardCpu4(src, _t4.bias0.Data<float>(), &_t4.scale1, &_t4.bias1, _count, _size, dst, _trans);
                break;
            case 10:
                ScaleForward32f(src, _t0.scale.Data<float>(), _t0.bias.Data<float>(), _count, 1, _size, dst, _format, 0);
                break;
            case 11:
                FusedLayerForwardCpu11(src, _count*_size, _t11.params, dst);
                break;
            default:
                assert(0);
            }
            src += _srcStride;
            dst += _dstStride;
        }
    }

    void FusedLayer::ForwardCpu22(const float* src0, const float* src1, float* dst0, float* dst1)
    {
        for (size_t i = 0; i < _num; ++i)
        {
            switch (_type)
            {
            case 9:
                FusedLayerForwardCpu9(src0, src1, _t5.scale.Data<float>(), _t5.bias.Data<float>(), _t5.count0, _t5.count1, _size, dst0, dst1, _trans);
                break;
            default:
                assert(0);
            }
            src0 += _t5.count0*_size;
            src1 += _t5.count1*_size;
            dst0 += _dstStride;
            if(dst1)
                dst1 += _dstStride;
        }
    }

    void FusedLayer::ForwardCpu31(const float* src0, const float* src1, const float* src2, float* dst)
    {
        for (size_t i = 0; i < _num; ++i)
        {
            switch (_type)
            {
            case 8:
                FusedLayerForwardCpu8(src0, src1, src2, _count, _size, dst, _trans);
                break;
            default:
                assert(0);
            }
            src0 += _srcStride;
            src1 += _srcStride;
            src2 += _count;
            dst += _dstStride;
        }
    }
}
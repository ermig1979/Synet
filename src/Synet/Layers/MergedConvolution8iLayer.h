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
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/MergedConvolutionLayer.h"

namespace Synet
{
    namespace Detail
    {
    }

    template <class T> class MergedConvolution8iLayer : public MergedConvolutionLayer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;       
        typedef typename Base::TensorPtr TensorPtr;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;

        MergedConvolution8iLayer(const LayerParam & param, QuantizationMethod method)
            : MergedConvolutionLayer<T>(param)
            , _method(method)
        {
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + _mergedConvolution8i.InternalBufferSize() +
                + _weight8i[0].MemoryUsage() + _norm32f[0].MemoryUsage() + _bias32f[0].MemoryUsage()
                + _weight8i[1].MemoryUsage() + _norm32f[1].MemoryUsage() + _bias32f[1].MemoryUsage();
        }

    protected:
        typedef typename MergedConvolutionLayer<T>::AlgParam AlgParam;

        virtual void Reshape(const TensorPtr& src, const TensorPtrs& buf, const TensorPtr& dst)
        {
            AlgParam& a = this->_alg;
            assert(a.add == 0);
            const ConvParam& back = a.conv[a.count - 1];
            _src8u = src->GetType() == TensorType8u;
            _dst8u = dst->GetType() == TensorType8u;
            _dw0 = a.conv[0].IsDepthwise();
            Shape shape = back.DstShape(a.batch);
            if (_dst8u)
                dst->As8u().Reshape(shape, src->Format());
            else
                dst->As32f().Reshape(shape, src->Format());

            _mergedConvolution8i.Init(a.batch, a.conv, a.count, _method);
            if (_mergedConvolution8i.Enable())
            {
                Base::Extend8u(buf, 0, Shp(_mergedConvolution8i.ExternalBufferSize()));
                const float* stats[6] = { 
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->min.data(),
                    this->Stats(0).empty() ? NULL : this->Stats(0)[0]->max.data(),
                    this->Stats(1).empty() ? NULL : this->Stats(1).back()->min.data(),
                    this->Stats(1).empty() ? NULL : this->Stats(1).back()->max.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->min.data(),
                    this->Stats(2).empty() ? NULL : this->Stats(2)[0]->max.data()};
                _mergedConvolution8i.SetParams(a.weight, a.internal, a.bias, a.params, stats);
            }
            else
            {
                if (_dw0)
                {
                    if (_src8u)
                        Base::Extend32f(buf, 0, a.conv[0].SrcShape(1));
                    Base::Extend32f(buf, 1, a.conv[0].DstShape(1));
                    Base::Extend8u(buf, 0, a.conv[1].SrcShape(1));
                    Base::Extend32i(buf, 0, a.conv[1].DstShape(1));
                    a.internal[1] = 1;
                }
                else
                {
                    if (!_src8u)
                        Base::Extend8u(buf, 0, a.conv[0].SrcShape(1));
                    if(!a.conv[0].Is1x1())
                        Base::Extend8u(buf, 1, Shp(a.conv[0].ImgSize()));
                    Base::Extend32i(buf, 0, a.conv[0].DstShape(1));
                    Base::Extend32f(buf, 0, a.conv[0].DstShape(1));
                    if (a.count == 3)
                    {
                        Base::Extend32f(buf, 1, a.conv[1].DstShape(1));
                        Base::Extend8u(buf, 1, a.conv[1].DstShape(1));
                        Base::Extend32i(buf, 0, a.conv[2].DstShape(1));
                        a.internal[2] = 1;
                    }
                    a.internal[0] = 1;
                }
                if(_dst8u)
                    Base::Extend32f(buf, 1, back.DstShape(1));
            //    Quantize();
            }
        }

        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_mergedConvolution8i.Enable())
                _mergedConvolution8i.Forward(src[0]->RawCpuData(), Base::Buf8u(buf, 0), dst[0]->RawCpuData());
            else
            {
                const AlgParam& a = this->_alg;
                float* src32f = _src8u ? (_dw0 ? Base::Buf32f(buf, 0) : NULL) : src[0]->As32f().CpuData();
                uint8_t* src8u = _src8u ? src[0]->As8u().CpuData() : (_dw0 ? NULL : Base::Buf8u(buf, 0));
                uint8_t* buf8u = Base::Buf8u(buf, 1);
                int32_t* sum32i = Base::Buf32i(buf, 0);
                float* buf32f = Base::Buf32f(buf, 0);
                float* dst32f = _dst8u ? Base::Buf32f(buf, 1) : dst[0]->As32f().CpuData();
                uint8_t* dst8u = _dst8u ? dst[0]->As8u().CpuData() : NULL;
                for (size_t b = 0; b < a.batch; ++b)
                {
                    if (!_src8u && !_dw0)
                    {
                        _srcCvt.Convert(src32f, src8u);
                        src32f += a.sSize;
                    }
                    if (_src8u && _dw0)
                    {
                        _srcCvt.Convert(src8u, src32f);
                        src8u += a.sSize;
                    }
                    //if(a.count == 3)
                    //ForwardCpu(src8u, buf8u, buf32f, sum32i, dst32f);
                    if (_src8u && !_dw0)
                        src8u += a.sSize;
                    if (!_src8u && _dw0)
                        src32f += a.sSize;
                    if (_dst8u)
                    {
                        _dstCvt.Convert(dst32f, dst8u);
                        dst8u += a.dSize;
                    }
                    else
                        dst32f += a.dSize;
                }
            }
        }

        void ForwardCpuCdc(uint8_t* src, uint8_t* buf, int32_t* sum, float* dst)
        {

        }

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u, _dw0;
        Converter _srcCvt, _intCvt, _dstCvt;
        Tensor8i _weight8i[2];
        Tensor32f _norm32f[2], _bias32f[2];

        MergedConvolution8i _mergedConvolution8i;
    };
}
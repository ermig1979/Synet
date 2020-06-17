/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar,
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

namespace Synet
{
    template <class T> class AddLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        AddLayer(const LayerParam & param, QuantizationMethod method)
            : Base(param)
            , _method(method)
        {
        }

        virtual bool Can8i() const
        {
            return false;// _method != QuantizationMethodUnknown;
        }

        virtual bool Is8i() const
        {
            return false;// _method != QuantizationMethodUnknown;
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

            if (_src8u)
                Init8i();

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
                    Add8i(src[0]->As8u().CpuData(), src[1]->As8u().CpuData(), dst[0]->As8u().CpuData(), NULL);
                else
                    Add8i(src[0]->As8u().CpuData(), src[1]->As8u().CpuData(), NULL, dst[0]->As32f().CpuData());
            }
            else
                CpuAdd(src[0]->As32f().CpuData(), src[1]->As32f().CpuData(), src[0]->Size(), dst[0]->As32f().CpuData());
        }

        void Init8i()
        {

        }
        
        void Add8i(const uint8_t* src0, const uint8_t* src1, uint8_t dst8u, float* dst32f)
        {

        }

    private:
        QuantizationMethod _method;
        bool _src8u, _dst8u;
        TensorFormat _format;
        size_t _batch, _channels, _height, _width;
    };
}
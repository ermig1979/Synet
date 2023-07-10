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
#include "Synet/Utils/InnerProduct.h"
#include "Synet/Layers/UnaryOperationLayer.h"

namespace Synet
{
    namespace Detail
    {

    }

    template <class T> class RnnGruBdLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        RnnGruBdLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
            _internal[0] = 0;
            _internal[1] = 0;
        }

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            assert(src.size() == 2 && dst.size() == 2 && src[0]->Count() == 3 && src[1]->Count() == 2);
            assert(src[0]->Axis(0) == src[1]->Axis(0) && src[0]->Axis(1) == 1);

            _batch = src[0]->Axis(0);
            _input = src[0]->Axis(2);
            _output = src[1]->Axis(1);

            const Tensors& weight = this->Weight();
            assert(weight.size() == 4);
            assert(weight[0].Axis(1) == _output + _input && weight[0].Axis(0) == weight[1].Axis(0) && weight[1].Axis(0) == 2 * _output);
            assert(weight[2].Axis(1) == _output + _input && weight[2].Axis(0) == weight[3].Axis(0) && weight[3].Axis(0) == _output);

            _innerProduct32f[0].Init(_batch, _input + _output, 2 * _output, 1);
            _innerProduct32f[1].Init(_batch, _input + _output, _output, 1);
            assert(_innerProduct32f[0].Enable() && _innerProduct32f[1].Enable());
            _innerProduct32f[0].SetParams(weight[0].CpuData(), &_internal[0], weight[1].CpuData(), NULL);
            _innerProduct32f[1].SetParams(weight[2].CpuData(), &_internal[1], weight[3].CpuData(), NULL);

            _buffer[0].Reshape(Shp(_batch, _input + _output));
            _buffer[1].Reshape(Shp(_batch, 2 * _output));
            _buffer[2].Reshape(Shp(_batch, _output));

            dst[0]->Reshape(Shp(_batch, 1, _output), src[0]->Format());
            dst[1]->Share(*src[1]);

            std::stringstream desc;
            desc << _batch << "x" << _input << "-" << _output;
            this->UsePerfStat(desc.str(), Flop());
            return true;
        }

        virtual int64_t Flop() const
        {
            return _batch * 2 * (_input + _output) * 3 * _output;
        }

        virtual size_t MemoryUsage() const
        {
            return Base::MemoryUsage() + (_buffer[0].Size() + _buffer[1].Size() + _buffer[2].Size() + 
                _innerProduct32f[0].InternalBufferSize() + _innerProduct32f[1].InternalBufferSize()) * sizeof(float);
        }

        virtual void CompactWeight()
        {
            if (_internal[0])
                ((Tensor&)this->Weight()[0]).Clear();
            if (_internal[1])
                ((Tensor&)this->Weight()[2]).Clear();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const float* src0 = src[0]->CpuData();
            const float* src1 = src[1]->CpuData();
            float* dst0 = dst[0]->CpuData();
            float* dst1 = dst[1]->CpuData();
            float* buf00 = _buffer[0].CpuData();
            float* buf01 = buf00 + _input;
            float* buf10 = _buffer[1].CpuData();
            float* buf11 = buf10 + _output;
            float* buf2 = _buffer[2].CpuData();

            if (_batch == 1)
            {
                memcpy(buf00, src0, _input * sizeof(float));
                memcpy(buf01, src1, _output * sizeof(float));

                _innerProduct32f[0].Forward(buf00, buf10);
                CpuSigmoid(buf10, _output * 2, buf10);

                for (size_t i = 0; i < _output; ++i)
                    buf01[i] = buf10[i]*src1[i];

                _innerProduct32f[1].Forward(buf00, buf2);
                Detail::UnaryOperationLayerForward(buf2, _output, UnaryOperationTypeTanh, buf2);

                for (size_t i = 0; i < _output; ++i)
                    dst0[i] = (1.0f - buf11[i]) * buf2[i] + src1[i]* buf11[i];

                for (size_t i = 0; i < _output; ++i)
                    dst1[i] = dst0[i];
            }
            else
                assert(0);
        }

        int _internal[2];
        InnerProduct32f _innerProduct32f[2];
        Tensor _buffer[3];
        size_t _batch, _input, _output;
    };
}
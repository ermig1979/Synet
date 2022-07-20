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
#include "Synet/Utils/Gemm.h"
#include "Synet/Layers/UnaryOperationLayer.h"

namespace Synet
{
    template <class T> class LstmLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::Tensors Tensors;
        typedef typename Base::TensorPtrs TensorPtrs;

        LstmLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 3 && src[0]->Count() == 3 && src[1]->Count() == 3 && src[2]->Count() == 3);
            const LstmParam & param = this->Param().lstm();
            const Tensors & weight = this->Weight();

            _hidS = param.hiddenSize();
            _dir = param.direction();
            _dirS = _dir == LstmDirectionTypeBidirectional ? 2 : 1;

            const Tensor & x = *src[0];
            _seqS = x.Axis(0);
            _batch = x.Axis(1);
            _srcS = x.Axis(2);
            const Tensor & h = *src[1];
            const Tensor & c = *src[2];

            _w0 = weight[0].CpuData(Shp(0, 0, 0));
            _w1 = _dirS > 1 ? weight[0].CpuData(Shp(1, 0, 0)) : NULL;

            _u0 = weight[1].CpuData(Shp(0, 0, 0));
            _u1 = _dirS > 1 ? weight[1].CpuData(Shp(1, 0, 0)) : NULL;

            _b0 = weight[2].CpuData(Shp(0, 0));
            _b1 = _dirS > 1 ? weight[2].CpuData(Shp(1, 0)) : NULL;

            buf[0]->Extend(Shp(4, _hidS));
            buf[1]->Extend(Shp(_seqS, _batch, _hidS));

            dst[0]->Reshape(Shp(_seqS, _dirS, _batch, _hidS));

            std::stringstream desc;
            desc << _seqS << "x" << _dirS << "x" << _batch << "x" << _hidS;
            this->UsePerfStat(desc.str(), Flop());
        }

        virtual int64_t Flop() const
        {
            return _batch * _dirS * _seqS * _hidS * 4 * (_srcS + _hidS) * 2;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T * x = src[0]->CpuData();
            const T * h0 = src[1]->CpuData(Shp(0, 0, 0));
            const T * c0 = src[2]->CpuData(Shp(0, 0, 0));
            T * b = buf[0]->CpuData();
            T * c = buf[1]->CpuData();
            T * h = dst[0]->CpuData();
            switch (_dir)
            {
            case LstmDirectionTypeForward:
                ForwardOneDir(x, h0, c0, _w0, _u0, _b0, b, c, h, false);
                break;
            case LstmDirectionTypeReverse:
                ForwardOneDir(x, h0, c0, _w0, _u0, _b0, b, c, h, true);
                break;
            case LstmDirectionTypeBidirectional:
                ForwardOneDir(x, h0, c0, _w0, _u0, _b0, b, c, h, false);
                const T * h1 = src[1]->CpuData(Shp(1, 0, 0));
                const T * c1 = src[2]->CpuData(Shp(1, 0, 0));
                ForwardOneDir(x, h1, c1, _w1, _u1, _b1, b, c, h + _batch * _hidS, true);
                break;
            }
        }

        virtual void ForwardOneDir(const T * x, const T * h0, const T * c0, const T * w, const T * u, const T * bias, T * buf, T * c, T * h, bool rev)
        {
            T * ib = buf + 0 * _hidS;
            T * ob = buf + 1 * _hidS;
            T * fb = buf + 2 * _hidS;
            T * cb = buf + 3 * _hidS;
            ptrdiff_t xStep = (rev ? -1 : 1) * _batch * _srcS;
            ptrdiff_t hStep = (rev ? -1 : 1) * _batch * _hidS * _dirS;
            size_t cStep = _batch * _hidS;
            const T * xBeg = x + (rev ? (1 - _seqS) * xStep : 0);
            T * hBeg = h + (rev ? (1 - _seqS) * hStep : 0);
            for (size_t i = 0; i < _seqS; ++i)
            {
                const T * xCurr = xBeg + i * xStep;
                T * hCurr = hBeg + i * hStep;
                const T * hPrev = i ? hCurr - hStep : h0;
                T * cCurr = c + i * cStep;
                const T * cPrev = i ? cCurr - cStep : c0;
                for (size_t b = 0; b < _batch; ++b)
                {
                    for (size_t k = 0; k < _hidS * 4; ++k)
                        buf[k] = 0;

                    Detail::CpuGemvN(_hidS * 4, _srcS, 1.0f, w, xCurr, buf);

                    Detail::CpuGemvN(_hidS * 4, _hidS, 1.0f, u, hPrev, buf);

                    for (size_t k = 0; k < _hidS * 4; ++k)
                        buf[k] = buf[k] + bias[k] + bias[_hidS * 4 + k];

                    CpuSigmoid(buf, 3 * _hidS, buf);

                    Detail::UnaryOperationLayerForward(cb, _hidS, UnaryOperationTypeTanh, cb);

                    //for (size_t k = 0; k < _hidS; ++k)
                    //{
                    //    cCurr[k] = fb[k] * cPrev[k] + ib[k] * cb[k];
                    //    hCurr[k] = ob[k] * ::tanh(cCurr[k]);
                    //}

                    for (size_t k = 0; k < _hidS; ++k)
                        cCurr[k] = fb[k] * cPrev[k] + ib[k] * cb[k];

                    Detail::UnaryOperationLayerForward(cCurr, _hidS, UnaryOperationTypeTanh, hCurr);

                    for (size_t k = 0; k < _hidS; ++k)
                        hCurr[k] *= ob[k];


                    xCurr += _srcS;
                    hPrev += _hidS;
                    cPrev += _hidS;
                    hCurr += _hidS;
                    cCurr += _hidS;
                }
            }
        }

        LstmDirectionType _dir;
        size_t _batch, _seqS, _srcS, _hidS, _dirS;
        const Type *_w0, *_w1, *_u0, *_u1, *_b0, *_b1;
    };
}
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
#include "Synet/Layers/Recurrent/LstmLayer.h"
#include "Synet/Layers/Math/UnaryOperationLayer.h"
#include "Synet/Utils/Gemm.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    LstmLayer::LstmLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
        for(int i = 0; i < IPS; ++i)
            _internal[i] = 0;
    }

    bool LstmLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 3 || dst.size() != 1)
            SYNET_ERROR("LstmLayer supports 3 input1 and 1 output!");
        if (src[0]->GetType() != TensorType32f || src[1]->GetType() != TensorType32f || src[2]->GetType() != TensorType32f)
            SYNET_ERROR("LstmLayer supports only FP32 type!");
        if (src[0]->Count() != 3 || src[1]->Count() != 3 || src[2]->Count() != 3)
            SYNET_ERROR("LstmLayer supports only 3D input tensors!");

        const LstmParam & param = this->Param().lstm();
        const Tensors & weight = this->Weight();

        _hidS = param.hiddenSize();
        _hidS4 = _hidS * 4;
        _dir = param.direction();
        _dirS = _dir == LstmDirectionTypeBidirectional ? 2 : 1;

        const Tensor & x = *src[0];
        _seqS = x.Axis(0);
        _batch = x.Axis(1);
        _srcS = x.Axis(2);
        const Tensor & h = *src[1];
        const Tensor & c = *src[2];

        _w0 = weight[0].Data<float>(Shp(0, 0, 0));
        _w1 = _dirS > 1 ? weight[0].Data<float>(Shp(1, 0, 0)) : NULL;

        _r0 = weight[1].Data<float>(Shp(0, 0, 0));
        _r1 = _dirS > 1 ? weight[1].Data<float>(Shp(1, 0, 0)) : NULL;

        _b0 = weight[2].Data<float>(Shp(0, 0));
        _b1 = _dirS > 1 ? weight[2].Data<float>(Shp(1, 0)) : NULL;

        buf[0]->Extend(TensorType32f, Shp(_seqS + 2, _batch, _hidS4));

        dst[0]->Reshape(TensorType32f, Shp(_seqS, _dirS, _batch, _hidS), TensorFormatUnknown);

        _innerProduct32f[0].Init(_seqS * _batch, _srcS, _hidS4, 1);
        _innerProduct32f[0].SetParams(_w0, &_internal[0], NULL, NULL);
        _innerProduct32f[2].Init(1, _hidS, _hidS4, 1);
        _innerProduct32f[2].SetParams(_r0, &_internal[2], NULL, NULL);
        if (_dirS > 1)
        {
            _innerProduct32f[1].Init(_seqS * _batch, _srcS, _hidS4, 1);
            _innerProduct32f[1].SetParams(_w1, &_internal[1], NULL, NULL);
            _innerProduct32f[3].Init(1, _hidS, _hidS4, 1);
            _innerProduct32f[3].SetParams(_r1, &_internal[3], NULL, NULL);
        }

        std::stringstream desc;
        desc << _seqS << "x" << _dirS << "x" << _batch << "x" << _hidS;
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }

    int64_t LstmLayer::Flop() const
    {
        return _batch * _dirS * _seqS * _hidS4 * (_srcS + _hidS) * 2;
    }

    size_t LstmLayer::MemoryUsage() const
    {
        size_t size = 0;
        for (int i = 0; i < IPS; ++i)
            size += _innerProduct32f[i].InternalBufferSize();
        return Layer::MemoryUsage() + size * sizeof(float);
    }

    void LstmLayer::CompactWeight()
    {
        if (_internal[0] && _internal[1])
            ((Tensor&)this->Weight()[0]).Clear();
        if (_internal[2] && _internal[3])
            ((Tensor&)this->Weight()[1]).Clear();
    }

    void LstmLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        const float * x = src[0]->Data<float>();
        const float * h0 = src[1]->Data<float>(Shp(0, 0, 0));
        const float * c0 = src[2]->Data<float>(Shp(0, 0, 0));
        float * pBuf = buf[0]->Data<float>();
        float * h = dst[0]->Data<float>();
        switch (_dir)
        {
        case LstmDirectionTypeForward:
            ForwardOneDir(x, h0, c0, _w0, _r0, _b0, pBuf, h, false);
            break;
        case LstmDirectionTypeReverse:
            ForwardOneDir(x, h0, c0, _w0, _r0, _b0, pBuf, h, true);
            break;
        case LstmDirectionTypeBidirectional:
            ForwardOneDir(x, h0, c0, _w0, _r0, _b0, pBuf, h, false);
            const float * h1 = src[1]->Data<float>(Shp(1, 0, 0));
            const float * c1 = src[2]->Data<float>(Shp(1, 0, 0));
            ForwardOneDir(x, h1, c1, _w1, _r1, _b1, pBuf, h + _batch * _hidS, true);
            break;
        }
    }

    void LstmLayer::ForwardOneDir(const float * x, const float * h0, const float * c0, const float * w, const float * r, const float * bias, float * buf, float * h, bool rev)
    {
        ptrdiff_t xStep = (rev ? -1 : 1) * _batch * _hidS4;
        ptrdiff_t hStep = (rev ? -1 : 1) * _batch * _hidS * _dirS;
        size_t cStep = _batch * _hidS;

        float * xBuf = buf;
        float * cBeg = xBuf + _seqS * _batch * _hidS4;
        float * buf0 = cBeg + 2 * _batch * _hidS;
        float * ib = buf0 + 0 * _hidS;
        float * ob = buf0 + 1 * _hidS;
        float * fb = buf0 + 2 * _hidS;
        float * cb = buf0 + 3 * _hidS;
        float * xBeg = xBuf + (rev ? (1 - _seqS) * xStep : 0);
        float * hBeg = h + (rev ? (1 - _seqS) * hStep : 0);
        size_t idx = (rev && _dirS == 2) ? 1 : 0;

        memcpy(cBeg + cStep, c0, cStep * sizeof(float));
        _innerProduct32f[0 + idx].Forward(x, xBuf);
        for (size_t i = 0; i < _seqS; ++i)
        {
            float * xCurr = xBeg + i * xStep;
            float * hCurr = hBeg + i * hStep;
            const float * hPrev = i ? hCurr - hStep : h0;
            float * cCurr = cBeg + (i&1) * cStep;
            float * cPrev = cBeg + ((i - 1)&1) * cStep;
            for (size_t b = 0; b < _batch; ++b)
            {
                _innerProduct32f[2 + idx].Forward(hPrev, buf0);

                for (size_t k = 0; k < _hidS4; ++k)
                    buf0[k] = xCurr[k] + buf0[k] + bias[k] + bias[_hidS4 + k];

                Sigmoid32f(buf0, 3 * _hidS, buf0);

                UnaryOperation32f(cb, _hidS, UnaryOperationTypeTanh, cb);

                for (size_t k = 0; k < _hidS; ++k)
                    cCurr[k] = fb[k] * cPrev[k] + ib[k] * cb[k];

                UnaryOperation32f(cCurr, _hidS, UnaryOperationTypeTanh, hCurr);

                for (size_t k = 0; k < _hidS; ++k)
                    hCurr[k] *= ob[k];

                xCurr += _hidS4;
                hPrev += _hidS;
                cPrev += _hidS;
                hCurr += _hidS;
                cCurr += _hidS;
            }
        }
    }
}
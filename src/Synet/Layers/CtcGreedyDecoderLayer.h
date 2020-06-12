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
#include "Synet/Utils/Math.h"

namespace Synet
{
    namespace Detail
    {
    }

    template <class T> class CtcGreedyDecoderLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;

        CtcGreedyDecoderLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src.size() == 2 && src[0]->Count() == 3);
            assert(src[1]->Count() == 2 && src[0]->Axis(0) == src[1]->Axis(0) && src[0]->Axis(1) == src[1]->Axis(1));

            _t = src[0]->Axis(0);
            _n = src[0]->Axis(1);
            _c = src[0]->Axis(2);
            dst[0]->Reshape({ size_t(1), _t, _n, size_t(1) });
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const T * pProbabilities = src[0]->CpuData();
            const T * pSequenceIndicators = src[1]->CpuData();
            T * pOutputSequences = dst[0]->CpuData();

            for (size_t i = 0, n = _t*_n; i < n; i++)
                pOutputSequences[i] = T(-1);

            for (size_t n = 0; n < _n; ++n) 
            {
                size_t prevClassIndex = -1;
                size_t outputIndex = n*_t;
                for (size_t t = 0; t < _t; ++t) 
                {
                    size_t maxClassIndex = 0;
                    const T * pProbs = pProbabilities + t*_c*_n + n*_c;
                    T maxProb = pProbs[0];
                    for (size_t c = 1; c < _c; ++c)
                    {
                        if (pProbs[c] > maxProb) 
                        {
                            maxClassIndex = size_t(c);
                            maxProb = pProbs[c];
                        }
                    }
                    if (maxClassIndex < size_t(_c) - 1 && maxClassIndex != prevClassIndex)
                    {
                        pOutputSequences[outputIndex] = T(maxClassIndex);
                        outputIndex++;
                    }
                    prevClassIndex = maxClassIndex;
                    if (t + 1 == _t || pSequenceIndicators[(t + 1)*_n + n] == 0)
                        break;
                }
            }
        }

    private:
        size_t _t, _n, _c;
    };
}
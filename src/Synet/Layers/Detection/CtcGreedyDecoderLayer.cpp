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

#include "Synet/Layers/Detection/CtcGreedyDecoderLayer.h"

namespace Synet
{
    CtcGreedyDecoderLayer::CtcGreedyDecoderLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool CtcGreedyDecoderLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("CtcGreedyDecoderLayer supports only 2 inputs and 1 output!");
        if(src[0]->Count() != 3 || src[1]->Count() != 2 || src[0]->Axis(0) != src[1]->Axis(0) || src[0]->Axis(1) != src[1]->Axis(1))
            SYNET_ERROR("CtcGreedyDecoderLayer has unsupported input shapes!");
        if (src[0]->GetType() != TensorType32f || src[1]->GetType() != TensorType32f)
            SYNET_ERROR("CtcGreedyDecoderLayer has unsupported input types!");

        _t = src[0]->Axis(0);
        _n = src[0]->Axis(1);
        _c = src[0]->Axis(2);
        dst[0]->Reshape(TensorType32f, Shp(1, _t, _n, 1), TensorFormatUnknown);
        this->UsePerfStat();
        return true;
    }

    void CtcGreedyDecoderLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        const float * pProbabilities = src[0]->Data<float>();
        const float* pSequenceIndicators = src[1]->Data<float>();
        float* pOutputSequences = dst[0]->Data<float>();

        for (size_t i = 0, n = _t * _n; i < n; i++)
            pOutputSequences[i] = -1.0f;

        for (size_t n = 0; n < _n; ++n) 
        {
            size_t prevClassIndex = -1;
            size_t outputIndex = n * _t;
            for (size_t t = 0; t < _t; ++t) 
            {
                size_t maxClassIndex = 0;
                const float * pProbs = pProbabilities + t * _c * _n + n * _c;
                float maxProb = pProbs[0];
                for (size_t c = 1; c < _c; ++c)
                {
                    if (pProbs[c] > maxProb) 
                    {
                        maxClassIndex = c;
                        maxProb = pProbs[c];
                    }
                }
                if (maxClassIndex < _c - 1 && maxClassIndex != prevClassIndex)
                {
                    pOutputSequences[outputIndex] = float(maxClassIndex);
                    outputIndex++;
                }
                prevClassIndex = maxClassIndex;
                if (t + 1 == _t || pSequenceIndicators[(t + 1) * _n + n] == 0)
                    break;
            }
        }
    }
}
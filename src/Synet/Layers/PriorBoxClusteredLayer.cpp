/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layers/PriorBoxClusteredLayer.h"

namespace Synet
{
    PriorBoxClusteredLayer::PriorBoxClusteredLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool PriorBoxClusteredLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        const PriorBoxClusteredParam & param = this->Param().priorBoxClustered();

        _heights = param.heights();
        _widths = param.widths();
        _clip = param.clip();
        _variance = param.variance();

        _numPriors = _widths.size();
        if (_variance.empty())
            _variance.push_back(0.1f);

        size_t layerH, layerW;
        if (src[0]->GetType() == TensorType64i)
        {
            const int64_t* src0 = src[0]->As64i().CpuData();
            const int64_t* src1 = src[1]->As64i().CpuData();
            layerH = src0[0];
            layerW = src0[1];
            _imgH = param.imgH() ? param.imgH() : src1[0];
            _imgW = param.imgW() ? param.imgW() : src1[1];
        }
        else
        {
            if (src[0]->Format() == TensorFormatNhwc)
            {
                layerH = src[0]->Axis(-3);
                layerW = src[0]->Axis(-2);
                _imgH = param.imgH() ? param.imgH() : src[1]->Axis(-3);
                _imgW = param.imgW() ? param.imgW() : src[1]->Axis(-2);
            }
            else
            {
                layerH = src[0]->Axis(-2);
                layerW = src[0]->Axis(-1);
                _imgH = param.imgH() ? param.imgH() : src[1]->Axis(-2);
                _imgW = param.imgW() ? param.imgW() : src[1]->Axis(-1);
            }
        }

        _stepH = param.stepH() == 0 ? param.step() : param.stepH();
        _stepW = param.stepW() == 0 ? param.step() : param.stepW();
        if (_stepH == 0 && _stepW == 0) 
        {
            _stepH = float(_imgH) / layerH;
            _stepW = float(_imgW) / layerW;
        }
        _offset = param.offset();

        Shape shape(3);
        shape[0] = 1;
        shape[1] = 2;
        shape[2] = layerW * layerH * _numPriors * 4;
        dst[0]->Reshape(TensorType32f, shape, TensorFormatUnknown);
        float * pDst0 = dst[0]->Data<float>({ 0, 0, 0 });
        float * pDst1 = dst[0]->Data<float>({ 0, 1, 0 });
        size_t varSize = _variance.size();

        for (size_t h = 0; h < layerH; ++h) 
        {
            for (size_t w = 0; w < layerW; ++w) 
            {
                float centerY = (h + _offset) * _stepH;
                float centerX = (w + _offset) * _stepW;

                for (size_t s = 0; s < _numPriors; ++s) 
                {
                    float boxW = _widths[s];
                    float boxH = _heights[s];

                    float xmin = (centerX - boxW / 2.0f) / _imgW;
                    float ymin = (centerY - boxH / 2.0f) / _imgH;
                    float xmax = (centerX + boxW / 2.0f) / _imgW;
                    float ymax = (centerY + boxH / 2.0f) / _imgH;
                    if (_clip) 
                    {
                        xmin = Min(Max(xmin, 0.0f), 1.0f);
                        ymin = Min(Max(ymin, 0.0f), 1.0f);
                        xmax = Min(Max(xmax, 0.0f), 1.0f);
                        ymax = Min(Max(ymax, 0.0f), 1.0f);
                    }
                    pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 0] = xmin;
                    pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 1] = ymin;
                    pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 2] = xmax;
                    pDst0[h * layerW * _numPriors * 4 + w * _numPriors * 4 + s * 4 + 3] = ymax;
                    for (size_t j = 0; j < varSize; j++)
                        pDst1[h * layerW * _numPriors * varSize + w * _numPriors * varSize +  s * varSize + j] = _variance[j];
                }
            }
        }
        dst[0]->SetConst(true);
        _const = true;
        return true;
    }

    void PriorBoxClusteredLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
    }
}
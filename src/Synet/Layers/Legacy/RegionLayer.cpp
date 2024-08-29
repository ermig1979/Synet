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

#include "Synet/Layers/SoftmaxLayer.h"
#include "Synet/Layers/Legacy/RegionLayer.h"
#include "Synet/Utils/Activation.h"

namespace Synet
{
    template<class T> void FlattenCpu(const T * src, size_t size, size_t layers, size_t batch, T * dst)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t c = 0; c < layers; ++c)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    size_t i1 = b*layers*size + c*size + i;
                    size_t i2 = b*layers*size + i*layers + c;
                    dst[i2] = src[i1];
                }
            }
        }
    }

    //-------------------------------------------------------------------------------------------------

    RegionLayer::RegionLayer(const LayerParam & param, Context* context)
        : Base(param, context)
    {
    }

    bool RegionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("RegionLayer supports 1 input and 1 output!");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("RegionLayer supports only FP32 input!");

        const RegionParam & param = this->Param().region();
        _coords = param.coords();
        _classes = param.classes();
        _num = param.num();
        _softmax = param.softmax();
        _anchors.resize(param.anchors().size());
        for (size_t i = 0; i < param.anchors().size(); ++i)
            _anchors[i] = param.anchors()[i];
        _classfix = 0;
        if (src[0]->Axis(1) != _num * (_coords + _classes + 1))
            SYNET_ERROR("RegionLayer wrong parameters!");
        dst[0]->Reshape(TensorType32f, src[0]->Shape(), TensorFormatUnknown);
        this->UsePerfStat();
        return true;
    }

    void RegionLayer::GetRegions(const TensorPtrs & src, Type threshold, Regions & dst) const
    {
        SYNET_PERF_FUNC();
        dst.clear();
        const float * pPredict = src[0]->Data<float>();
        size_t height = src[0]->Axis(2);
        size_t width = src[0]->Axis(3);
        size_t outputs = src[0]->Size(1);
        for (size_t i = 0; i < width*height; ++i) 
        {
            size_t row = i / height;
            size_t col = i % width;
            for (size_t n = 0; n < _num; ++n) 
            {
                size_t index = i*_num + n;
                size_t predictIndex = index * (_classes + 5) + 4;
                Type scale = pPredict[predictIndex];
                if (_classfix == -1 && scale < Type(0.5)) 
                    scale = Type(0);
                size_t regionIndex = index * (_classes + 5);
                Region r;
                r.x = (col + CpuSigmoid(pPredict[regionIndex + 0])) / width;
                r.y = (row + CpuSigmoid(pPredict[regionIndex + 1])) / height;
                r.w = ::exp(pPredict[regionIndex + 2]) * _anchors[2 * n] / width;
                r.h = ::exp(pPredict[regionIndex + 3]) * _anchors[2 * n + 1] / height;
                size_t classIndex = index * (_classes + 5) + 5;
                for (size_t id = 0; id < _classes; ++id)
                {
                    Type prob = scale*pPredict[classIndex + id];
                    if (prob > threshold)
                    {
                        r.prob = prob;
                        r.id = (int)id;
                        dst.push_back(r);
                    }
                }
            }
        }
    }

    void RegionLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        size_t size = _coords + _classes + 1;
        size_t batch = src[0]->Axis(0);
        size_t height = src[0]->Axis(2);
        size_t width = src[0]->Axis(3);
        size_t outputs = src[0]->Size(1);
        float * pDst = dst[0]->Data<float>();
        FlattenCpu(src[0]->Data<float>(), width*height, size*_num, batch, pDst);
        for (size_t b = 0; b < batch; ++b) 
        {
            for (size_t i = 0; i < height*width*_num; ++i)
            {
                size_t index = size*i + b*outputs;
                CpuSigmoid(pDst + index + 4, 1, pDst + index + 4);
            }
        }
        if (_softmax) 
        {
            for (size_t b = 0; b < batch; ++b) 
            {
                for (size_t i = 0; i < height*width*_num; ++i)
                {
                    size_t index = size*i + b*outputs;
                    SoftmaxLayerForward(pDst + index + 5, 1, _classes, 1, pDst + index + 5);
                }
            }
        }
    }
}
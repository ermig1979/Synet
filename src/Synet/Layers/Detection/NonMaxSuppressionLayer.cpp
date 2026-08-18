/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Detection/NonMaxSuppressionLayer.h"

namespace Synet
{
    struct SelectedIndex 
    {
        int64_t batchIndex = 0;
        int64_t classIndex = 0;
        int64_t boxIndex = 0;

        SYNET_INLINE SelectedIndex(int64_t batchIdx = 0, int64_t classIdx = 0, int64_t boxIdx = 0)
            : batchIndex(batchIdx)
            , classIndex(classIdx)
            , boxIndex(boxIdx)
        {
        }
    };

    struct BoxInfo
    {
        float score;
        int64_t index;

        SYNET_INLINE BoxInfo(float scr = 0.0f, int64_t idx = 0)
            : score(scr)
            , index(idx) 
        {
        }

        SYNET_INLINE bool operator<(const BoxInfo& rhs) const 
        {
            return score < rhs.score || (score == rhs.score && index > rhs.index);
        }
    };

    //-------------------------------------------------------------------------------------------------

    NonMaxSuppressionLayer::NonMaxSuppressionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool NonMaxSuppressionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("NonMaxSuppressionLayer supports only 2 inputs and 1 output!");
        if (src[0]->Count() != 3 || src[1]->Count() != 3)
            SYNET_ERROR("NonMaxSuppressionLayer supports only 3D input tensors!");
        if (src[0]->GetType() != TensorType32f || src[1]->GetType() != TensorType32f)
            SYNET_ERROR("NonMaxSuppressionLayer supports only FP32 input tensors!");
        if (src[0]->Axis(0) != src[1]->Axis(0) || src[0]->Axis(1) != src[1]->Axis(2) || src[0]->Axis(2) != 4)
            SYNET_ERROR("NonMaxSuppressionLayer: check input tensors shapes!");

        const NonMaxSuppressionParam& param = this->Param().nonMaxSuppression();
        _maxOutputBoxesPerClass = param.maxOutputBoxesPerClass();
        _threshold = param.scoreThreshold();
        _overlap = param.iouThreshold();
        _batch = src[0]->Axis(0);
        _size = src[0]->Axis(1);
        _classNum = src[1]->Axis(1);

        dst[0]->Reshape(TensorType64i, Shp(_maxOutputBoxesPerClass, 3));
        this->UsePerfStat();

        return true;
    }

    void NonMaxSuppressionLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
    }
}
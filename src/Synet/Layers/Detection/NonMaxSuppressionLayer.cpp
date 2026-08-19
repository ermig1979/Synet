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
    };

    //-------------------------------------------------------------------------------------------------

    SYNET_INLINE void MaxMin(float lhs, float rhs, float& min, float& max) 
    {
        if (lhs >= rhs) 
        {
            min = rhs;
            max = lhs;
        }
        else 
        {
            min = lhs;
            max = rhs;
        }
    }

    SYNET_INLINE bool SuppressByIOU(const float* boxes_data, int64_t box_index1, int64_t box_index2, BoxEncodingType boxEncoding, float iou_threshold)
    {
        float x1_min{};
        float y1_min{};
        float x1_max{};
        float y1_max{};
        float x2_min{};
        float y2_min{};
        float x2_max{};
        float y2_max{};
        float intersection_x_min{};
        float intersection_x_max{};
        float intersection_y_min{};
        float intersection_y_max{};

        const float* box1 = boxes_data + 4 * box_index1;
        const float* box2 = boxes_data + 4 * box_index2;
        if (boxEncoding == BoxEncodingTypeCorner)
        {
            // boxes data format [y1, x1, y2, x2],
            MaxMin(box1[1], box1[3], x1_min, x1_max);
            MaxMin(box2[1], box2[3], x2_min, x2_max);

            intersection_x_min = Max(x1_min, x2_min);
            intersection_x_max = Min(x1_max, x2_max);
            if (intersection_x_max <= intersection_x_min)
                return false;

            MaxMin(box1[0], box1[2], y1_min, y1_max);
            MaxMin(box2[0], box2[2], y2_min, y2_max);
            intersection_y_min = Max(y1_min, y2_min);
            intersection_y_max = Min(y1_max, y2_max);
            if (intersection_y_max <= intersection_y_min)
                return false;
        }
        else 
        {
            // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
            float box1_width_half = box1[2] / 2;
            float box1_height_half = box1[3] / 2;
            float box2_width_half = box2[2] / 2;
            float box2_height_half = box2[3] / 2;

            x1_min = box1[0] - box1_width_half;
            x1_max = box1[0] + box1_width_half;
            x2_min = box2[0] - box2_width_half;
            x2_max = box2[0] + box2_width_half;

            intersection_x_min = Max(x1_min, x2_min);
            intersection_x_max = Min(x1_max, x2_max);
            if (intersection_x_max <= intersection_x_min)
                return false;

            y1_min = box1[1] - box1_height_half;
            y1_max = box1[1] + box1_height_half;
            y2_min = box2[1] - box2_height_half;
            y2_max = box2[1] + box2_height_half;

            intersection_y_min = Max(y1_min, y2_min);
            intersection_y_max = Min(y1_max, y2_max);
            if (intersection_y_max <= intersection_y_min)
                return false;
        }

        const float intersection_area = (intersection_x_max - intersection_x_min) *
            (intersection_y_max - intersection_y_min);

        if (intersection_area <= .0f) 
            return false;

        const float area1 = (x1_max - x1_min) * (y1_max - y1_min);
        const float area2 = (x2_max - x2_min) * (y2_max - y2_min);
        const float union_area = area1 + area2 - intersection_area;

        if (area1 <= .0f || area2 <= .0f || union_area <= .0f)
            return false;

        const float intersection_over_union = intersection_area / union_area;

        return intersection_over_union > iou_threshold;
    }

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
        _boxEncoding = param.boxEncoding();
        _maxOutputBoxesPerClass = param.maxOutputBoxesPerClass();
        _threshold = param.scoreThreshold();
        _overlap = param.iouThreshold();
        _batch = src[0]->Axis(0);
        _size = src[0]->Axis(1);
        _classNum = src[1]->Axis(1);

        Layer::Extend8u(buf, 0, Shp((_size + _maxOutputBoxesPerClass)*sizeof(BoxInfo)));

        dst[0]->Reshape(TensorType64i, Shp(_maxOutputBoxesPerClass, 3));
        this->UsePerfStat();

        return true;
    }

    void NonMaxSuppressionLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        BoxInfo* candidates = (BoxInfo*)Layer::Buf8u(buf, 0), *selected = candidates + _size;
        std::vector<SelectedIndex> selectedIndices;
        for (size_t b = 0; b < _batch; ++b)
        {
            for (size_t c = 0; c < _classNum; ++c)
            {
                const float* boxes = src[0]->Data<float>(Shp(b, 0, 0));
                const float* scores = src[1]->Data<float>(Shp(b, c, 0));
                size_t detected = 0;
                for (size_t i = 0; i < _size; ++i, ++scores)
                {
                    if (*scores > _threshold)
                        candidates[detected++] = BoxInfo(*scores, i);
                }
                std::sort(candidates, candidates + detected, [](const BoxInfo& a, const BoxInfo& b) -> bool
                    { return a.score > b.score || (a.score == b.score && a.index < b.index); });

                size_t selectNumber = 0;
                for (size_t i = 0; i < detected && selectNumber < _maxOutputBoxesPerClass; ++i)
                {
                    bool select = true;
                    for (size_t j = 0; j < selectNumber; ++j)
                    {
                        if (SuppressByIOU(boxes, candidates[i].index, selected[j].index, _boxEncoding, _overlap))
                        {
                            select = false;
                            break;
                        }
                    }

                    if (select) 
                    {
                        selected[selectNumber] = candidates[i];
                        selectedIndices.emplace_back(b, c, candidates[i].index);
                        selectNumber++;
                    }
                }
            }
        }
        if (1)
        {
            std::sort(selectedIndices.begin(), selectedIndices.end(), [src](const SelectedIndex& a, const SelectedIndex& b) -> bool
                { 
                    float aScore = src[1]->Data<float>(Shp(a.batchIndex, a.classIndex, a.boxIndex))[0];
                    float bScore = src[1]->Data<float>(Shp(b.batchIndex, b.classIndex, b.boxIndex))[0];
                    return aScore > bScore; 
                });
        }
        size_t num = Min(_maxOutputBoxesPerClass, selectedIndices.size());
        memcpy(dst[0]->RawData(), selectedIndices.data(), num * sizeof(SelectedIndex));
    }
}
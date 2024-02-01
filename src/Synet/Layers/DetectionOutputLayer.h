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

#pragma once

#include "Synet/Layer.h"

namespace Synet
{
    class DetectionOutputLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;

        DetectionOutputLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual size_t MemoryUsage() const;

        void GetRegions(const TensorPtrs& src, float threshold, Regions& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        struct NormalizedBBox
        {
            float xmin;
            float ymin;
            float xmax;
            float ymax;
            int32_t label;
            bool difficult;
            float score;
            float size;

            NormalizedBBox()
                : xmin(0), ymin(0), xmax(0), ymax(0)
                , label(0), difficult(false), score(0), size(0)
            {
            }
        };

        typedef typename Base::Tensor Tensor;
        typedef std::map<int, std::vector<NormalizedBBox>> LabelBBox;
        typedef std::vector<LabelBBox> LabelBBoxes;
        typedef std::map<int, Floats> LabelPred;
        typedef std::vector<LabelPred> LabelPreds;
        typedef std::vector<NormalizedBBox> NormalizedBBoxes;
        typedef std::vector<float*> Variances;
        typedef std::map<int, Index> IndexMap;
        typedef std::vector<IndexMap> IndexMaps;
        typedef std::pair<float, size_t> ScoreIndex;
        typedef std::vector<ScoreIndex> ScoreIndeces;
        typedef std::pair<size_t, size_t> IndexPair;
        typedef std::pair<float, IndexPair> ScoreIndexPair;
        typedef std::vector<ScoreIndexPair> ScoreIndexPairs;

        bool _shareLocation, _varianceEncodedInTarget, _keepMaxClassScoresOnly, _clip;
        size_t _numClasses, _numLocClasses, _numPriors;
        ptrdiff_t _backgroundLabelId, _keepTopK, _topK;
        PriorBoxCodeType _codeType;
        float _confidenceThreshold, _nmsThreshold, _eta;
        NormalizedBBoxes _priorBboxes;
        Variances _priorVariances;
        LabelBBoxes _allLocPreds, _allDecodeBboxes;
        LabelPreds _allConfScores;

        void GetLocPredictions(const float* pLoc, size_t num, size_t numPredsPerClass, size_t numLocClasses, bool shareLocation, LabelBBoxes& locPreds);

        void GetConfidenceScores(const float* pConf, size_t num, size_t numPredsPerClass, size_t numClasses, LabelPreds& confPreds, bool keepMaxScoreOnly);

        void GetPriorBBoxes(const float* pPrior, size_t numPriors, NormalizedBBoxes& priorBboxes, Variances& priorVariances);

       float BBoxSize(const NormalizedBBox& bbox);

        void ClipBBox(const NormalizedBBox& src, NormalizedBBox& dst);

        void DecodeBBox(const NormalizedBBox& priorBbox, const float* priorVariance, PriorBoxCodeType codeType,
            bool varianceEncodedInTarget, bool clipBbox, const NormalizedBBox& bbox, NormalizedBBox& decodeBbox);

        void DecodeBBoxes(const NormalizedBBoxes& priorBboxes, const Variances& priorVariances,
            PriorBoxCodeType codeType, bool varianceEncodedInTarget, bool clipBbox,
            const NormalizedBBoxes& bboxes, NormalizedBBoxes& decodeBboxes);

        void DecodeBBoxesAll(const LabelBBoxes& allLocPreds, const NormalizedBBoxes& priorBboxes, const Variances& priorVariances,
            size_t num, bool shareLocation, size_t numLocClasses, size_t backgroundLabelId,
            PriorBoxCodeType codeType, bool varianceEncodedInTarget, bool clip, LabelBBoxes& allDecodeBboxes);

        void GetMaxScoreIndex(const Floats& scores, float threshold, ptrdiff_t topK, ScoreIndeces& scoreIndeces);

        void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, NormalizedBBox& intersect);

        float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2, bool normalized);

        void ApplyNMSFast(const NormalizedBBoxes& bboxes, const Floats& scores, float scoreThreshold, float nmsThreshold, float eta, ptrdiff_t topK, Index& indices);
    };
}
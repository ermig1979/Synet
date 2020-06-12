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

namespace Synet
{
    template <class T> class DetectionOutputLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef Synet::Region<T> Region;
        typedef std::vector<Region> Regions;

        DetectionOutputLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const DetectionOutputParam & param = this->Param().detectionOutput();
            _numClasses = param.numClasses();
            _shareLocation = param.shareLocation();
            _numLocClasses = _shareLocation ? 1 : _numClasses;
            _backgroundLabelId = param.backgroundLabelId();
            _codeType = param.codeType();
            _varianceEncodedInTarget = param.varianceEncodedInTarget();
            _keepTopK = param.keepTopK();
            _confidenceThreshold = param.confidenceThreshold();
            _keepMaxClassScoresOnly = param.keepMaxClassScoresOnly();
            _clip = param.clip();
            _nmsThreshold = param.nms().nmsThreshold();
            _topK = param.nms().topK();
            _eta = param.nms().eta();            
            
            //assert(src[0]->Shape() == src[1]->Shape());
            _bboxPreds.Reshape(src[0]->Shape());
            if (_shareLocation)
                _bboxPermute.Reshape(src[0]->Shape());
            _confPermute.Reshape(src[1]->Shape());
            _numPriors = src[2]->Axis(2) / 4;
            assert(_numPriors * _numLocClasses * 4 == src[0]->Axis(1));
            assert(_numPriors * _numClasses == src[1]->Axis(1));
            Shape shape(2, 1);
            shape.push_back(1);
            shape.push_back(7);
            dst[0]->Reshape(shape);
            this->UsePerfStat();
        }

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

        void GetRegions(const TensorPtrs & src, Type threshold, Regions & dst)
        {
            SYNET_PERF_FUNC();
            dst.clear();
            const Type * pSrc = src[0]->CpuData();
            size_t count = src[0]->Axis(2);
            for (size_t i = 0; i < count; ++i)
            {
                if (pSrc[2] > threshold)
                {
                    Region r;
                    r.id = (size_t)pSrc[1];
                    r.prob = pSrc[2];
                    r.w = pSrc[5] - pSrc[3];
                    r.h = pSrc[6] - pSrc[4];
                    r.x = (pSrc[3] + pSrc[5]) / Type(2);
                    r.y = (pSrc[4] + pSrc[6]) / Type(2);
                    dst.push_back(r);
                }
                pSrc += 7;
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const Type * pLoc = src[0]->CpuData();
            const Type * pConf = src[1]->CpuData();
            const Type * pPrior = src[2]->CpuData();
            size_t num = src[0]->Axis(0);

            LabelBBoxes allLocPreds;
            GetLocPredictions(pLoc, num, _numPriors, _numLocClasses, _shareLocation, allLocPreds);

            LabelPreds allConfScores;
            GetConfidenceScores(pConf, num, _numPriors, _numClasses, allConfScores, false);

            if (_keepMaxClassScoresOnly)
            {
                for (size_t i = 0; i < num; ++i) 
                {
                    LabelPred & labelScores = allConfScores[i];
                    for (size_t p = 0; p < _numPriors; ++p)
                    {
                        float maxScore = 0.0f;
                        int maxScoreIdx = -1;
                        for (size_t c = 0; c < _numClasses; ++c)
                        {
                            if (labelScores[(int)c][p] >= maxScore && c != _backgroundLabelId)
                            {
                                maxScoreIdx = (int)c;
                                maxScore = labelScores[(int)c][p];
                            }
                        }
                        for (size_t c = 0; c < _numClasses; ++c)
                            if (c != maxScoreIdx && c != _backgroundLabelId)
                                labelScores[(int)c][p] = 0.0f;
                    }
                }
            }

            NormalizedBBoxes priorBboxes;
            Variances priorVariances;
            GetPriorBBoxes(pPrior, _numPriors, priorBboxes, priorVariances);

            LabelBBoxes allDecodeBboxes;
            DecodeBBoxesAll(allLocPreds, priorBboxes, priorVariances, num, _shareLocation, _numLocClasses, 
                _backgroundLabelId, _codeType, _varianceEncodedInTarget, _clip, allDecodeBboxes);

            size_t numKept = 0;
            IndexMaps allIndices;
            for (size_t i = 0; i < num; ++i) 
            {
                const LabelBBox & decodeBboxes = allDecodeBboxes[i];
                const LabelPred & confScores = allConfScores[i];
                IndexMap indices;
                size_t numDet = 0;
                for (size_t c = 0; c < _numClasses; ++c)
                {
                    if (c == _backgroundLabelId)
                        continue;
                    assert(confScores.find((int)c) != confScores.end());
                    const Floats & scores = confScores.find((int)c)->second;
                    int label = _shareLocation ? -1 : (int)c;
                    assert(decodeBboxes.find(label) != decodeBboxes.end());
                    const NormalizedBBoxes & bboxes = decodeBboxes.find(label)->second;
                    ApplyNMSFast(bboxes, scores, _confidenceThreshold, _nmsThreshold, _eta, _topK, indices[(int)c]);
                    numDet += indices[(int)c].size();
                }
                if (_keepTopK > -1 && numDet > (size_t)_keepTopK)
                {
                    ScoreIndexPairs scoreIndexPairs;
                    for (IndexMap::iterator it = indices.begin(); it != indices.end(); ++it) 
                    {
                        int label = it->first;
                        const Index & labelIndices = it->second;
                        assert(confScores.find(label) != confScores.end());
                        const Floats & scores = confScores.find(label)->second;
                        for (size_t j = 0; j < labelIndices.size(); ++j) 
                        {
                            size_t idx = labelIndices[j];
                            scoreIndexPairs.push_back(ScoreIndexPair(scores[idx], IndexPair(label, idx)));
                        }
                    }
                    std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), [](const ScoreIndexPair & a, const ScoreIndexPair & b) { return a.first > b.first; });
                    scoreIndexPairs.resize(_keepTopK);
                    IndexMap newIndices;
                    for (size_t j = 0; j < scoreIndexPairs.size(); ++j)
                    {
                        size_t label = scoreIndexPairs[j].second.first;
                        size_t idx = scoreIndexPairs[j].second.second;
                        newIndices[(int)label].push_back(idx);
                    }
                    allIndices.push_back(newIndices);
                    numKept += _keepTopK;
                }
                else 
                {
                    allIndices.push_back(indices);
                    numKept += numDet;
                }
            }

            Shape shape(2, 1);
            shape.push_back(numKept);
            shape.push_back(7);
            Type * pDst;
            if (numKept == 0) 
            {
                shape[2] = num;
                dst[0]->Reshape(shape);
                pDst = dst[0]->CpuData();
                CpuSet(dst[0]->Size(), Type(-1), pDst);
                for (size_t i = 0; i < num; ++i) 
                {
                    pDst[0] = Type(i);
                    pDst += 7;
                }
            }
            else 
            {
                dst[0]->Reshape(shape);
                pDst = dst[0]->CpuData();
            }

            size_t count = 0;
            for (size_t i = 0; i < num; ++i) 
            {
                const LabelPred & confScores = allConfScores[i];
                const LabelBBox & decodeBboxes = allDecodeBboxes[i];
                for (IndexMap::iterator it = allIndices[i].begin(); it != allIndices[i].end(); ++it) 
                {
                    int label = it->first;
                    assert(confScores.find(label) != confScores.end());
                    const Floats & scores = confScores.find(label)->second;
                    int locLabel = _shareLocation ? -1 : label;
                    assert(decodeBboxes.find(locLabel) != decodeBboxes.end());
                    const NormalizedBBoxes & bboxes = decodeBboxes.find(locLabel)->second;
                    Index & indices = it->second;
                    for (size_t j = 0; j < indices.size(); ++j) 
                    {
                        size_t idx = indices[j];
                        pDst[count * 7 + 0] = Type(i);
                        pDst[count * 7 + 1] = Type(label);
                        pDst[count * 7 + 2] = scores[idx];
                        const NormalizedBBox & bbox = bboxes[idx];
                        pDst[count * 7 + 3] = bbox.xmin;
                        pDst[count * 7 + 4] = bbox.ymin;
                        pDst[count * 7 + 5] = bbox.xmax;
                        pDst[count * 7 + 6] = bbox.ymax;
                        ++count;
                    }
                }
            }
        }
    private:
        typedef typename Base::Tensor Tensor;
        typedef std::map<int, std::vector<NormalizedBBox>> LabelBBox;
        typedef std::vector<LabelBBox> LabelBBoxes;
        typedef std::map<int, Floats> LabelPred;
        typedef std::vector<LabelPred> LabelPreds;
        typedef std::vector<NormalizedBBox> NormalizedBBoxes;
        typedef std::vector<Type*> Variances;
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
        Tensor _bboxPreds, _bboxPermute, _confPermute;

        void GetLocPredictions(const Type * pLoc, size_t num, size_t numPredsPerClass, size_t numLocClasses, bool shareLocation, LabelBBoxes & locPreds)
        {
            locPreds.clear();
            locPreds.resize(num);
            for (size_t i = 0; i < num; ++i) 
            {
                LabelBBox & labelBbox = locPreds[i];
                for (size_t p = 0; p < numPredsPerClass; ++p)
                {
                    size_t startIdx = p * numLocClasses * 4;
                    for (size_t c = 0; c < numLocClasses; ++c)
                    {
                        int label = shareLocation ? -1 : (int)c;
                        if (labelBbox.find(label) == labelBbox.end())
                            labelBbox[label].resize(numPredsPerClass);
                        labelBbox[label][p].xmin = pLoc[startIdx + c * 4 + 0];
                        labelBbox[label][p].ymin = pLoc[startIdx + c * 4 + 1];
                        labelBbox[label][p].xmax = pLoc[startIdx + c * 4 + 2];
                        labelBbox[label][p].ymax = pLoc[startIdx + c * 4 + 3];
                    }
                }
                pLoc += numPredsPerClass * numLocClasses * 4;
            }
        }

        void GetConfidenceScores(const Type * pConf, size_t num, size_t numPredsPerClass, size_t numClasses, LabelPreds & confPreds, bool keepMaxScoreOnly)
        {
            confPreds.clear();
            confPreds.resize(num);
            for (size_t i = 0; i < num; ++i) 
            {
                LabelPred & labelScores = confPreds[i];
                for (size_t p = 0; p < numPredsPerClass; ++p)
                {
                    size_t startIdx = p * numClasses;
                    size_t maxClassScoreIdx = 0;
                    float maxClassScore = 0;
                    for (size_t c = 0; c < numClasses; ++c)
                    {
                        labelScores[(int)c].push_back(pConf[startIdx + c]);
                        if (pConf[startIdx + c] >= maxClassScore)
                        {
                            maxClassScoreIdx = c;
                            maxClassScore = pConf[startIdx + c];
                        }
                    }
                    if (keepMaxScoreOnly)
                    {
                        for (size_t c = 0; c < numClasses; ++c)
                            if (c != maxClassScoreIdx)
                                labelScores[(int)c].back() = 0.0;
                    }
                }
                pConf += numPredsPerClass * numClasses;
            }
        }

        void GetPriorBBoxes(const Type * pPrior, size_t numPriors, NormalizedBBoxes & priorBboxes, Variances & priorVariances)
        {
            priorBboxes.clear();
            priorVariances.clear();
            priorBboxes.reserve(numPriors);
            priorVariances.reserve(numPriors);
            for (size_t i = 0; i < numPriors; ++i)
            {
                size_t startIdx = i * 4;
                NormalizedBBox bbox;
                bbox.xmin = pPrior[startIdx + 0];
                bbox.ymin = pPrior[startIdx + 1];
                bbox.xmax = pPrior[startIdx + 2];
                bbox.ymax = pPrior[startIdx + 3];
                float bboxSize = BBoxSize(bbox, true);
                bbox.size = bboxSize;
                priorBboxes.push_back(bbox);
                priorVariances.push_back((Type*)(pPrior + (numPriors + i) * 4));
            }
        }

        float BBoxSize(const NormalizedBBox & bbox, bool normalized) 
        {
            if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) 
                return 0;
            else 
            {
                float w = bbox.xmax - bbox.xmin;
                float h = bbox.ymax - bbox.ymin;
                if (normalized)
                    return w * h;
                else 
                    return (w + 1) * (h + 1);
            }
        }

        void ClipBBox(const NormalizedBBox & src, NormalizedBBox & dst) 
        {
            dst.xmin = std::max(std::min(src.xmin, 1.f), 0.f);
            dst.ymin = std::max(std::min(src.ymin, 1.f), 0.f);
            dst.xmax = std::max(std::min(src.xmax, 1.f), 0.f);
            dst.ymax = std::max(std::min(src.ymax, 1.f), 0.f);
            dst.size = BBoxSize(src, true);
            dst.difficult = src.difficult;
        }

        void DecodeBBox(const NormalizedBBox & priorBbox, const T * priorVariance, PriorBoxCodeType codeType, bool varianceEncodedInTarget,
            bool clipBbox, const NormalizedBBox & bbox, NormalizedBBox & decodeBbox) 
        {
            if (codeType == PriorBoxCodeTypeCorner)
            {
                if (varianceEncodedInTarget)
                {
                    decodeBbox.xmin = priorBbox.xmin + bbox.xmin;
                    decodeBbox.ymin = priorBbox.ymin + bbox.ymin;
                    decodeBbox.xmax = priorBbox.xmax + bbox.xmax;
                    decodeBbox.ymax = priorBbox.ymax + bbox.ymax;
                }
                else
                {

                    decodeBbox.xmin = priorBbox.xmin + priorVariance[0] * bbox.xmin;
                    decodeBbox.ymin = priorBbox.ymin + priorVariance[1] * bbox.ymin;
                    decodeBbox.xmax = priorBbox.xmax + priorVariance[2] * bbox.xmax;
                    decodeBbox.ymax = priorBbox.ymax + priorVariance[3] * bbox.ymax;
                }
            }
            else if (codeType == PriorBoxCodeTypeCenterSize)
            {
                float priorW = priorBbox.xmax - priorBbox.xmin;
                float priorH = priorBbox.ymax - priorBbox.ymin;
                float priorX = (priorBbox.xmin + priorBbox.xmax) / 2.0f;
                float priorY = (priorBbox.ymin + priorBbox.ymax) / 2.0f;
                float bboxX, bboxY, bboxW, bboxH;
                if (varianceEncodedInTarget)
                {
                    bboxX = bbox.xmin * priorW + priorX;
                    bboxY = bbox.ymin * priorH + priorY;
                    bboxW = ::exp(bbox.xmax) * priorW;
                    bboxH = ::exp(bbox.ymax) * priorH;
                }
                else
                {
                    bboxX = priorVariance[0] * bbox.xmin * priorW + priorX;
                    bboxY = priorVariance[1] * bbox.ymin * priorH + priorY;
                    bboxW = ::exp(priorVariance[2] * bbox.xmax) * priorW;
                    bboxH = ::exp(priorVariance[3] * bbox.ymax) * priorH;
                }
                decodeBbox.xmin = bboxX - bboxW / 2.0f;
                decodeBbox.ymin = bboxY - bboxH / 2.0f;
                decodeBbox.xmax = bboxX + bboxW / 2.0f;
                decodeBbox.ymax = bboxY + bboxH / 2.0f;
            }
            else if (codeType == PriorBoxCodeTypeCornerSize)
            {
                float priorW = priorBbox.xmax - priorBbox.xmin;
                float priorH = priorBbox.ymax - priorBbox.ymin;
                if (varianceEncodedInTarget)
                {
                    decodeBbox.xmin = priorBbox.xmin + bbox.xmin * priorW;
                    decodeBbox.ymin = priorBbox.ymin + bbox.ymin * priorH;
                    decodeBbox.xmax = priorBbox.xmax + bbox.xmax * priorW;
                    decodeBbox.ymax = priorBbox.ymax + bbox.ymax * priorH;
                }
                else
                {
                    decodeBbox.xmin = priorBbox.xmin + priorVariance[0] * bbox.xmin * priorW;
                    decodeBbox.ymin = priorBbox.ymin + priorVariance[1] * bbox.ymin * priorH;
                    decodeBbox.xmax = priorBbox.xmax + priorVariance[2] * bbox.xmax * priorW;
                    decodeBbox.ymax = priorBbox.ymax + priorVariance[3] * bbox.ymax * priorH;
                }
            }
            else
                assert(0);
            decodeBbox.size = BBoxSize(decodeBbox, true);
            if (clipBbox)
                ClipBBox(decodeBbox, decodeBbox);
        }

        void DecodeBBoxes(const NormalizedBBoxes & priorBboxes, const Variances & priorVariances, PriorBoxCodeType codeType, bool varianceEncodedInTarget,
            bool clipBbox, const NormalizedBBoxes & bboxes, NormalizedBBoxes & decodeBboxes) 
        {
            size_t numBboxes = priorBboxes.size();
            decodeBboxes.clear();
            for (size_t i = 0; i < numBboxes; ++i) 
            {
                NormalizedBBox decodeBbox;
                DecodeBBox(priorBboxes[i], priorVariances[i], codeType, varianceEncodedInTarget, clipBbox, bboxes[i], decodeBbox);
                decodeBboxes.push_back(decodeBbox);
            }
        }

        void DecodeBBoxesAll(const LabelBBoxes & allLocPreds, const NormalizedBBoxes & priorBboxes, const Variances & priorVariances, size_t num, 
            bool shareLocation, size_t numLocClasses, size_t backgroundLabelId, PriorBoxCodeType codeType, bool varianceEncodedInTarget, bool clip, LabelBBoxes & allDecodeBboxes)
        {
            allDecodeBboxes.clear();
            allDecodeBboxes.resize(num);
            for (size_t i = 0; i < num; ++i) 
            {
                LabelBBox & decodeBboxes = allDecodeBboxes[i];
                for (size_t c = 0; c < numLocClasses; ++c) 
                {
                    int label = shareLocation ? -1 : (int)c;
                    if (label == backgroundLabelId)
                        continue;
                    assert(allLocPreds[i].find(label) != allLocPreds[i].end());
                    const NormalizedBBoxes & labelLocPreds = allLocPreds[i].find(label)->second;
                    DecodeBBoxes(priorBboxes, priorVariances, codeType, varianceEncodedInTarget, clip, labelLocPreds, decodeBboxes[label]);
                }
            }
        }

        void GetMaxScoreIndex(const Floats & scores, float threshold, ptrdiff_t topK, ScoreIndeces & scoreIndeces)
        {
            scoreIndeces.clear();
            for (size_t i = 0; i < scores.size(); ++i) 
            {
                if (scores[i] > threshold)
                    scoreIndeces.push_back(ScoreIndex(scores[i], i));
            }
            std::stable_sort(scoreIndeces.begin(), scoreIndeces.end(), [](const ScoreIndex & a, const ScoreIndex & b) { return a.first > b.first; });
            if (topK > -1 && (size_t)topK < scoreIndeces.size())
                scoreIndeces.resize(topK);
        }

        void IntersectBBox(const NormalizedBBox & bbox1, const NormalizedBBox & bbox2, NormalizedBBox & intersect) 
        {
            if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin || bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin)
            {
                intersect.xmin = 0;
                intersect.ymin = 0;
                intersect.xmax = 0;
                intersect.ymax = 0;
            }
            else 
            {
                intersect.xmin = std::max(bbox1.xmin, bbox2.xmin);
                intersect.ymin = std::max(bbox1.ymin, bbox2.ymin);
                intersect.xmax = std::min(bbox1.xmax, bbox2.xmax);
                intersect.ymax = std::min(bbox1.ymax, bbox2.ymax);
            }
        }

        float JaccardOverlap(const NormalizedBBox & bbox1, const NormalizedBBox & bbox2, bool normalized) 
        {
            NormalizedBBox intersect;
            IntersectBBox(bbox1, bbox2, intersect);
            float intersectW, intersectH;
            if (normalized) 
            {
                intersectW = intersect.xmax - intersect.xmin;
                intersectH = intersect.ymax - intersect.ymin;
            }
            else 
            {
                intersectW = intersect.xmax - intersect.xmin + 1;
                intersectH = intersect.ymax - intersect.ymin + 1;
            }
            if (intersectW > 0 && intersectH > 0) 
            {
                float intersectS = intersectW * intersectH;
                float bbox1Size = BBoxSize(bbox1, true);
                float bbox2Size = BBoxSize(bbox2, true);
                return intersectS / (bbox1Size + bbox2Size - intersectS);
            }
            else
                return 0;
        }

        void ApplyNMSFast(const NormalizedBBoxes & bboxes, const Floats & scores, float scoreThreshold, float nmsThreshold, float eta, ptrdiff_t topK, Index & indices)
        {
            ScoreIndeces scoreIndeces;
            GetMaxScoreIndex(scores, scoreThreshold, topK, scoreIndeces);
            float adaptiveThreshold = nmsThreshold;
            indices.clear();
            while (scoreIndeces.size() != 0)
            {
                size_t idx = scoreIndeces.front().second;
                bool keep = true;
                for (int k = 0; k < indices.size(); ++k) 
                {
                    if (keep)
                    {
                        size_t keptIdx = indices[k];
                        float overlap = JaccardOverlap(bboxes[idx], bboxes[keptIdx], true);
                        keep = overlap <= adaptiveThreshold;
                    }
                    else
                        break;
                }
                if (keep)
                    indices.push_back(idx);
                scoreIndeces.erase(scoreIndeces.begin());
                if (keep && eta < 1 && adaptiveThreshold > 0.5)
                    adaptiveThreshold *= eta;
            }
        }
    };
}
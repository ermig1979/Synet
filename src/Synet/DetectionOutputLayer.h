/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
    template <class T, template<class> class A> class DetectionOutputLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        DetectionOutputLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
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
            _nmsThreshold = param.nms().nmsThreshold();
            _topK = param.nms().topK();
            _eta = param.nms().eta();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            assert(src[0]->Shape() == src[1]->Shape());
            _bboxPreds.Reshape(src[0]->Shape());
            if (_shareLocation)
                _bboxPermute.Reshape(src[0]->Shape());
            _confPermute.Reshape(src[1]->Shape());

//            num_priors_ = bottom[2]->height() / 4;
//            CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->channels())
//                << "Number of priors must match number of location predictions.";
//            CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
//                << "Number of priors must match number of confidence predictions.";
//            // num() and channels() are 1.
//            vector<int> top_shape(2, 1);
//            // Since the number of bboxes to be kept is unknown before nms, we manually
//            // set it to (fake) 1.
//            top_shape.push_back(1);
//            // Each row is a 7 dimension vector, which stores
//            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
//            top_shape.push_back(7);
//            top[0]->Reshape(top_shape);
//            dst[0]->Share(*src[0]);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
//            const Dtype* loc_data = bottom[0]->cpu_data();
//            const Dtype* conf_data = bottom[1]->cpu_data();
//            const Dtype* prior_data = bottom[2]->cpu_data();
//            const int num = bottom[0]->num();
//
//            // Retrieve all location predictions.
//            vector<LabelBBox> all_loc_preds;
//            GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
//                share_location_, &all_loc_preds);
//
//            // Retrieve all confidences.
//            vector<map<int, vector<float> > > all_conf_scores;
//            GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
//                &all_conf_scores);
//
//            if (keep_max_class_scores_only_) {
//                for (int i = 0; i < num; ++i) {
//                    map<int, vector<float> >& label_scores = all_conf_scores[i];
//                    for (int p = 0; p < num_priors_; ++p) {
//                        float max_score = 0.0f;
//                        int max_score_idx = -1;
//                        for (int c = 0; c < num_classes_; ++c) {
//                            if (label_scores[c][p] >= max_score && c != background_label_id_) {
//                                max_score_idx = c;
//                                max_score = label_scores[c][p];
//                            }
//                        }
//                        CHECK_GE(max_score_idx, 0);
//                        CHECK_NE(max_score_idx, background_label_id_);
//                        for (int c = 0; c < num_classes_; ++c)
//                            if (c != max_score_idx && c != background_label_id_)
//                                label_scores[c][p] = 0.0f;
//                    }
//                }
//            }
//
//            // Retrieve all prior bboxes. It is same within a batch since we assume all
//            // images in a batch are of same dimension.
//            vector<NormalizedBBox> prior_bboxes;
//            vector<vector<float> > prior_variances;
//            GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);
//
//            // Decode all loc predictions to bboxes.
//            vector<LabelBBox> all_decode_bboxes;
//            const bool clip_bbox = false;
//            DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
//                share_location_, num_loc_classes_, background_label_id_,
//                code_type_, variance_encoded_in_target_, clip_bbox,
//                &all_decode_bboxes);
//
//            int num_kept = 0;
//            vector<map<int, vector<int> > > all_indices;
//            for (int i = 0; i < num; ++i) {
//                const LabelBBox& decode_bboxes = all_decode_bboxes[i];
//                const map<int, vector<float> >& conf_scores = all_conf_scores[i];
//                map<int, vector<int> > indices;
//                int num_det = 0;
//                for (int c = 0; c < num_classes_; ++c) {
//                    if (c == background_label_id_) {
//                        // Ignore background class.
//                        continue;
//                    }
//                    if (conf_scores.find(c) == conf_scores.end()) {
//                        // Something bad happened if there are no predictions for current label.
//                        LOG(FATAL) << "Could not find confidence predictions for label " << c;
//                    }
//                    const vector<float>& scores = conf_scores.find(c)->second;
//                    int label = share_location_ ? -1 : c;
//                    if (decode_bboxes.find(label) == decode_bboxes.end()) {
//                        // Something bad happened if there are no predictions for current label.
//                        LOG(FATAL) << "Could not find location predictions for label " << label;
//                        continue;
//                    }
//                    const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
//                    ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
//                        top_k_, &(indices[c]));
//                    num_det += indices[c].size();
//                }
//                if (keep_top_k_ > -1 && num_det > keep_top_k_) {
//                    vector<pair<float, pair<int, int> > > score_index_pairs;
//                    for (map<int, vector<int> >::iterator it = indices.begin();
//                        it != indices.end(); ++it) {
//                        int label = it->first;
//                        const vector<int>& label_indices = it->second;
//                        if (conf_scores.find(label) == conf_scores.end()) {
//                            // Something bad happened for current label.
//                            LOG(FATAL) << "Could not find location predictions for " << label;
//                            continue;
//                        }
//                        const vector<float>& scores = conf_scores.find(label)->second;
//                        for (int j = 0; j < label_indices.size(); ++j) {
//                            int idx = label_indices[j];
//                            CHECK_LT(idx, scores.size());
//                            score_index_pairs.push_back(std::make_pair(
//                                scores[idx], std::make_pair(label, idx)));
//                        }
//                    }
//                    // Keep top k results per image.
//                    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
//                        SortScorePairDescend<pair<int, int> >);
//                    score_index_pairs.resize(keep_top_k_);
//                    // Store the new indices.
//                    map<int, vector<int> > new_indices;
//                    for (int j = 0; j < score_index_pairs.size(); ++j) {
//                        int label = score_index_pairs[j].second.first;
//                        int idx = score_index_pairs[j].second.second;
//                        new_indices[label].push_back(idx);
//                    }
//                    all_indices.push_back(new_indices);
//                    num_kept += keep_top_k_;
//                }
//                else {
//                    all_indices.push_back(indices);
//                    num_kept += num_det;
//                }
//            }
//
//            vector<int> top_shape(2, 1);
//            top_shape.push_back(num_kept);
//            top_shape.push_back(7);
//            Dtype* top_data;
//            if (num_kept == 0) {
//                LOG(INFO) << "Couldn't find any detections";
//                top_shape[2] = num;
//                top[0]->Reshape(top_shape);
//                top_data = top[0]->mutable_cpu_data();
//                caffe_set<Dtype>(top[0]->count(), -1, top_data);
//                // Generate fake results per image.
//                for (int i = 0; i < num; ++i) {
//                    top_data[0] = i;
//                    top_data += 7;
//                }
//            }
//            else {
//                top[0]->Reshape(top_shape);
//                top_data = top[0]->mutable_cpu_data();
//            }
//
//            int count = 0;
//            for (int i = 0; i < num; ++i) {
//                const map<int, vector<float> >& conf_scores = all_conf_scores[i];
//                const LabelBBox& decode_bboxes = all_decode_bboxes[i];
//                for (map<int, vector<int> >::iterator it = all_indices[i].begin();
//                    it != all_indices[i].end(); ++it) {
//                    int label = it->first;
//                    if (conf_scores.find(label) == conf_scores.end()) {
//                        // Something bad happened if there are no predictions for current label.
//                        LOG(FATAL) << "Could not find confidence predictions for " << label;
//                        continue;
//                    }
//                    const vector<float>& scores = conf_scores.find(label)->second;
//                    int loc_label = share_location_ ? -1 : label;
//                    if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
//                        // Something bad happened if there are no predictions for current label.
//                        LOG(FATAL) << "Could not find location predictions for " << loc_label;
//                        continue;
//                    }
//                    const vector<NormalizedBBox>& bboxes =
//                        decode_bboxes.find(loc_label)->second;
//                    vector<int>& indices = it->second;
//                    if (need_save_) {
//                        CHECK(label_to_name_.find(label) != label_to_name_.end())
//                            << "Cannot find label: " << label << " in the label map.";
//                        CHECK_LT(name_count_, names_.size());
//                    }
//                    for (int j = 0; j < indices.size(); ++j) {
//                        int idx = indices[j];
//                        top_data[count * 7] = i;
//                        top_data[count * 7 + 1] = label;
//                        top_data[count * 7 + 2] = scores[idx];
//                        const NormalizedBBox& bbox = bboxes[idx];
//                        top_data[count * 7 + 3] = bbox.xmin();
//                        top_data[count * 7 + 4] = bbox.ymin();
//                        top_data[count * 7 + 5] = bbox.xmax();
//                        top_data[count * 7 + 6] = bbox.ymax();
//                        ++count;
//                    }
//                }
//            }
        }
    private:
        typedef Base::Tensor Tensor;

        bool _shareLocation, _varianceEncodedInTarget, _keepMaxClassScoresOnly;
        size_t _numClasses, _numLocClasses;
        ptrdiff_t _backgroundLabelId, _keepTopK, _topK;
        PriorBoxCodeType _codeType;
        float _confidenceThreshold, _nmsThreshold, _eta;
        Tensor _bboxPreds, _bboxPermute, _confPermute;

        //int num_;
        //int num_priors_;


        //map<int, string> label_to_name_;
        //map<int, string> label_to_display_name_;
        //vector<string> names_;
        //vector<pair<int, int> > sizes_;
        //int num_test_image_;
        //int name_count_;

        //shared_ptr<DataTransformer<Dtype> > data_transformer_;
    };
}
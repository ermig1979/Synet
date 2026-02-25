/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

#include "Cvt/OnnxRuntime/Common.h"

namespace Synet
{
    bool ConvertAddNode(const onnx::NodeProto& node, LayerParams& layers, const Bytes& original, const OnnxParam& onnxParam, LayerParam& layer);

    bool ConvertConstantNode(const onnx::NodeProto& node, LayerParam& layer, Bytes& original, Bytes& reordered);

    bool ConvertConvOrConvTransposeNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap, UniqNames& merged);

    bool ConvertBatchNormalizationNode(const onnx::NodeProto& node, const LayerParams& layers, Bytes& original, LayerParam& layer, Bytes& reordered);

    bool ConvertDequantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer);

    bool ConvertDropoutNode(const onnx::NodeProto& node, LayerParam& layer);

    bool ConvertFlattenNode(const onnx::NodeProto& node, LayerParam& layer);

    bool ConvertGemmNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& original, LayerParam& layer, Bytes& reordered, TensorFormatMap* tensorFormatMap, UniqNames& merged);

    bool ConvertGlobalAveragePoolNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer, UniqNames& merged);

    bool ConvertInitializer(const onnx::TensorProto& tensor, Synet::NetworkParam& network, Bytes& weight, Renames& renames);

    bool ConvertInput(const onnx::ValueInfoProto& input, bool trans, Synet::NetworkParam& network, Renames& renames);

    bool ConvertNonMaxSuppressionNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& bin, LayerParam& layer);

    bool ConvertPreluNode(const onnx::NodeProto& node, LayerParams& layers, LayerParam& layer);

    bool ConvertQLinearAddNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer);

    bool ConvertQLinearAveragePoolNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer);

    bool ConvertQLinearConvNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap);

    bool ConvertQLinearGlobalAveragePoolNode(const onnx::NodeProto& node, const LayerParams& layers, const Bytes& srcBin, LayerParam& layer);

    bool ConvertQLinearMatMulNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, const Bytes& srcBin, LayerParam& layer, Bytes& dstBin, TensorFormatMap* tensorFormatMap);

    bool ConvertQuantizeLinearNode(const onnx::NodeProto& node, bool trans, const LayerParams& layers, const Bytes& original, LayerParam& layer);

    bool ConvertReduceL2Node(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer);

    bool ConvertReduceMeanNode(const onnx::NodeProto& node, bool trans, LayerParams& layers, LayerParam& layer, UniqNames& merged);
}

#endif
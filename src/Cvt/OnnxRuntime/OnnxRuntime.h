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

//#define SYNET_ONNX_PARSE_STOP_ON_ERROR

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/SynetUtils.h"
#include "Cvt/Optimizer/Optimizer.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

#include "Cvt/OnnxRuntime/Common.h"
#include "Cvt/OnnxRuntime/Attribute.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(String srcGraphPath, bool trans, const String& dstModelPath, const String& dstWeightPath, const OnnxParam& onnxParam, const OptimizerParam& optParam);

    private:
        bool LoadModel(const String& path, onnx::ModelProto& model);

        bool ConvertModel(const onnx::ModelProto& model, bool trans, const OnnxParam& onnxParam, Synet::NetworkParam& network, Bytes& reordered);

        void SetSrcAndDst(const onnx::NodeProto& node, Renames& renames, LayerParam& layer);

        bool ManualInsertToNchwPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        bool ManualInsertToNhwcPermute(const OnnxParam& onnxParam, LayerParams& layers, Renames& renames);

        bool PrintGraph(const onnx::GraphProto& graph, std::ostream & os, bool printConst = false, bool filterInput = true);

        String ValueInfoString(const onnx::ValueInfoProto& info);

        String TensorString(const onnx::TensorProto& tensor, size_t printSizeMax = 3);

        String AttributeString(const onnx::AttributeProto& attribute);

        String NodeString(const onnx::NodeProto& node);

        void NotImplemented(const onnx::NodeProto& node, LayerParam& dst);

        bool ErrorMessage(size_t index, const onnx::NodeProto& node);
    };

    //---------------------------------------------------------------------------------------------

    bool ConvertOnnxToSynet(const String& srcGraph, bool trans, const String& dstXml, const String& dstBin, const OnnxParam& onnxParam, const OptimizerParam& optParam);
}

#endif
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"
#include "Synet/Converters/SynetUtils.h"
#include "Synet/Utils/FileUtils.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

namespace Synet
{
    class OnnxToSynet : public SynetUtils
    {
    public:
        bool Convert(const String& srcParamPath, const String& srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            onnx::ModelProto model;
            if (!LoadModel(srcGraphPath, model))
                return false;

            Synet::NetworkParamHolder holder;
            Vector weight;
            if (!ConvertModel(model, trans, holder(), weight))
                return false;

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
                return false;

            if (!holder.Save(dstModelPath, false))
            {
                std::cout << "Can't save Synet model '" << dstModelPath << "' !" << std::endl;
                return false;
            }

            if (!SaveBinaryData(weight, dstWeightPath))
            {
                std::cout << "Can't save Synet weight '" << dstWeightPath << "' !" << std::endl;
                return false;
            }

            return true;
        }

    private:

        bool LoadModel(const String& path, onnx::ModelProto& model)
        {
            std::ifstream ifs(path.c_str(), std::ios::ate | std::ios_base::binary);
            if (!ifs.is_open())
            {
                std::cout << "Can't open file '" << path << "' !" << std::endl;
                return false;
            }
            size_t size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            ifs.read(buffer.data(), size);
            ifs.close();
            std::cout << "Parse " << path << ", size: " << size << std::endl;

            if (!model.ParseFromArray(buffer.data(), size))
            {
                std::cout << "Can't parse file '" << path << "' !" << std::endl;
                return false;
            }

            return true;
        }

        bool ConvertModel(const onnx::ModelProto & model, bool trans, Synet::NetworkParam& network, Vector& reordered)
        {
            const onnx::GraphProto& graph = model.graph();

            PrintGraph(graph, std::cout);

            return true;
        }

        bool PrintGraph(const onnx::GraphProto& graph, std::ostream & os)
        {
            os << std::endl;
            os << "graph name: " << graph.name() << std::endl;
            for (size_t i = 0; i < graph.input_size(); ++i)
                os << " input[" << i << "]: " << ValueInfoString(graph.input(i)) << std::endl;
            for (size_t i = 0; i < graph.node_size(); ++i)
            {
                const onnx::NodeProto& node = graph.node(i);
                os << " node[" << i << "] " << node.op_type() << " : " << node.name() << " (";
                for (size_t j = 0; j < node.input_size(); ++j)
                    os << " " << node.input(j);
                os << " ) -> (";
                for (size_t j = 0; j < node.output_size(); ++j)
                    os << " " << node.output(j);
                os << " )";
                os << std::endl;
            }
            for (size_t i = 0; i < graph.output_size(); ++i)
                os << " output[" << i << "]: " << ValueInfoString(graph.output(i)) << std::endl;
            os << std::endl;

            return true;
        }

        String ValueInfoString(const onnx::ValueInfoProto& info)
        {
            std::stringstream ss;
            ss << info.name();
            if (info.type().has_tensor_type())
            {
                Shape shape = Convert(info.type().tensor_type().shape());
                ss << " {";
                for (size_t j = 0; j < shape.size(); ++j)
                    ss << " " << ptrdiff_t(shape[j]);
                ss << " }";
            }
            return ss.str();
        }

        Shape Convert(const onnx::TensorShapeProto& shapeProto)
        {
            Shape shape;
            for (size_t i = 0; i < shapeProto.dim_size(); ++i)
            {
                if (shapeProto.dim(i).has_dim_value())
                    shape.push_back((size_t)shapeProto.dim(i).dim_value());
                else
                    shape.push_back(size_t(-1));
            }
            return shape;
        }
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif
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
#include "Synet/Params.h"
#include "Synet/Tensor.h"
#include "Synet/Converters/Optimizer.h"

#if defined(SYNET_ONNX_ENABLE)

namespace Synet
{
    class OnnxToSynet
    {
    public:
        bool Convert(const String& srcParamPath, const String& srcGraphPath, bool trans, const String & dstModelPath, const String & dstWeightPath)
        {
            if (!Synet::FileExist(srcParamPath))
            {
                std::cout << "File '" << srcParamPath << "' is not exist!" << std::endl;
                return false;
            }

            if (!Synet::FileExist(srcGraphPath))
            {
                std::cout << "File '" << srcGraphPath << "' is not exist!" << std::endl;
                return false;
            }

            Synet::NetworkParamHolder holder;
            Vector weight;

            return false;

            OptimizerParamHolder param;
            Optimizer optimizer(param());
            if (!optimizer.Run(holder(), weight))
                return false;

            if (!holder.Save(dstModelPath, false))
                return false;

            if (!SaveWeight(weight, dstWeightPath))
                return false;

            return true;
        }

    private:
        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<float> Vector;

        bool SaveWeight(const Vector& bin, const String& path)
        {
            std::ofstream ofs(path.c_str(), std::ofstream::binary);
            if (!ofs.is_open())
                return false;
            ofs.write((const char*)bin.data(), bin.size() * sizeof(float));
            bool result = (bool)ofs;
            ofs.close();
            return result;
        }
    };

    bool ConvertOnnxToSynet(const String& srcParam, const String& srcGraph, bool trans, const String& dstXml, const String& dstBin)
    {
        OnnxToSynet onnxToSynet;
        return onnxToSynet.Convert(srcParam, srcGraph, trans, dstXml, dstBin);
    }
}

#endif
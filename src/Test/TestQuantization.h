/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#include "TestCommon.h"
#include "TestOptions.h"
#include "TestSynet.h"
#include "TestImage.h"

#include "Synet/Converters/Deoptimizer.h"
#include "Synet/Converters/Optimizer.h"

namespace Test
{
    class Quantizer
    {
    public:
        Quantizer(const Options & options)
            : _options(options)
        {
            if (_options.debugPrint)
            {
                _opt = Test::MakePath(_options.outputDirectory, "fp32_optimized.xml");
                _deopt = Test::MakePath(_options.outputDirectory, "fp32_deoptimized.xml");
                _stats = Test::MakePath(_options.outputDirectory, "fp32_deopt_stats.xml");
            }
            else
            {
                _opt = _options.secondModel;
                _deopt = _options.secondModel;
                _stats = _options.secondModel;
            }
        }

        bool Run()
        {
            PrintStartMessage();
            if (!LoadTestParam())
                return PrintFinishMessage(false);
            if (!CreateDirectories())
                return PrintFinishMessage(false);
            if (!OptimizeModel())
                return PrintFinishMessage(false);
            if (!DeoptimizeModel())
                return PrintFinishMessage(false);
            if (!CollectStatistics())
                return PrintFinishMessage(false);
            if (!PerformQuntization())
                return PrintFinishMessage(false);
            return PrintFinishMessage(true);
        }

    private:
        TestParamHolder _param;
        const Options& _options;
        String _opt, _deopt, _stats;
        Strings _images;
        typedef Synet::Network<float> SyNet;
        SyNet _synet;

        void PrintStartMessage() const
        {
            std::cout << "Start quantization of Synet model : ";
            if (!_options.consoleSilence)
                std::cout << std::endl;
        }

        bool PrintFinishMessage(bool result) const
        {
            if (result)
                std::cout << (_options.consoleSilence ? " OK." : "Quantization is finished successful.") << std::endl;
            else
                std::cout << "Quantization is finished with errors!" << std::endl;
            return result;
        }

        bool LoadTestParam()
        {
            if (!_param.Load(_options.testParam))
            {
                std::cout << "Can't load file '" << _options.testParam << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
            {
                std::cout << "Can't create output directory '" << _options.outputDirectory << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool OptimizeModel()
        {
            if (!_options.consoleSilence)
                std::cout << "Optimize Synet FP32 model : ";
            bool result = Synet::OptimizeSynetModel(_options.firstModel, "", _opt, "");
            if (!_options.consoleSilence)
                std::cout << (result ? " OK." : "Optimization is finished with errors!") << std::endl;
            return result;
        }

        bool DeoptimizeModel()
        {
            if(!_options.consoleSilence)
                std::cout << "Deoptimize Synet FP32 model : ";
            bool result = Synet::DeoptimizeSynetModel(_opt, _deopt);
            if (_options.debugPrint)
                Synet::OptimizeSynetModel(_deopt, "", Test::MakePath(_options.outputDirectory, "fp32_optimized_back.xml"), "");
            if (!_options.consoleSilence)
                std::cout << (result ? " OK." : "Deoptimization is finished with errors!") << std::endl;
            return result;
        }

        bool CollectStatistics()
        {
            if (!_options.consoleSilence)
                std::cout << "Collect quantization statistics : ";
            bool result = InitSynet(_deopt, _options.firstWeight);
            result = result && CreateImageList();
            for (size_t i = 0; i < _images.size() && result; ++i)
                result = result && UpdateStatistics(_images[i]);
            result = result && _synet.Save(_stats);
            if (!_options.consoleSilence)
                std::cout << (result ? " OK." : "Statistics collection is finished with errors!") << std::endl;
            return result;
        }

        bool InitSynet(const String& model, const String& weight)
        {
            if (!_synet.Load(model, weight))
            {
                std::cout << "Can't load Synet model from '" << model << "' and '" << weight << "' !" << std::endl;
                return false;
            }
            if (_synet.Format() != Synet::TensorFormatNhwc)
            {
                std::cout << "Quantizer supports only models in NHWC format!" << std::endl;
                return false;
            }
            return true;
        }

        bool CreateImageList()
        {
            String directory = _options.imageDirectory;
            if (directory.empty())
                directory = Test::MakePath(DirectoryByPath(_options.testParam), _param().images());
            if (!DirectoryExists(directory))
            {
                std::cout << "Image directory '" << directory << "' is not exists!" << std::endl;
                return false;
            }
            StringList images = GetFileList(directory, _options.imageFilter, true, false);
            images.sort();
            if (images.empty())
            {
                std::cout << "There is no one image in '" << directory << "' for '" << _options.imageFilter << "' filter!" << std::endl;
                return false;
            }

            _images.reserve(images.size());
            size_t curr = 0;
            for (StringList::const_iterator it = images.begin(); it != images.end(); ++it, ++curr)
                if (curr >= _options.imageBegin && curr < _options.imageEnd)
                    _images.push_back(MakePath(directory, *it));
            if (_images.empty())
            {
                std::cout << "There is no one image in '" << _options.imageDirectory << "' for '" << _options.imageFilter
                    << "' filter in range [" << _options.imageBegin << " ... " <<  _options.imageEnd << "] !" << std::endl;
                return false;
            }

            return true;
        }

        bool UpdateStatistics(const String & path)
        {
            View original;
            if (!LoadImage(path, original))
            {
                std::cout << "Can't load image in '" << path << "' !" << std::endl;
                return false;
            }
            View resized(_synet.NchwShape()[3], _synet.NchwShape()[2], original.format);
            Simd::Resize(original, resized, SimdResizeMethodArea);
            if (!_synet.SetInput(resized, _param().lower(), _param().upper()))
            {
                std::cout << "Can't set input for '" << path << "' image !" << std::endl;
                return false;
            }
            _synet.Forward();
            _synet.UpdateStatistics(_options.quantizationQuantile, 0.000001f);
            if (!_options.consoleSilence)
                std::cout << ".";
            return true;
        }

        void ScaleToConvolution(Synet::LayerParam& layer)
        {
            if (layer.type() != Synet::LayerTypeScale)
                return;
            const SyNet::Tensor* tensor = _synet.GetInternalTensor(layer.src()[0]);
            if (!(tensor && tensor->Format() == Synet::TensorFormatNhwc && tensor->Count() == 4))
                return;
            size_t channels = tensor->Axis(3);
            layer.type() = Synet::LayerTypeConvolution;
            layer.convolution().biasTerm() = layer.scale().biasTerm();
            layer.convolution().outputNum() = (uint32_t)channels;
            layer.convolution().group() = (uint32_t)channels;
            layer.convolution().kernel() = Shp(1, 1);
            layer.weight()[0].dim() = Shp(1, 1, 1, channels);
            layer.weight()[0].format() = Synet::TensorFormatNhwc;
            layer.scale() = Synet::ScaleParam();
            layer.convolution().quantizationLevel() = Synet::TensorType8i;
        }

        void EltwiseToAdd(Synet::LayerParam& layer)
        {
            if (layer.type() != Synet::LayerTypeEltwise || 
                layer.eltwise().operation() != Synet::EltwiseOperationTypeSum ||
                layer.src().size() != 2)
                return;
            const SyNet::Tensor* src0 = _synet.GetInternalTensor(layer.src()[0]);
            if (!(src0 && src0->Format() == Synet::TensorFormatNhwc && src0->Count() == 4))
                return;
            const SyNet::Tensor* src1 = _synet.GetInternalTensor(layer.src()[1]);
            if (!(src1 && src1->Format() == Synet::TensorFormatNhwc && src1->Count() == 4))
                return;
            if (src0->Shape() != src1->Shape())
                return;
            layer.eltwise() = Synet::EltwiseParam();
            layer.type() = Synet::LayerTypeAdd;
        }

        void QuantizeConvolution(Synet::LayerParam& layer)
        {
            if (layer.type() != Synet::LayerTypeConvolution)
                return;
            const SyNet::Tensor* tensor = _synet.GetInternalTensor(layer.src()[0]);
            if (tensor && tensor->Format() == Synet::TensorFormatNhwc)
            {
                Shape shape = tensor->Shape();
                if (shape[1] == 1 && shape[2] == 1)
                    return;
            }
            layer.convolution().quantizationLevel() = Synet::TensorType8i;
        }

        void QuantizeInnerProduct(Synet::LayerParam& layer)
        {
            if (layer.type() != Synet::LayerTypeInnerProduct)
                return;
            layer.innerProduct().quantizationLevel() = Synet::TensorType8i;
        }

        void HighlightGlobalPooling(Synet::LayerParam& layer)
        {
            if (layer.type() != Synet::LayerTypePooling)
                return;
            const SyNet::Tensor* tensor = _synet.GetInternalTensor(layer.src()[0]);
            if (tensor && tensor->Format() == Synet::TensorFormatNhwc)
            {
                const Shape & shape = tensor->Shape();
                const Shape& kernel = layer.pooling().kernel();
                const Shape& stride = layer.pooling().stride();
                if (shape.size() == 4 && shape[1] == kernel[0] && shape[2] == kernel[1] && 
                    layer.pooling().stride() == Shp(1, 1) && layer.pooling().pad() == Shp(0, 0, 0, 0))
                {
                    layer.pooling().globalPooling() = true;
                    layer.pooling().kernel().clear();
                    layer.pooling().stride().clear();
                    layer.pooling().pad().clear();
                }
            }
        }

        bool PerformQuntization()
        {
            if (!_options.consoleSilence)
                std::cout << "Perform model quantization : ";
            Synet::NetworkParamHolder network;
            if (!network.Load(_stats))
            {
                std::cout << "Can't load Synet model '" << _stats << "' !" << std::endl;
                return false;
            }
            for (size_t i = 0; i < network().layers().size(); ++i)
            {
                Synet::LayerParam& layer = network().layers()[i];
                EltwiseToAdd(layer);
                //ScaleToConvolution(layer);
                QuantizeConvolution(layer);
                QuantizeInnerProduct(layer);
                HighlightGlobalPooling(layer);
            }
            network().quantization().method() = (Synet::QuantizationMethod)_options.quantizationMethod;
            Floats bin;
            Synet::Optimizer optimizer;
            if (!optimizer.Run(network(), bin))
            {
                std::cout << "Can't optimize Synet model!" << std::endl;
                return false;
            }
            if (!network.Save(_options.secondModel, false))
            {
                std::cout << "Can't save Synet model '" << _options.secondModel << "' !" << std::endl;
                return false;
            }
            if (!_options.consoleSilence)
                std::cout << " OK." << std::endl;
            return true;
        }
    };
}



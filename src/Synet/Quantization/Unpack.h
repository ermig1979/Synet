/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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

#include "Synet/Converters/Optimizer.h"

namespace Synet
{
	namespace Quantization
	{
		class ComplexLayerUnpacker
		{
		public:
			bool Run(Synet::NetworkParam& network)
			{
				if (!UnpackLayers(network, 0))
					return false;
				if (!UnpackLayers(network, 1))
					return false;
				return true;
			}

		private:
			struct Change
			{
				size_t start;
				String prev, curr;
				Change(size_t s, const String & p, const String & c)
					: start(s), prev(p), curr(c) {}
			};
			typedef std::vector<Change> Changes;
			typedef std::vector<Synet::LayerParam> LayerParams;

            bool UnpackLayers(Synet::NetworkParam & network, int stage)
            {
                Changes changes;
                LayerParams unpacked;
                for (size_t i = 0; i < network.layers().size(); ++i)
                {
					const LayerParam& layer = network.layers()[i];
                    switch (stage)
                    {
                    case 0:
                    {
                        if (UnpackMergedConvolution(layer, unpacked, changes))
                            continue;
                        break;
                    }
                    case 1:
                    {
                        if (UnpackConvolutionOrDeconvolution(layer, unpacked, changes))
                            continue;
                        break;
                    }
                    default:
                        assert(0);
                        return false;
                    }
					unpacked.push_back(layer);
                }
                Rename(changes, unpacked);
                network.layers() = unpacked;
                return true;
            }

			bool UnpackConvolutionOrDeconvolution(const Synet::LayerParam & layer, LayerParams & unpacked, Changes & changes)
			{
				if (layer.type() != LayerTypeConvolution && layer.type() != LayerTypeDeconvolution)
					return false;
				if (layer.convolution().activationType() == ActivationFunctionTypeIdentity)
					return false;
				LayerParam convolution = layer;
				LayerParam activation;
				if (layer.convolution().activationType() == ActivationFunctionTypeRelu)
				{
					activation.type() = LayerTypeRelu;
					activation.name() = layer.name() + "_relu";
					activation.relu().negativeSlope() = layer.convolution().activationParam0();
				}
				else
					return false;
				convolution.convolution().activationType() = ActivationFunctionTypeIdentity;
				convolution.convolution().activationParam0() = 0.0f;
				convolution.convolution().activationParam1() = 6.0f;
				convolution.dst()[0] = convolution.name();
				activation.src() = convolution.dst();
				activation.dst().push_back(activation.name());
				unpacked.push_back(convolution);
				unpacked.push_back(activation);
				changes.push_back(Change(unpacked.size(), convolution.name(), activation.name()));
				return true;
			}

			size_t WeightCount(const Synet::ConvolutionParam& convolution) const
			{
				return 1 + (convolution.biasTerm() ? 1 : 0) + (convolution.activationType() == ActivationFunctionTypePrelu ? 1 : 0);
			}

			bool UnpackMergedConvolution(const Synet::LayerParam& layer, LayerParams& unpacked, Changes& changes)
			{
				if (layer.type() != LayerTypeMergedConvolution || layer.mergedConvolution().add())
					return false;

				LayerParam conv[3];
				for (int i = 0, w = 0; i < 3; ++i)
				{
					conv[i].type() = LayerTypeConvolution;
					conv[i].name() = layer.name() + (i == 2 ? String("") : String("_conv") + Synet::ValueToString(i));
					conv[i].src() = i ? conv[i - 1].dst() : layer.src();
					conv[i].dst().push_back(conv[i].name());
					conv[i].convolution() = layer.mergedConvolution().conv()[i];
					for (size_t end = w + WeightCount(conv[i].convolution()); w < end; ++w)
						conv[i].weight().push_back(layer.weight()[w]);
					unpacked.push_back(conv[i]);
				}

				return true;
			}

			bool Rename(const Change& change, LayerParams& layers)
			{
				for (size_t i = change.start; i < layers.size(); ++i)
				{
					for (size_t j = 0; j < layers[i].src().size(); ++j)
					{
						if (layers[i].src()[j] == change.prev)
							layers[i].src()[j] = change.curr;
					}
				}
				return true;
			}

			bool Rename(const Changes& changes, LayerParams& layers)
			{
				for (size_t k = 0; k < changes.size(); ++k)
				{
					if (!Rename(changes[k], layers))
						return false;
				}
				return true;
			}
		};

		inline bool UnpackComplexLayers(const String& src, const String& dst)
		{
			if (!FileExist(src))
			{
				std::cout << "File '" << src << "' is not exist!" << std::endl;
				return false;
			}

			NetworkParamHolder network;
			if (!network.Load(src))
			{
				std::cout << "Can't load '" << src << "' model file!" << std::endl;
				return false;
			}

			ComplexLayerUnpacker unpacker;
			if (!unpacker.Run(network()))
			{
				std::cout << "Can't unpack complex layers in '" << src << "' model!" << std::endl;
				return false;
			}

			if (!network.Save(dst, false))
			{
				std::cout << "Can't save '" << dst << "' model file!" << std::endl;
				return false;
			}

			return true;
		}

		inline bool MergeLayersBack(const String& src, const String& dst)
		{
			NetworkParamHolder network;
			if (network.Load(src))
			{
				Optimizer optimizer;
				Floats bin;
				optimizer.Run(network(), bin);
			}
			return network.Save(dst, false);
		}
	}
}
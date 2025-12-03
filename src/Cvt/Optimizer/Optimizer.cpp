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

#include "Cvt/Optimizer/Common.h"
#include "Cvt/Optimizer/Optimizer.h"
#include "Cvt/Optimizer/Bf16OptSetter.h"

namespace Synet
{
    Optimizer::Optimizer(const OptimizerParam& param)
        : _param(param)
    {
    }

    bool Optimizer::Run(Synet::NetworkParam& network, Bytes& bin)
    {
        Bf16OptSetter bf16OptSetter(_param.bf16());
        if (!bf16OptSetter.Run(network, bin))
            return false;
        for (int stage = 0; stage < 10; stage++)
        {
            if (!OptimizeLayers(network, bin, stage))
                return false;
        }
        if (!bf16OptSetter.Run(network, bin))
            return false;
        if (!RemoveStub(network))
            return false;
        if (!RemoveUnusedConst(network.layers()))
            return false;
        if (_param.reuseLayers())
        {
            if (!ReuseLayers(network))
                return false;
        }
        return true;
    }

    bool Optimizer::OptimizeLayers(Synet::NetworkParam& network, Bytes& bin, int stage)
    {
        QuantizationMethod method = network.quantization().method();
        const bool is8i = network.quantization().method() != QuantizationMethodUnknown;
        const bool isNhwc = IsNnwc(network);
        Changes changes;
        LayerParams merged;
        Bytes buf;
        for (size_t i = 0; i < network.layers().size(); ++i)
        {
            switch (stage)
            {
            case 0:
            {
                if (ReduceTensorIteratorIO(network.layers(), i, bin, buf, merged))
                    continue;
                break;
            }
            case 1:
            {
                if (TransposeInnerProduct(network.layers(), i, bin, buf, merged))
                    continue;
                break;
            }
            case 2:
            {
                if (MergeCurrentAndBias(network.layers(), i, bin, merged, changes))
                    continue;
                break;
            }
            case 3:
            {
                if (MergePowerAndScaleAndPower(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeBiasAndScale(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeConvolutionAndScale(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeConvolutionAndPower(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeInnerProductAndPower(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeInnerProductAndScale(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (SimplifyInterp(network.layers(), i, merged, changes))
                    continue;
                break;
            }
            case 4:
            {
                if (MergeHswish(network.layers(), i, merged, changes))
                    continue;
                if (MergeHswishV2(network.layers(), i, merged, changes))
                    continue;
                if (MergeMish(network.layers(), i, merged, changes))
                    continue;
                if (MergePrelu0(network.layers(), i, bin, merged, changes))
                    continue;
                if (MergePrelu1(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeShuffle0(network.layers(), i, merged, changes))
                    continue;
                if (MergeShuffle1(network.layers(), i, merged, changes))
                    continue;
                if (MergeShuffle2(network.layers(), i, merged, changes))
                    continue;
                if (MergeShuffle3(network.layers(), i, merged, changes))
                    continue;
                if (MergeShuffle3cut(network.layers(), i, isNhwc, merged, changes))
                    continue;
                if (MergeShuffle4(network.layers(), i, merged, changes))
                    continue;
                if (MergeSoftmax(network.layers(), i, merged, changes))
                    continue;
                if (MergePermute(network.layers(), i, merged, changes))
                    continue;
                if (MergePooling(network.layers(), i, merged, changes))
                    continue;
                if (MergeSpaceToDepth(network.layers(), i, merged, changes))
                    continue;
                if (MergeSwish(network.layers(), i, merged, changes))
                    continue;
                if (MergeNormalize(network.layers(), i, merged, changes))
                    continue;
                if (MergeNormalizeV2(network.layers(), i, isNhwc, merged, changes))
                    continue;
                if (MergeNormalizeV4(network.layers(), i, isNhwc, merged, changes))
                    continue;
                if (MergeNormalizeV5(network.layers(), i, merged, changes))
                    continue;
                if (MergeGelu(network.layers(), i, merged, changes))
                    continue;
                if (MergeGeluV2(network.layers(), i, merged, changes))
                    continue;
                if (MergeScale(network.layers(), i, merged, changes))
                    continue;
                if (MergeTiledScale2D(network.layers(), i, merged, changes))
                    continue;
                if (MergeUnpack4(network.layers(), i, isNhwc, merged, changes))
                    continue;
                if (MergeQuantizedShuffleV0(network.layers(), i, merged, changes))
                    continue;
                if (MergeQuantizedScale(network.layers(), i, merged, changes))
                    continue;
                if (MergeQuantizedPrelu(network.layers(), i, merged, changes))
                    continue;
                if (MergeQuantizedAdd(network.layers(), i, merged, changes))
                    continue;
                break;
            }
            case 5:
            {
                if (MergePowerAndScaleAndPower(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeConvolutionOrOtherAndActivation(network.layers(), i, method, merged, changes))
                    continue;
                if (MergeRnnGruBd(network.layers(), i, merged, changes))
                    continue;
                break;
            }
            case 6:
            {
                if (_param.convToNhwc() && isNhwc && TransposeConvolutions(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeOtherAndQuantizeLinear(network.layers(), i, method, merged, changes))
                    continue;
                if (SkipUnnecessaryDequantizeQuantizeV0(network.layers(), i, method, merged, changes))
                    continue;
                if (SkipUnnecessaryDequantizeQuantizeV1(network.layers(), i, method, merged, changes))
                    continue;
                if (SkipUnnecessaryDequantize(network.layers(), i, method, merged, changes))
                    continue;
                if (MergeQuantizedConvolutionAndQuantizedActivation(network.layers(), i, method, merged, changes))
                    continue;
                break;
            }
            case 7:
            {
                if (MergeThreeConvolutions(network.layers(), i, method, _param, merged, changes))
                    continue;
                if (MergeThreeQuantizedConvolutions(network.layers(), i, _param, merged, changes))
                    continue;
                if (MergeSqueezeExcitation(network.layers(), i, merged, changes))
                    continue;
                if (_param.skipPermute() && SkipTwoPermutes(network.layers(), i, merged))
                    continue;
                break;
            }
            case 8:
            {
                if (MergeTwoConvolutions(network.layers(), i, method, _param, merged, changes))
                    continue;
                if (MergeTwoQuantizedConvolutions(network.layers(), i, _param, merged, changes))
                    continue;
                break;
            }
            case 9:
            {
                if (MergeParallelConvolutions(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeConcatedParallelConvolutions(network.layers(), i, bin, buf, merged, changes))
                    continue;
                if (MergeYoloV7(network.layers(), i, merged, changes))
                    continue;
                //if (MergeParallelScaleAndDepthwiseConvolution(network.layers(), i, bin, buf, merged, changes))
                //    continue;
                if (MergeParallelDepthwiseConvolutions(network.layers(), i, bin, buf, merged, changes))
                    continue;
                break;
            }
            default:
                assert(0);
                return false;
            }
            merged.push_back(network.layers()[i]);
        }
        Rename(changes, merged);
        network.layers() = merged;
        if (buf.size())
            bin.swap(buf);
        return true;
    }

    //--------------------------------------------------------------------------------------------------

    bool OptimizeSynetModel(const String& srcXml, const String& srcBin, const String& dstXml, const String & dstBin, const OptimizerParam & param)
    {
        NetworkParamHolder network;
        if (!network.Load(srcXml))
        {
            std::cout << "Can't load Synet model '" << srcXml << "' !" << std::endl;
            return false;
        }
        Bytes bin;
        if (!srcBin.empty() && !LoadBinaryData(srcBin, bin))
        {
            std::cout << "Can't load Synet weight '" << srcBin << "' !" << std::endl;
            return false;
        }
        Optimizer optimizer(param);
        if (!optimizer.Run(network(), bin))
        {
            std::cout << "Can't optimize Synet model!" << std::endl;
            return false;
        }
        if (!network.Save(dstXml, false))
        {
            std::cout << "Can't save Synet model '" << dstXml << "' !" << std::endl;
            return false;
        }
        if (!dstBin.empty() && !SaveBinaryData(bin, dstBin))
        {
            std::cout << "Can't save Synet weight '" << dstBin << "' !" << std::endl;
            return false;
        }
        return true;
    }
}
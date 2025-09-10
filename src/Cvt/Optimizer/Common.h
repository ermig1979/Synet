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
#include "Synet/Params.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"

namespace Synet
{
    typedef std::vector<Synet::LayerParam> LayerParams;
    typedef std::pair<String, String> Change;
    typedef std::vector<Change> Changes;
    typedef std::vector<LayerType> LayerTypes;
    typedef std::set<String> StringSet;

    //--------------------------------------------------------------------------------------------------

    inline bool InsideLink(const LayerParams& src, size_t start, size_t count, size_t skip = 0, const LayerTypes& ignored = LayerTypes())
    {
        for (size_t i = start + count + skip; i < src.size(); ++i)
        {
            bool ignore = false;
            for (size_t j = 0; j < ignored.size(); ++j)
                if (src[i].type() == ignored[j])
                    ignore = true;
            if (ignore)
                continue;
            for (size_t j = 0; j < src[i].src().size(); ++j)
            {
                for (size_t k = 0; k < count - 1; ++k)
                {
                    if (src[i].src()[j] == src[start + k].name())
                        return true;
                }
            }
        }
        return false;
    }

    //--------------------------------------------------------------------------------------------------

    bool MergeConvolutionAndScale(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeInnerProductAndScale(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeGelu(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeGeluV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeHswish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeHswishV2(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeMish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergePrelu0(const LayerParams& src, size_t& index, const Bytes& bin, LayerParams& dst, Changes& changes);

    bool MergePrelu1(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeSwish(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeParallelConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeParallelDepthwiseConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeParallelScaleAndDepthwiseConvolution(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes);

    bool MergeQuantizedShuffleV0(const LayerParams& src, size_t& index, LayerParams& dst, Changes& changes);

    bool MergeThreeConvolutions(const LayerParams& src, size_t& index, QuantizationMethod method, const OptimizerParam& param, LayerParams& dst, Changes& changes);

    bool MergeTwoQuantizedConvolutions(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes);

    bool MergeThreeQuantizedConvolutions(const LayerParams& src, size_t& index, const OptimizerParam& param, LayerParams& dst, Changes& changes);

    bool MergeTwoConvolutions(const LayerParams& src, size_t& index, QuantizationMethod method, const OptimizerParam& param, LayerParams& dst, Changes& changes);

    bool SkipUnnecessaryDequantizeQuantizeV0(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes);

    bool SkipUnnecessaryDequantizeQuantizeV1(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes);

    bool SkipUnnecessaryDequantize(const LayerParams& src, size_t& index, QuantizationMethod method, LayerParams& dst, Changes& changes);
}
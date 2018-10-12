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
#include "Synet/Params.h"

namespace Synet
{
    class Optimizer
    {
    public:

        bool Run(Synet::NetworkParam & network)
        {
            LayerParams merged;
            if (!Merge(network.layers(), merged))
                return false;

            network.layers() = merged;

            return true;
        }

    private:
        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef std::pair<String, String> Change;
        typedef std::vector<Change> Changes;

        bool Merge(const LayerParams & src, LayerParams & dst)
        {
            Changes changes;
            for (size_t i = 0; i < src.size(); ++i)
            {
                if (!Merge(src, i, dst, changes))
                    dst.push_back(src[i]);
            }
            for (size_t k = 0; k < changes.size(); ++k)
            {
                for (size_t i = 0; i < dst.size(); ++i)
                {
                    for (size_t j = 0; j < dst[i].src().size(); ++j)
                    {
                        if (dst[i].src()[j] == changes[k].first)
                            dst[i].src()[j] = changes[k].second;
                    }
                }            
            }   
            return true;
        }

        bool Merge(const LayerParams & src, size_t index, LayerParams & dst, Changes & changes)
        {
            if (index == 0)
                return false;
            if (src[index - 1].type() != LayerTypeConvolution)
                return false;
            for (size_t i = index + 1; i < src.size(); ++i)
            {
                for (size_t j = 0; j < src[i].src().size(); ++j)
                {
                    if (src[i].src()[j] == src[index - 1].name())
                        return false;
                }
            }
            if (src[index].type() == LayerTypeRestrictRange)
            {
                dst.back().convolution().activationType() = ActivationFunctionTypeRestrictRange;
                dst.back().convolution().activationParam0() = src[index].restrictRange().lower();
                dst.back().convolution().activationParam1() = src[index].restrictRange().upper();
                changes.push_back(Change(src[index].name(), src[index - 1].name()));
                return true;
            }
            if (src[index].type() == LayerTypeRelu)
            {
                dst.back().convolution().activationType() = src[index].relu().negativeSlope() == 0.0f ? ActivationFunctionTypeRelu : ActivationFunctionTypeLeakyRelu;
                dst.back().convolution().activationParam0() = src[index].relu().negativeSlope();
                changes.push_back(Change(src[index].name(), src[index - 1].name()));
                return true;
            }
            return false;
        }
    };
}
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

namespace Synet
{
    bool MergeConcatedParallelConvolutions(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        const LayerParam& l0 = src[index];
        Shape parts;
        for (; index + parts.size() < src.size();)
        {
            const LayerParam& l = src[index + parts.size()];
            if (l.type() != LayerTypeConvolution)
                break;
            const ConvolutionParam& c = l.convolution();
            if (c.group() != 1)
                break;
            if (parts.size())
            {
                if (l.src() != l0.src())
                    break;
                if (l.weight().size() != l0.weight().size())
                    break;
                if (l.weight()[0].format() != l0.weight()[0].format())
                    break;
                const ConvolutionParam& c0 = l0.convolution();
                if (c.kernel() != c0.kernel())
                    break;
                if (c.pad() != c0.pad())
                    break;
                if (c.stride() != c0.stride())
                    break;
                if (c.dilation() != c0.dilation())
                    break;
                if (c.biasTerm() != c0.biasTerm())
                    break;
                if (c.activationType() != c0.activationType())
                    break;
                if (c.activationParam0() != c0.activationParam0())
                    break;
                if (c.activationParam1() != c0.activationParam1())
                    break;
            }
            parts.push_back(c.outputNum());
        }
        if (parts.size() < 2)
            return false;
        const LayerParam& conc = src[index + parts.size()];
        if (conc.type() != LayerTypeConcat || conc.src().size() != parts.size())
            return false;
        for (size_t s = 0; s < conc.src().size(); ++s)
            if (conc.src()[s] != src[index + s].dst()[0])
                return false;

        LayerParam conv;
        conv.type() = LayerTypeConvolution;
        conv.name() = conc.name();
        conv.src() = l0.src();
        conv.convolution() = l0.convolution();
        conv.weight() = l0.weight();
        for (size_t p = 1; p < parts.size(); ++p)
        {
            const LayerParam& l = src[index + p];
            conv.convolution().outputNum() += l.convolution().outputNum();
            conv.name() = conv.name() + "_" + l.name();
            for (size_t w = 0; w < l0.weight().size(); ++w)
            {
                if (l.weight()[0].format() == TensorFormatNhwc)
                    conv.weight()[w].dim().back() += l.weight()[w].dim().back();
                else
                    conv.weight()[w].dim().front() += l.weight()[w].dim().front();
                conv.weight()[w].size() += l.weight()[w].size();
            }
        }
        conv.dst() = conc.dst();

        if (buf.empty())
            buf = bin;
        size_t newSize = buf.size();
        for (size_t w = 0; w < conv.weight().size(); ++w)
            newSize += conv.weight()[w].size();
        conv.weight()[0].offset() = buf.size();
        buf.resize(newSize);
        for (size_t w = 0; w < conv.weight().size(); ++w)
        {
            if (w)
                conv.weight()[w].offset() = conv.weight()[w - 1].offset() + conv.weight()[w - 1].size();
            const Shape& dim = conv.weight()[w].dim();
            std::vector<const float*> pSrc(parts.size());
            for (size_t p = 0; p < parts.size(); ++p)
                pSrc[p] = GetWeight<float>(bin, src[index + p].weight()[w]);
            float* pDst = GetWeight<float>(buf, conv.weight()[w]);
            if (l0.weight()[0].format() == TensorFormatNhwc && w == 0)
            {
                for (size_t o = 0, outer = dim[0] * dim[1] * dim[2]; o < outer; ++o)
                {
                    for (size_t p = 0; p < parts.size(); ++p)
                    {
                        for (size_t i = 0; i < parts[p]; ++i)
                            *pDst++ = *pSrc[p]++;
                    }
                }
            }
            else
            {
                for (size_t p = 0; p < parts.size(); ++p)
                {
                    memcpy(pDst, pSrc[p], src[index + p].weight()[w].size());
                    pDst += src[index + p].weight()[w].size() / sizeof(float);
                }
            }
        }

        dst.push_back(conv);

        index += parts.size();

        return true;
    }
}

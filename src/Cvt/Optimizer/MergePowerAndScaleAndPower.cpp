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
    bool MergePowerAndScaleAndPower(const LayerParams& src, size_t& index, Bytes& bin, Bytes& buf, LayerParams& dst, Changes& changes)
    {
        bool pre = false, scale = false, post = false;
        if (src.size() > index + 0 && src[index + 0].type() == LayerTypePower && src[index + 0].power().power() == 1.0f)
            pre = true;
        if (src.size() > index + 1 && src[index + 1].type() == LayerTypeScale && (pre ? src[index + 1].src()[0] == src[index + 0].name() : true) && src[index + 1].scale().biasTerm())
            scale = true;
        if (src.size() > index + 2 && src[index + 2].type() == LayerTypePower && src[index + 2].power().power() == 1.0f && src[index + 2].src()[0] == src[index + 1].name())
            post = true;
        if (!(scale && (pre || post)))
            return false;
        if (InsideLink(src, index + (pre ? 0 : 1), 1 + (pre ? 1 : 0) + (post ? 1 : 0), 0, LayerTypes({ LayerTypePriorBox, LayerTypePriorBoxClustered, LayerTypeMeta })))
            return false;
        LayerParam layer;
        layer.type() = LayerTypeScale;
        layer.name() = src[index + 1].name();
        layer.src().push_back(pre ? src[index + 0].src()[0] : src[index + 1].src()[0]);
        layer.dst().push_back(post ? src[index + 2].dst()[0] : src[index + 1].dst()[0]);
        layer.scale() = src[index + 1].scale();
        layer.weight() = src[index + 1].weight();
        float preScale = pre ? src[index + 0].power().scale() : 1.0f;
        float preBias = pre ? src[index + 0].power().shift() : 0.0f;
        float postScale = post ? src[index + 2].power().scale() : 1.0f;
        float postBias = post ? src[index + 2].power().shift() : 0.0f;
        if (buf.empty())
            buf = bin;
        float* pScale = GetWeight<float>(buf, layer.weight()[0]);
        float* pShift = GetWeight<float>(buf, layer.weight()[1]);
        size_t size = TensorSize(layer.weight()[0].dim());
        for (size_t i = 0; i < size; ++i)
        {
            pShift[i] = (preBias * pScale[i] + pShift[i]) * postScale + postBias;
            pScale[i] = preScale * pScale[i] * postScale;
        }
        if (pre)
            changes.push_back(Change(src[index + 0].dst()[0], layer.dst()[0]));
        else
            dst.push_back(src[index + 0]);
        index += 1 + (post ? 1 : 0);
        dst.push_back(layer);
        return true;
    }
}

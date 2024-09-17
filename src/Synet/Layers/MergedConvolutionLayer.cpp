/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "Synet/Layer.h"
#include "Synet/Utils/MergedConvolution.h"
#include "Synet/Layers/ActivationLayers.h"
#include "Synet/Utils/ImgToCol.h"
#include "Synet/Utils/Activation.h"
#include "Synet/Layers/MergedConvolutionLayer.h"
#include "Synet/Layers/PreluLayer.h"

namespace Synet
{
    MergedConvolutionLayer::MergedConvolutionLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool MergedConvolutionLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("MergedConvolutionLayer supports only 1 input and 1 output!");
        if (src[0]->Count() != 4 || src[0]->Format() != TensorFormatNhwc)
            SYNET_ERROR("MergedConvolutionLayer supports only 4D NHWC input tensor!");

        const MergedConvolutionParam & p = this->Param().mergedConvolution();
        const ConvolutionParam * conv = p.conv().data();
        AlgParam& a = _alg;
        a.count = p.conv().size();
        if (a.count < 2 && a.count > 3)
            SYNET_ERROR("MergedConvolutionLayer supports only 2 or 3 merged convolutions!");

        const Tensors & weight = this->Weight();
        for (size_t i = 0, next = 0; i < a.count; ++i)
        {
            a.conv[i].Set(conv[i]);
            if(i)
                a.conv[i].Set(a.conv[i - 1], true, conv[i].autoPad());
            else
                a.conv[i].Set(*src[0], *dst[0], true, conv[i].autoPad());

            a.index[i] = next++;
            const Tensor & w = weight[a.index[i]];
            if(w.Shape() != a.conv[i].WeightShape(true, true) || w.Format() != src[0]->Format())
                SYNET_ERROR("MergedConvolutionLayer: check weight[" << a.index[i] << "] size or format!");
            a.weight[i] = w.Data<float>();

            a.biasTerm[i] = conv[i].biasTerm();
            if (a.biasTerm[i])
            {
                const Tensor & b = weight[next++];
                if(b.Size() != a.conv[i].dstC)
                    SYNET_ERROR("MergedConvolutionLayer has wrong bias[" << i << "] (weight[" << next - 1 << "]) size!");
                a.bias[i] = b.Data<float>();
            }
            else
                a.bias[i] = NULL;

            if (a.conv[i].activation == ActivationFunctionTypePrelu)
            {
                const Tensor & p = weight[next++];
                if (p.Size() == 1)
                    a.conv[i].activation = ActivationFunctionTypeLeakyRelu;
                else
                {
                    if(p.Size() != a.conv[i].dstC)
                        SYNET_ERROR("MergedConvolutionLayer has wrong weight[" << next - 1 << "] size!");
                }
                a.params[i] = p.Data<float>();
            }
            else
            {
                a.actParam[i][0] = conv[i].activationParam0();
                a.actParam[i][1] = conv[i].activationParam1();
                a.params[i] = a.actParam[i];
            }
            a.internal[i] = 0;
        }

        a.add = (this->LowPrecision(TensorType8u) != LowPrecisionTypeActive && a.count == 3 && p.add()) ? 1 : 0;
        a.batch = src[0]->Axis(0);

        if (!Reshape(src[0], buf, dst[0]))
            return false;

        a.sSize = src[0]->Size(1);
        a.dSize = dst[0]->Size(1);
        if (a.add)
        {
            if(a.sSize != a.dSize)
                SYNET_ERROR("MergedConvolutionLayer with add=1 parameter must have input and output of the same size!");
        }

        std::stringstream desc;
        desc << a.count << ": " << a.batch << "x" << a.conv[0].srcC << "x" << a.conv[0].srcH << "x" << a.conv[0].srcW;
        for(size_t i = 0; i < a.count; ++i)
            desc << "-" << (a.conv[i].IsDepthwise() ? String("") : Cpl::ToStr(a.conv[i].dstC) + "x") << a.conv[i].kernelY << "x" << a.conv[i].strideY;
        desc << InternalInfo();
        this->UsePerfStat(desc.str(), Flop());
        return true;
    }

    void MergedConvolutionLayer::CompactWeight()
    {
        const AlgParam& a = _alg;
        for(size_t i = 0; i < a.count; ++i)
            if (a.internal[i])
                ((Tensor&)this->Weight()[a.index[i]]).Clear();
    }

    int64_t MergedConvolutionLayer::Flop() const
    {
        const AlgParam& a = _alg;
        int64_t flop = 0;
        for (size_t i = 0; i < a.count; ++i)
            flop += a.batch * a.conv[i].kernelY * a.conv[i].kernelX * a.conv[i].srcC * a.conv[i].dstH * a.conv[i].dstW * a.conv[i].dstC / a.conv[i].group * 2;
        return flop;
    }
}
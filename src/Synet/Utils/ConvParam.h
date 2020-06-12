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

#include "Synet/Params.h"

namespace Synet
{
    struct ConvParam
    {
        size_t srcC;
        size_t srcH;
        size_t srcW;
        TensorType srcT;
        TensorFormat srcF;
        size_t dstC;
        size_t dstH;
        size_t dstW;
        TensorType dstT;
        TensorFormat dstF;
        size_t kernelY;
        size_t kernelX;
        size_t dilationY;
        size_t dilationX;
        size_t strideY;
        size_t strideX;
        size_t padY;
        size_t padX;
        size_t padH;
        size_t padW;
        size_t group;
        ActivationFunctionType activation;

        void Set(const ConvolutionParam & conv)
        {
            const Shape & kernel = conv.kernel();
            assert(kernel.size() == 1 || kernel.size() == 2);
            kernelY = kernel[0];
            kernelX = kernel.size() > 1 ? kernel[1] : kernelY;
            assert(kernelY > 0 && kernelX > 0);

            const Shape & stride = conv.stride();
            assert(stride.size() <= 2);
            strideY = stride.size() > 0 ? stride[0] : 1;
            strideX = stride.size() > 1 ? stride[1] : strideY;
            assert(strideY > 0 && strideX > 0);

            const Shape & dilation = conv.dilation();
            assert(dilation.size() <= 2);
            dilationY = dilation.size() > 0 ? dilation[0] : 1;
            dilationX = dilation.size() > 1 ? dilation[1] : dilationY;
            assert(dilationY > 0 && dilationX > 0);

            const Shape & pad = conv.pad();
            assert(pad.size() <= 4 && pad.size() != 3);
            padY = pad.size() > 0 ? pad[0] : 0;
            padX = pad.size() > 1 ? pad[1] : padY;
            padH = pad.size() > 2 ? pad[2] : padY;
            padW = pad.size() > 3 ? pad[3] : padX;
            assert(padY >= 0 && padX >= 0 && padH >= 0 && padW >= 0);

            activation = conv.activationType();
            group = conv.group();
            dstC = conv.outputNum();
            assert(dstC > 0 && dstC % group == 0);
        }

        template<class T> void Set(const Tensor<T> & src, const Tensor<T>& dst, bool conv)
        {
            if (src.Format() == TensorFormatNhwc)
            {
                srcC = src.Axis(-1);
                srcH = src.Axis(-3);
                srcW = src.Axis(-2);
            }
            else
            {
                srcC = src.Axis(-3);
                srcH = src.Axis(-2);
                srcW = src.Axis(-1);
            }
            srcT = src.GetType();
            srcF = src.Format();
            dstT = dst.GetType() == TensorTypeUnknown ? srcT : dst.GetType();
            dstF = dst.Format() == TensorFormatUnknown ? srcF : dst.Format();
            SetDst(conv);
        }

        void Set(const ConvParam & src, bool conv)
        {
            srcC = src.dstC;
            srcH = src.dstH;
            srcW = src.dstW;
            srcT = src.srcT;
            srcF = src.srcF;
            dstT = src.dstT;
            dstF = src.dstF;
            SetDst(conv);
        }

        void SetDst(bool conv)
        {
            if (conv)
            {
                dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
                dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            }
            else
            {
                dstH = strideY * (srcH - 1) + dilationY * (kernelY - 1) + 1 - padY - padH;
                dstW = strideX * (srcW - 1) + dilationX * (kernelX - 1) + 1 - padX - padW;
            }
        }

        bool Trans() const
        {
            return srcF == TensorFormatNhwc;
        }

        bool Is1x1() const
        {
            return kernelY == 1 && kernelX == 1 && strideY == 1 && strideX == 1 &&
                dilationY == 1 && dilationX == 1 && padY == 0 && padX == 0 && padH == 0 && padW == 0;
        }

        bool IsDepthwise() const
        {
            return group == srcC && group == dstC;
        }

        Shape WeightShape(bool trans, bool conv) const 
        {
            if (conv)
            {
                if (trans)
                    return Shp(kernelY, kernelX, srcC / group, dstC);
                else
                    return Shp(dstC, srcC / group, kernelY, kernelX);
            }
            else
            {
                if (trans)
                    return Shp(srcC, kernelY, kernelX, dstC / group);
                else
                    return Shp(srcC, dstC / group, kernelY, kernelX);
            }
        }

        Shape DstShape(size_t batch) const
        {
            if (dstF == TensorFormatNhwc)
                return Shp(batch, dstH, dstW, dstC);
            else
                return Shp(batch, dstC, dstH, dstW);
        }
    };
}
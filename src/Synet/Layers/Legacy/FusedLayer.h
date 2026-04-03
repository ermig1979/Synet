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

#pragma once

#include "Synet/Layer.h"

namespace Synet
{
    class FusedLayer : public Layer
    {
    public:
        FusedLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

        void ForwardCpu11(const float* src, float* dst);

        void ForwardCpu22(const float* src0, const float* src1, float* dst0, float* dst1);

        void ForwardCpu31(const float* src0, const float* src1, const float* src2, float* dst);

    private:
        typedef Layer::Tensor Tensor;
        typedef Layer::Tensors Tensors;

        TensorFormat _format;
        int _type, _trans;
        size_t _count, _size, _num, _srcStride, _dstStride;

        struct T0
        {
            Tensor bias, scale;
        } _t0;

        struct T1
        {
            Tensor bias0, scale1, bias1;
        } _t1;

        struct T2
        {
            Tensor scale, bias;
            float slope;
        } _t2;

        struct T3
        {
            Tensor bias, scale;
        } _t3;

        struct T4
        {
            Tensor bias0;
            float scale1, bias1;
        } _t4;

        struct T5
        {
            Tensor bias, scale;
            size_t count0, count1;
        } _t5;

        struct T11
        {
            float params[4];
        } _t11;
    };
}
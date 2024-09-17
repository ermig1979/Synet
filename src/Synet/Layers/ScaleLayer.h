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
#include "Synet/Utils/Scale.h"

namespace Synet
{
    void ScaleForward32f(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, TensorFormat format, int compatibility);

    //-------------------------------------------------------------------------------------------------

    class ScaleLayer : public Layer
    {
    public:
        ScaleLayer(const LayerParam& param, Context* context, QuantizationMethod method);

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual size_t MemoryUsage() const;

        virtual void CompactWeight();

        virtual int64_t Flop() const;

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        void Scale32f(const float* src, float* dst);

        void Init8i();

    private:
        QuantizationMethod _method;
        TensorFormat _format;
        size_t _axis, _batch, _channels, _height, _width;
        int _compatibility, _lower, _upper;
        bool _biasTerm, _src8u, _dst8u, _is8i, _src16b, _dst16b;
        Tensor _scale, _shift;
        Scale8i _scale8i;
    };
}
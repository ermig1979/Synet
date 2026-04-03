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

#include "Synet/Layer.h"

namespace Synet
{
    class GridSampleLayer : public Layer
    {
    public:
        GridSampleLayer(const LayerParam& param, Context* context);

        virtual ~GridSampleLayer();

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual size_t MemoryUsage() const;

        typedef void (*GridSample2dPtr)(const uint8_t* src8, size_t batch, size_t channels, size_t srcH, size_t srcW, const uint8_t* grid8, size_t dstH, size_t dstW, uint8_t* dst8);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        TensorType _type;
        GridSampleInterpMode _interp;
        GridSamplePaddingMode _padding;
        bool _align;
        size_t _batch, _channels, _srcH, _srcW, _dstH, _dstW;
        GridSample2dPtr _gridSample2d;
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        std::vector<void*> _context;
#endif
    };
}
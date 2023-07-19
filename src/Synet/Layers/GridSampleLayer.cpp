/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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

#include "Synet/Layers/GridSampleLayer.h"

namespace Synet
{
    template <typename T, bool corners> SYNET_INLINE T Denormalize(T pos, int64_t dim) 
    {
        if (corners)
            return T((pos + 1) / 2.0f * (dim - 1));
        else
            return T(((pos + 1) * dim - 1) / 2.0f);
    }

    template <typename T> SYNET_INLINE T Reflect(T x, float min, float max) 
    {
        float fx = float(x);
        float range = max - min;
        if (fx < min)
        {
            float dx = min - fx;
            int n = int(dx / range);
            float r = dx - n * range;
            return n % 2 == 0 ? T(min + r) : T(max - r);
        }
        else if (fx > max)
        {
            float dx = fx - max;
            int n = int(dx / range);
            float r = dx - n * range;
            return n % 2 == 0 ? T(max - r) : T(min + r);
        }
        else
            return T(fx);
    }

    template<class T, GridSampleInterpMode interp, GridSamplePaddingMode padding, bool corners> 
    void GridSample2d(const uint8_t* src8, size_t batch, size_t channels, size_t srcH, size_t srcW, const uint8_t* grid8, size_t dstH, size_t dstW, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        const T* grid = (const T*)grid8;
        T* dst = (T*)dst8;
        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                const T* pgr = grid;
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    for (size_t dx = 0; dx < dstW; ++dx)
                    {
                        T x = Denormalize<T, corners>(pgr[0], srcW);
                        T y = Denormalize<T, corners>(pgr[1], srcH);

                        pgr += 2; dst += 1;
                    }
                }
                src += srcH * srcW;
            }
            grid += dstH * dstW * 2;
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T, GridSampleInterpMode interp, GridSamplePaddingMode padding> GridSampleLayer::GridSample2dPtr GetGridSample2d(bool corners)
    {
        return corners ? GridSample2d<T, interp, padding, true> : GridSample2d<T, interp, padding, false>;
    }

    template<class T, GridSampleInterpMode interp> GridSampleLayer::GridSample2dPtr GetGridSample2d(GridSamplePaddingMode padding, bool corners)
    {
        switch (padding)
        {
        case GridSamplePaddingModeZeros: return GetGridSample2d<T, interp, GridSamplePaddingModeZeros>(corners);
        case GridSamplePaddingModeBorder: return GetGridSample2d<T, interp, GridSamplePaddingModeBorder>(corners);
        case GridSamplePaddingModeReflection: return GetGridSample2d<T, interp, GridSamplePaddingModeReflection>(corners);
        default:
            return NULL;
        }
    }

    template<class T> GridSampleLayer::GridSample2dPtr GetGridSample2d(GridSampleInterpMode interp, GridSamplePaddingMode padding, bool corners)
    {
        switch (interp)
        {
        case GridSampleInterpModeBilinear: return GetGridSample2d<T, GridSampleInterpModeBilinear>(padding, corners);
        case GridSampleInterpModeNearest: return GetGridSample2d<T, GridSampleInterpModeNearest>(padding, corners);
        case GridSampleInterpModeBicubic: return GetGridSample2d<T, GridSampleInterpModeBicubic>(padding, corners);
        default:
            return NULL;
        }
    }

    GridSampleLayer::GridSample2dPtr GetGridSample2d(TensorType type, GridSampleInterpMode interp, GridSamplePaddingMode padding, bool corners)
    {
        switch (type)
        {
        case TensorType32f: return GetGridSample2d<float>(interp, padding, corners);
        default:
            return NULL;
        }
    }

    //-------------------------------------------------------------------------------------------------

    GridSampleLayer::GridSampleLayer(const LayerParam& param, Context* context)
        : Base(param, context)
    {
    }

    bool GridSampleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 && dst.size() != 1)
            SYNET_ERROR("GridSampleLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("GridSampleLayer don't src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " != src[1] type "  << Cpl::ToStr(src[1]->GetType()) << " !");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("GridSampleLayer support only TensorType32f inputs!");

        Shape srcShape = src[0]->Shape();
        Shape gridShape = src[1]->Shape();
        _rank = srcShape.size() - 2;
        if (srcShape.size() < 3 || srcShape.size() != gridShape.size() || srcShape[0] != gridShape[0] || gridShape[_rank + 1] != _rank)
            SYNET_ERROR("GridSampleLayer has incompatible input shapes: src[0] :" << ToStr(srcShape) << " and src[1]: " << ToStr(gridShape) << " !");
        if (srcShape.size() != 4)
            SYNET_ERROR("GridSampleLayer can have only 4D inputs but: src[0]: " << ToStr(srcShape) << " and src[1]: " << ToStr(gridShape) << " !");

        Shape dstShape = Shp(srcShape[0], srcShape[1]);
        for(size_t r = 1; r <= _rank; ++r)
            dstShape.push_back(gridShape[r]);

        GridSampleParam gridSample = this->Param().gridSample();
        _gridSample2d = GetGridSample2d(src[0]->GetType(), gridSample.interpMode(), gridSample.paddingMode(), gridSample.alignCorners());
        if (_gridSample2d == NULL)
            SYNET_ERROR("GridSampleLayer can't get worker!");

        dst[0]->Reshape(src[0]->GetType(), dstShape, src[0]->Format());
        this->UsePerfStat();

        return true;
    }

    void GridSampleLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        _gridSample2d(src[0]->RawData(), _batch, _channels, _srcH, _srcW, src[1]->RawData(), _dstH, _dstW, dst[0]->RawData());
    }
}
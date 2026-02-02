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

#include "Synet/Layers/GridSampleLayer.h"

namespace Synet
{
    template <typename T, bool corners> SYNET_INLINE T Denormalize(T pos, ptrdiff_t dim) 
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

    SYNET_INLINE void CubicCoeffs(float x, float k[4])
    {
        static const float a = -0.75f;
        x = std::abs(x);
        k[0] = ((a * (x + 1.0f) - 5.0f * a) * (x + 1.0f) + 8.0f * a) * (x + 1.0f) - 4.0f * a;
        k[1] = ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        k[2] = ((a + 2.0f) * (1.0f - x) - (a + 3.0f)) * (1.0f - x) * (1.0f - x) + 1.0f;
        k[3] = ((a * (2.0f - x) - 5.0f * a) * (2.0f - x) + 8.0f * a) * (2.0f - x) - 4.0f * a;
    }

    template <typename T> SYNET_INLINE T BicubicInterp(T p[4][4], float x, float y)
    {
        float v[4];
        float coeffs[4];
        CubicCoeffs(x, coeffs);
        for (int i = 0; i < 4; i++) 
            v[i] = coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
        CubicCoeffs(y, coeffs);
        return T(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
    }

    template <typename T, GridSamplePaddingMode padding> SYNET_INLINE T PixelAtGrid(const T* src, ptrdiff_t y, ptrdiff_t x, ptrdiff_t H, ptrdiff_t W, float border[4])
    {
        if (padding == GridSamplePaddingModeZeros) 
            return x >= 0 && x < W && y >= 0 && y < H ? src[y * W + x] : T(0);
        else if (padding == GridSamplePaddingModeBorder)
        {
            x = RestrictRange<ptrdiff_t>(x, 0, W - 1);
            y = RestrictRange<ptrdiff_t>(y, 0, H - 1);
            return src[y * W + x];
        }
        else if (padding == GridSamplePaddingModeReflection) 
        {
            x = ptrdiff_t(Reflect(T(x), border[0], border[2]));
            y = ptrdiff_t(Reflect(T(y), border[1], border[3]));
            return src[y * W + x];
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<class T, GridSampleInterpMode interp, GridSamplePaddingMode padding, bool corners> 
    void GridSample2d(const uint8_t* src8, size_t batch, size_t channels, size_t srcH, size_t srcW, const uint8_t* grid8, size_t dstH, size_t dstW, uint8_t* dst8)
    {
        const T* src = (const T*)src8;
        const T* grid = (const T*)grid8;
        T* dst = (T*)dst8;
        float border[4];
        if (corners)
        {
            border[0] = 0.0f;
            border[1] = 0.0f;
            border[2] = srcW - 1.0f;
            border[3] = srcH - 1.0f;
        }
        else
        {
            border[0] = -0.5f;
            border[1] = -0.5f;
            border[2] = srcW - 0.5f;
            border[3] = srcH - 0.5f;
        }
        for (size_t b = 0; b < batch; ++b)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                const T* gr = grid;
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    for (size_t dx = 0; dx < dstW; ++dx)
                    {
                        T x = Denormalize<T, corners>(gr[0], srcW);
                        T y = Denormalize<T, corners>(gr[1], srcH);
                        if (interp == GridSampleInterpModeNearest)
                        {
                            x = T(Round(float(x)));
                            y = T(Round(float(y)));
                        }
                        if (x < border[0] || x > border[2] || y < border[1] || y > border[3])
                        {
                            if (padding == GridSamplePaddingModeBorder) 
                            {
                                x = RestrictRange<T>(x, 0, T(srcW - 1));
                                y = RestrictRange<T>(y, 0, T(srcH - 1));
                            }
                            else if (padding == GridSamplePaddingModeReflection)
                            {
                                x = Reflect(x, border[0], border[2]);
                                y = Reflect(y, border[1], border[3]);
                            }
                        }

                        if (interp == GridSampleInterpModeNearest) 
                        {
                            dst[0] = PixelAtGrid<T, padding>(src, ptrdiff_t(y), ptrdiff_t(x), srcH, srcW, border);
                        }
                        if (interp == GridSampleInterpModeBilinear)
                        {
                            ptrdiff_t x1 = ptrdiff_t(std::floor(x));
                            ptrdiff_t y1 = ptrdiff_t(std::floor(y));
                            ptrdiff_t x2 = x1 + 1;
                            ptrdiff_t y2 = y1 + 1;

                            T p11 = PixelAtGrid<T, padding>(src, y1, x1, srcH, srcW, border);
                            T p12 = PixelAtGrid<T, padding>(src, y1, x2, srcH, srcW, border);
                            T p21 = PixelAtGrid<T, padding>(src, y2, x1, srcH, srcW, border);
                            T p22 = PixelAtGrid<T, padding>(src, y2, x2, srcH, srcW, border);

                            T dx2 = T(x2) - x;
                            T dx1 = x - T(x1);
                            T dy2 = T(y2) - y;
                            T dy1 = y - T(y1);
                            dst[0] = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);
                        }
                        if (interp == GridSampleInterpModeBicubic)
                        {
                            ptrdiff_t x0 = ptrdiff_t(std::floor(x)) - 1;
                            ptrdiff_t y0 = ptrdiff_t(std::floor(y)) - 1;
                            T p[4][4];
                            for (ptrdiff_t h = 0; h < 4; h++)
                                for (ptrdiff_t w = 0; w < 4; w++)
                                    p[h][w] = PixelAtGrid<T, padding>(src, h + y0, w + x0, srcH, srcW, border);
                            T dx1 = T(x - x0 - 1);
                            T dy1 = T(y - y0 - 1);
                            dst[0] = BicubicInterp(p, float(dx1), float(dy1));
                        }
                        gr += 2; dst += 1;
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
        : Layer(param, context)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        _context = NULL;
#endif
    }

    GridSampleLayer::~GridSampleLayer()
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        if (_context)
            SimdRelease(_context);
#endif
    }

    bool GridSampleLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (src.size() != 2 || dst.size() != 1)
            SYNET_ERROR("GridSampleLayer supports only 2 inputs and 1 output!");
        if (src[0]->GetType() != src[1]->GetType())
            SYNET_ERROR("GridSampleLayer don't src[0] type: " << Cpl::ToStr(src[0]->GetType()) << " != src[1] type "  << Cpl::ToStr(src[1]->GetType()) << " !");
        if (src[0]->GetType() != TensorType32f)
            SYNET_ERROR("GridSampleLayer support only TensorType32f inputs!");

        Shape srcShape = src[0]->Shape();
        Shape gridShape = src[1]->Shape();
        if (srcShape.size() != 4 || srcShape.size() != gridShape.size() || srcShape[0] != gridShape[0] || gridShape[3] != 2)
            SYNET_ERROR("GridSampleLayer has incompatible input shapes: src[0] :" << ToStr(srcShape) << " and src[1]: " << ToStr(gridShape) << " !");

        _type = src[0]->GetType();
        _batch = srcShape[0];
        _channels = srcShape[1];
        _srcH = srcShape[2];
        _srcW = srcShape[3];
        _dstH = gridShape[1];
        _dstW = gridShape[2];
        Shape dstShape = Shp(_batch, _channels, _dstH, _dstW);

        GridSampleParam gridSample = this->Param().gridSample();
        _interp = gridSample.interpMode();
        _padding = gridSample.paddingMode();
        _align = gridSample.alignCorners();

        _gridSample2d = GetGridSample2d(_type, _interp, _padding, _align);
        if (_gridSample2d == NULL)
            SYNET_ERROR("GridSampleLayer can't get worker!");

        dst[0]->Reshape(src[0]->GetType(), dstShape, src[0]->Format());

        std::stringstream desc;
        desc << _batch << "x" << _channels << "x" << _srcH << "x" << _srcW;
        desc << "-" << _dstH << "x" << _dstW;
        this->UsePerfStat(desc.str());

#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        _context = SimdSynetGridSample2dInit(_batch, _channels, _srcH, _srcW, _dstH, _dstW, (SimdTensorDataType)_type,
            (SimdGridSampleInterpType)_interp, (SimdGridSamplePaddingType)_padding, _align ? SimdTrue : SimdFalse);
#endif

        return true;
    }

    size_t GridSampleLayer::MemoryUsage() const
    {
        size_t size = Layer::MemoryUsage();
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        if (_context)
            size += SimdSynetGridSample2dInternalBufferSize(_context);
#endif
        return size;
    }

    void GridSampleLayer::Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
        if (_context)
        {
            SimdSynetGridSample2dForward(_context, src[0]->RawData(), src[1]->RawData(), dst[0]->RawData());
            return;
        }
#endif
        _gridSample2d(src[0]->RawData(), _batch, _channels, _srcH, _srcW, src[1]->RawData(), _dstH, _dstW, dst[0]->RawData());
    }
}
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
#include "Synet/Utils/Math.h"
#include "Synet/Utils/SetInput.h"

#include "Synet/Network.h"

namespace Synet
{
#ifdef SYNET_SIMD_LIBRARY_ENABLE

#if defined(SYNET_SIMD_SYNET_DISABLE)
    namespace Detail
    {
        SYNET_INLINE int BgrToGray(int blue, int green, int red)
        {
            return Round(blue * 0.114 + green * 0.587 + red * 0.299);
        }

        SYNET_INLINE float Convert8uTo32f(int value, float scale, float shift)
        {
            return value * scale + shift;
        }

        template<SimdPixelFormatType format> SYNET_INLINE int ToGray(const uint8_t* src);

        template<> SYNET_INLINE int ToGray<SimdPixelFormatGray8>(const uint8_t* src)
        {
            return src[0];
        }

        template<> SYNET_INLINE int ToGray<SimdPixelFormatBgr24>(const uint8_t* src)
        {
            return BgrToGray(src[0], src[1], src[2]);
        }

        template<> SYNET_INLINE int ToGray<SimdPixelFormatRgb24>(const uint8_t* src)
        {
            return BgrToGray(src[2], src[1], src[0]);
        }

        template<SimdPixelFormatType format, size_t step> void SetInput1(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                    *dst++ = Convert8uTo32f(ToGray<format>(src), scale[0], shift[0]);
                src += (stride - width * step);
            }
        }

        template<SimdPixelFormatType format> SYNET_INLINE int ToBgr(const uint8_t* src, size_t channel);

        template<> SYNET_INLINE int ToBgr<SimdPixelFormatGray8>(const uint8_t* src, size_t channel)
        {
            return src[0];
        }

        template<> SYNET_INLINE int ToBgr<SimdPixelFormatBgr24>(const uint8_t* src, size_t channel)
        {
            return src[channel];
        }

        template<> SYNET_INLINE int ToBgr<SimdPixelFormatRgb24>(const uint8_t* src, size_t channel)
        {
            return src[2 - channel];
}

        template<SimdPixelFormatType format, size_t step> void SetInputNchw3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst0)
        {
            float* dst1 = dst0 + width * height;
            float* dst2 = dst1 + width * height;
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                {
                    *dst0++ = Convert8uTo32f(ToBgr<format>(src, 0), scale[0], shift[0]);
                    *dst1++ = Convert8uTo32f(ToBgr<format>(src, 1), scale[1], shift[1]);
                    *dst2++ = Convert8uTo32f(ToBgr<format>(src, 2), scale[2], shift[2]);
                }
                src += (stride - width * step);
            }
        }

        template<SimdPixelFormatType format, size_t step> void SetInputNhwc3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                {
                    *dst++ = Convert8uTo32f(ToBgr<format>(src, 0), scale[0], shift[0]);
                    *dst++ = Convert8uTo32f(ToBgr<format>(src, 1), scale[1], shift[1]);
                    *dst++ = Convert8uTo32f(ToBgr<format>(src, 2), scale[2], shift[2]);
                }
                src += (stride - width * step);
            }
        }

        inline void SetInput(const uint8_t* src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float* lower, const float* upper, float* dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            float scale[3];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SetInput1<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SetInputNchw3<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SetInputNhwc3<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
            default: assert(0);
            }
        }
    }
#endif

    bool SetInput(class Network & network, const Views & views, Floats lower, Floats upper)
    {
        SYNET_PERF_FUNC();

        if (network.Src().size() != 1 || views.empty() || lower.size() != upper.size())
            return false;
        if (network.Src()[0]->GetType() != TensorType32f)
            return false;
        const Shape & shape = network.NchwShape();
        if (shape.size() != 4 || shape[0] != views.size())
            return false;
        if (shape[1] != 1 && shape[1] != 3)
            return false;
        if (lower.size() != 1 && lower.size() != shape[1])
            return false;
        for (size_t i = 0; i < views.size(); ++i)
        {
            if (views[i].width != shape[3] || views[i].height != shape[2] || views[i].format != views[0].format)
                return false;
            if (views[i].format != View::Gray8 && views[i].format != View::Bgr24 &&
                views[i].format != View::Bgra32 && views[i].format != View::Rgb24)
                return false;
        }
        if (lower.size() == 1)
            lower.resize(shape[1], lower[0]);
        if (upper.size() == 1)
            upper.resize(shape[1], upper[0]);
        float * dst = network.Src()[0]->CpuData();
        for (size_t i = 0; i < views.size(); ++i)
        {
#if defined(SYNET_SIMD_SYNET_DISABLE)
            Detail::SetInput(views[i].data, views[i].width, views[i].height, views[i].stride, (SimdPixelFormatType)views[i].format,
                lower.data(), upper.data(), dst, shape[1], (SimdTensorFormatType)network.Format());
#else
            SimdSynetSetInput(views[i].data, views[i].width, views[i].height, views[i].stride, (SimdPixelFormatType)views[i].format,
                lower.data(), upper.data(), dst, shape[1], (SimdTensorFormatType)network.Format());
#endif
            dst += shape[1] * shape[2] * shape[3];
        }
        return true;
    }
#endif
}

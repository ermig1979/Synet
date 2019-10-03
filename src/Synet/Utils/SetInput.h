/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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

#include "Synet/Common.h"

namespace Synet
{
#ifdef SYNET_SIMD_LIBRARY_ENABLE
    template <template<class> class Network> bool SetInput(Network<float> & network, const Views & views, Floats lower, Floats upper)
    {
        if (network.Src().size() != 1 || views.empty() || lower.size() != upper.size())
            return false;
        const Shape & shape = network.NchwShape();
        if (shape.size() != 4 || shape[0] != views.size())
            return false;
        if (lower.size() != 1 && lower.size() != shape[1])
            return false;
        for (size_t i = 0; i < views.size(); ++i)
            if (views[i].width != shape[3] || views[i].height != shape[2] || views[i].ChannelCount() != shape[1] || views[i].ChannelSize() != 1)
                return false;
        if (lower.size() == 1)
            lower.resize(shape[1], lower[0]);
        if (upper.size() == 1)
            upper.resize(shape[1], upper[0]);
        float * dst = network.Src()[0]->CpuData();
        if (network.Src()[0]->Format() == TensorFormatNchw)
        {
            for (size_t i = 0; i < views.size(); ++i)
            {
                View tmp[3];
                if (shape[1] == 3)
                {
                    for (size_t c = 0; c < shape[1]; ++c)
                        tmp[c].Recreate(views[i].Size(), View::Gray8);
                    Simd::DeinterleaveBgr(views[i], tmp[0], tmp[1], tmp[2]);
                }
                else
                    tmp[0] = views[i];
                for (size_t c = 0, e = 0; c < shape[1]; ++c)
                {
                    for (size_t y = 0; y < tmp[c].height; ++y)
                    {
                        ::SimdUint8ToFloat32(tmp[c].Row<uint8_t>(y), tmp[c].width, &lower[c], &upper[c], dst);
                        dst += tmp[c].width;
                    }
                }
            }
            return true;
        }
        else if (network.Src()[0]->Format() == TensorFormatNhwc)
        {
            for (size_t i = 0; i < views.size(); ++i)
            {
                for (size_t y = 0; y < views[i].height; ++y)
                {
                    ::SimdUint8ToFloat32(views[i].Row<uint8_t>(y), shape[1] * shape[3], &lower[0], &upper[0], dst);
                    dst += shape[1] * shape[3];
                }
            }
            return true;
        }
        else
            return false;

    }
#endif
}

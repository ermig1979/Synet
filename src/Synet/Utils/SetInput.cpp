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
    bool SetInput(class Network & network, const Views & views, Floats lower, Floats upper, bool rgb, size_t thread)
    {
        SYNET_PERF_FUNC();
        if(thread >= network.GetThreads())
            SYNET_ERROR("SetInput: network supports only " << network.GetThreads() << " threads but current is " << thread << " !");
        if (network.Src(thread).size() != 1)
            SYNET_ERROR("SetInput can process only models with one input!");
        if (views.empty() || lower.size() != upper.size())
            SYNET_ERROR("SetInput: check input parameters!");
        const Shape & shape = network.NchwShape();
        if (network.Src(thread)[0]->GetType() != TensorType32f || shape.size() != 4)
            SYNET_ERROR("SetInput: network input must be 4D FP32 tensor!");
        if (shape[0] != views.size())
            SYNET_ERROR("SetInput: wrong batch!");
        if (shape[1] != 1 && shape[1] != 3)
            SYNET_ERROR("SetInput: wrong channels!");
        if (lower.size() != 1 && lower.size() != shape[1])
            SYNET_ERROR("SetInput: wrong 'lower' parameter!");
        for (size_t i = 0; i < views.size(); ++i)
        {
            if (views[i].width != shape[3] || views[i].height != shape[2] || views[i].format != views[0].format)
                SYNET_ERROR("SetInput: wrong size of input image!");
            if (views[i].format != View::Gray8 && views[i].format != View::Bgr24 &&
                views[i].format != View::Bgra32 && views[i].format != View::Rgb24 && views[i].format != View::Rgba32)
                SYNET_ERROR("SetInput: wrong format of input image!");
        }
        if (lower.size() == 1)
            lower.resize(shape[1], lower[0]);
        if (upper.size() == 1)
            upper.resize(shape[1], upper[0]);
        float * dst = network.Src(thread)[0]->Data<float>();
        for (size_t i = 0; i < views.size(); ++i)
        {
            Simd::SynetSetInput(views[i], lower.data(), upper.data(), dst, shape[1], (SimdTensorFormatType)network.Format(), rgb);
            dst += shape[1] * shape[2] * shape[3];
        }
        return true;
    }
#endif
}

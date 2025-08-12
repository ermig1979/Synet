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

#include "Synet/Common.h"
#include "Synet/Quantization/Bf16.h"

namespace Synet
{
    struct Options
    {
        enum PerfomanceLog
        {
            PerfomanceLogEmpty = 0,
            PerfomanceLogLayer,
            PerfomanceLogSize,
            PerfomanceLogSubnet,
        };

        enum Bf16Support
        {
            Bf16SupportNone = 0,
            Bf16SupportHard,
            Bf16SupportSoft
        };

        PerfomanceLog performanceLog;
        Bf16Support bf16Support;

        Options()
        {
            performanceLog = PerfomanceLogEmpty;
            bf16Support = Bf16SupportHard;
        }

        bool BFloat16Enable() const
        {
            return (bf16Support == Bf16SupportSoft) || (bf16Support == Bf16SupportHard && BFloat16HardwareSupport());
        }
    };

    struct Context
    {
        Options options;
        std::map<String, size_t> tensorUsers;
        size_t batchSize;

        Context()
            : batchSize(1)
        {
        }

        void Clear()
        {
            tensorUsers.clear();
        }
    };
}
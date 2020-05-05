/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#include "TestCommon.h"
#include "TestOptions.h"

namespace Test
{
    namespace Quantization
    {
        struct Options : public Test::OptionsBase
        {
            String mode;
            String binPath;
            String fp32Path;
            String tempPath;
            String int8Path;
            String testImages;

            Options(int argc, char* argv[])
                : OptionsBase(argc, argv)
            {
                mode = GetArg("-mode");

                binPath = GetArg("-bp", "data.bin");
                fp32Path = GetArg("-fp", "fp32.xml");
                tempPath = GetArg("-tp", "temp.xml");
                int8Path = GetArg("-ip", "int8.xml");
                testImages = GetArg("-ti", "../../images/persons/");
            }
        };
    }
}



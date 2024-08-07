/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#define FIRST_MODEL_DEFAULT "synet.xml"
#define FIRST_WEIGHT_DEFAULT "synet.bin"
#define SECOND_MODEL_DEFAULT "int8.xml"
#define IMAGE_DIRECTORY_DEFAULT ""

#include "TestCompare.h"
#include "TestReport.h"
#include "TestQuantization.h"

namespace Test
{
    struct Synet8iNetwork : public SynetNetwork
    {
        virtual String Type() const
        {
            return "int8";
        }
    };
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    if (options.mode == "convert")
    {
        Test::Quantizer quantizer(options);
        options.result = quantizer.Run();
    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::SynetNetwork, Test::Synet8iNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else
        CPL_LOG_SS(Error, "Unknown mode : " << options.mode);

    return options.result ? 0 : 1;
}
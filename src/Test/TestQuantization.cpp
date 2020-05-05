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

#include "Test/TestCompare.h"
#include "Test/TestReport.h"
#include "Test/TestQuantization.h"

#include "Synet/Quantization/Unpack.h"

Test::PerformanceMeasurerStorage Test::PerformanceMeasurerStorage::s_storage;

int main(int argc, char* argv[])
{
    Test::Quantization::Options options(argc, argv);
    Test::Quantization::TestParamHolder param;
    Test::String path = Test::MakePath(options.test, options.param);
    if (!param.Load(path))
    {
        std::cout << "Can't load test param file '" << path  << "' !" << std::endl;
        return 1;
    }

    if (options.mode == "unpack")
    {
        SYNET_PERF_FUNC();
        std::cout << "Unpack network complex layers : ";
        options.result = Synet::Quantization::UnpackComplexLayers(
            Test::MakePath(options.test, param().fp32()), Test::MakePath(options.test, param().temp()));
        std::cout << (options.result ? "OK." : " Unpacking finished with errors!") << std::endl;
    }
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}
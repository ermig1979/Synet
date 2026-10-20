/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#define FIRST_MODEL_DEFAULT "synet1.xml"
#define FIRST_WEIGHT_DEFAULT "synet1.bin"
#define SECOND_MODEL_DEFAULT "synet3.xml"
#define SECOND_WEIGHT_DEFAULT "synet3.bin"

#include "TestCompare.h"
#include "TestReport.h"

#include "Cvt/Int8/Int8Quantizer.h"

namespace Test
{
    bool ConvertFp32ToInt8(const Test::Options& options)
    {
        Test::TestParamHolder param;
        if (FileExists(options.testParam) && !param.Load(options.testParam))
            SYNET_ERROR("Can't load file '" << options.testParam << "' !");
        param().optimizer().int8().enable() = true;
        return Synet::Int8QuantizeSynetModel(options.firstModel, options.firstWeight, options.secondModel, options.secondWeight, param().optimizer().int8());
    }

    struct SynetFp32Network : public SynetNetwork
    {
        virtual String Type() const
        {
            return "fp32";
        }

        virtual int PerfLogMask() const
        {
            return 0;
        }
    };

    struct SynetInt8Network : public SynetNetwork
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
        int64_t start = Cpl::TimeCounter();
        std::cout << "Convert Synet FP32 model to INT8 :";
        options.result = Test::ConvertFp32ToInt8(options);
        if (options.result)
            std::cout << " finished successfully in " << Test::ExecTimeStr(start) << "." << std::endl;
        else
            std::cout << " Conversion finished with errors!" << std::endl;
    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::SynetFp32Network, Test::SynetInt8Network> comparer(options);
        options.result = comparer.Run();
    }
    else
        CPL_LOG_SS(Error, "Unknown mode : " << options.mode);

    return options.result ? 0 : 1;
}

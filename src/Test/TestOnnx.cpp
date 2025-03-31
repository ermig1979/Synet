/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#define FIRST_MODEL_DEFAULT "other.onnx"
#define FIRST_WEIGHT_DEFAULT "other.onnx"

#include "TestCompare.h"
#include "TestReport.h"

#if defined(SYNET_TEST_FIRST_RUN) && defined(SYNET_ONNXRUNTIME_ENABLE)
#include "Synet/Converters/OnnxRuntime.h"
#include "TestOnnxRuntime.h"

namespace Test
{
    OnnxRuntimeNetwork::Env OnnxRuntimeNetwork::s_env;

    bool ConvertOnnxToSynet(const Test::Options &options)
    {
        Test::TestParamHolder param;
        if (FileExists(options.testParam) && !param.Load(options.testParam))
            SYNET_ERROR("Can't load file '" << options.testParam << "' !");
        param().optimizer().bf16().enable() = options.bf16;
        param().optimizer().saveUnoptimized() = options.saveUnoptimized;
        return Synet::ConvertOnnxToSynet(options.firstWeight, options.tensorFormat == 1, options.secondModel, options.secondWeight, param().onnx(), param().optimizer());
    }
}
#else
namespace Test
{
    struct OnnxRuntimeNetwork : public Network
    {
    };
}
#endif

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

#if defined(SYNET_ONNXRUNTIME_ENABLE)
    if (options.mode == "convert")
    {
        int64_t start = Cpl::TimeCounter();
        std::cout << "Convert network from Onnx to Synet :" << std::flush;
        options.result = Test::ConvertOnnxToSynet(options);
        if (options.result)
            std::cout << " finished successfully in " << Test::ExecTimeStr(start) << "." << std::endl;
        else
            std::cout << " Conversion finished with errors!" << std::endl;
    }
    else 
#endif
    if (options.mode == "compare")
    {
        Test::Comparer<Test::OnnxRuntimeNetwork, Test::SynetNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else
    {
        CPL_LOG_SS(Error, "Unknown mode : " << options.mode);
    }

    return options.result ? 0 : 1;
}



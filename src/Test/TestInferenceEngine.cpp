/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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

#include "TestCompare.h"
#include "TestReport.h"

#include "Synet/Converters/InferenceEngine.h"

#ifdef SYNET_TEST_FIRST_RUN

#if defined(SYNET_TEST_OPENVINO_API)
#include "TestOpenVino.h"
#else
#include "TestInferenceEngine.h"
#endif

#else //SYNET_FIRST_RUN
namespace Test
{
    struct InferenceEngineNetwork : public Network
    {
    };
}
#endif//SYNET_FIRST_RUN

namespace Test
{
    bool ConvertTextWeightToBinary(const String& src, const String& dst)
    {
        std::ifstream ifs(src.c_str());
        if (!ifs.is_open())
        {
            std::cout << "Can't open input text file '" << src << "' with weight!" << std::endl;
            return false;
        }
        std::ofstream ofs(dst.c_str(), std::ofstream::binary);
        if (!ofs.is_open())
        {
            std::cout << "Can't open output binary file '" << dst << "' with weight!" << std::endl;
            ifs.close();
            return false;
        }
        while (!ifs.eof())
        {
            std::string str;
            ifs >> str;
            float val = std::stof(str);
            ofs.write((char*)&val, 4);
        }
        ifs.close();
        ofs.close();
        return true;
    }

    bool ConvertInferenceEngineToSynet(const Test::Options & options)
    {
        CPL_PERF_FUNC();
        Test::TestParamHolder param;
        if (FileExists(options.testParam) && !param.Load(options.testParam))
        {
            std::cout << "Can't load file '" << options.testParam << "' !" << std::endl;
            return false;
        }
        param().optimizer().bf16Enable() = options.bf16;
        return Synet::ConvertInferenceEngineToSynet(options.firstModel, options.firstWeight, 
            options.tensorFormat == 1, options.secondModel, options.secondWeight, param().optimizer());
    }
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    if (options.mode == "convert")
    {
        std::cout << "Convert network from Inference Engine to Synet : ";
        options.result = ConvertInferenceEngineToSynet(options);
        std::cout << (options.result ? "OK." : " Conversion finished with errors!") << std::endl;
    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::InferenceEngineNetwork, Test::SynetNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else if (options.mode == "txt2bin")
    {
        std::cout << "Convert text weight to binary : ";
        options.result = Test::ConvertTextWeightToBinary(options.textWeight, options.firstWeight);
        std::cout << (options.result ? "OK." : " Conversion finished with errors!") << std::endl;
    }
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}
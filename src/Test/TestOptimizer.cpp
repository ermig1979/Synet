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

#define FIRST_MODEL_DEFAULT "unopt.xml"
#define FIRST_WEIGHT_DEFAULT "unopt.bin"

#include "TestCompare.h"
#include "TestReport.h"

#include "Synet/Converters/Optimizer.h"

namespace Test
{
    bool OptimizeSynet(const Test::Options& options)
    {
        CPL_PERF_FUNC();
        Test::TestParamHolder param;
        if (FileExists(options.testParam) && !param.Load(options.testParam))
        {
            std::cout << "Can't load file '" << options.testParam << "' !" << std::endl;
            return false;
        }
        return Synet::OptimizeSynetModel(options.firstModel, options.firstWeight, options.secondModel, options.secondWeight, param().optimizer());
    }

    struct SynetSrcNetwork : public SynetNetwork
    {
        virtual String Type() const
        {
            return "src";
        }
    };

    struct SynetOptNetwork : public SynetNetwork
    {
        virtual String Type() const
        {
            return "opt";
        }
    };
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    if (options.mode == "convert")
    {
        std::cout << "Optimize Synet network : ";
        options.result = OptimizeSynet(options);
        std::cout << (options.result ? "OK." : " Optimization finished with errors!") << std::endl;

    }
    else if (options.mode == "compare")
    {
        Test::Comparer<Test::SynetSrcNetwork, Test::SynetOptNetwork> comparer(options);
        options.result = comparer.Run();
    }
    else
        std::cout << "Unknown mode : " << options.mode << std::endl;

    return options.result ? 0 : 1;
}
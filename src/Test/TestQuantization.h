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

#include "Synet/Converters/Deoptimizer.h"
#include "Synet/Converters/Optimizer.h"

namespace Test
{
    class Quantizer
    {
    public:
        Quantizer(const Options & options)
            : _options(options)
        {
            if (_options.debugPrint)
            {
                _deopt = Test::MakePath(_options.outputDirectory, "fp32_deoptimized.xml");
            }
            else
            {
                _deopt = _options.otherModel;
            }
        }

        bool Run()
        {
            if (!CreateDirectories())
                return false;
            if (!Deoptimize())
                return false;
            return true;
        }

    private:
        const Options& _options;
        String _deopt;

        bool CreateDirectories()
        {
            if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
            {
                std::cout << "Can't create output directory '" << _options.outputDirectory << "' !" << std::endl;
                return false;
            }
            return true;
        }

        bool Deoptimize()
        {
            if(!_options.consoleSilence)
                std::cout << "Deoptimize Synet model : ";
            bool result = Synet::DeoptimizeSynetModel(_options.synetModel, _deopt);
            if (_options.debugPrint)
                Synet::OptimizeSynetModel(_deopt, "", Test::MakePath(_options.outputDirectory, "fp32_optimized_back.xml"), "");
            if (!_options.consoleSilence)
                std::cout << (result ? "OK." : " Deoptimization is finished with errors!") << std::endl;
            return result;
        }
    };

}



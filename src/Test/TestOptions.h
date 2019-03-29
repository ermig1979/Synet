/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

#include "TestPerformance.h"

namespace Test
{
    struct Options
    {
        String mode;
        String otherModel;
        String otherWeight;
        String synetModel;
        String synetWeight;
        String testParam;
        String imageDirectory;
        String imageFilter;
        String outputDirectory;
        size_t repeatNumber;
        size_t threadNumber;
        float threshold;
        String logName;
        int tensorFormat;
        bool result;        
        mutable size_t synetMemoryUsage;

        Options(int argc, char* argv[])
            : _argc(argc)
            , _argv(argv)
            , result(false)
            , synetMemoryUsage(0)
        {
            mode = GetArg("-m");
            otherModel = GetArg("-om", "./other.dsc");
            otherWeight = GetArg("-ow", "./other.dat");
            synetModel = GetArg("-sm", "./synet.xml");
            synetWeight = GetArg("-sw", "./synet.bin");
            testParam = GetArg("-tp", "./param.xml");
            imageDirectory = GetArg("-id", "./image");
            imageFilter = GetArg("-if", "*.ppm");
            outputDirectory = GetArg("-od", "./output");
            repeatNumber = FromString<size_t>(GetArg("-rn", "1"));
            threadNumber = FromString<size_t>(GetArg("-tn", "1"));
            threshold = FromString<float>(GetArg("-t", "0.001"));
            logName = GetArg("-ln", "", false);
            tensorFormat = FromString<int>(GetArg("-tf", "0"));
        }

        ~Options()
        {
            if (mode == "compare" && result)
            {
                std::stringstream ss;
                if (synetMemoryUsage)
                    ss << "Synet memory usage: " << synetMemoryUsage / (1024 * 1024) << " MB." << std::endl;
                PerformanceMeasurerStorage::s_storage.Print(ss);
                std::cout << ss.str();
                if (!logName.empty())
                {
                    std::ofstream log(logName.c_str());
                    if (log.is_open())
                    {
                        log << ss.str();
                        log.close();
                    }
                }
            }
        }

    private:
        String GetArg(const String & name, const String & default_ = String(), bool exit = true)
        {
            Strings values;
            for (int i = 1; i < _argc; ++i)
            {
                String arg = _argv[i];
                if (arg.substr(0, name.size()) == name && arg.substr(name.size(), 1) == "=")
                    values.push_back(arg.substr(name.size() + 1));
            }
            if (values.empty())
            {
                if (default_.empty() && exit)
                {
                    std::cout << "Argument '" << name << "' is absent!" << std::endl;
                    ::exit(1);
                }
                else
                    return default_;
            }
            return values[0];
        }

        int _argc;
        char ** _argv;
    };
}


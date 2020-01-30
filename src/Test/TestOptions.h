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

#include "TestPerformance.h"

namespace Test
{
    const int ENABLE_OTHER = 1;
    const int ENABLE_SYNET = 2;

    struct Options
    {
        String mode;
        int enable;
        String textWeight;
        String otherModel;
        String otherWeight;
        String synetModel;
        String synetWeight;
        String testParam;
        String imageDirectory;
        String imageFilter;
        String outputDirectory;
        size_t repeatNumber;
        size_t workThreads;
        size_t testThreads;
        float threshold;
        String logName;
        int tensorFormat;
        int batchSize;
        int debugPrint;
        int debugPrintFirst;
        int debugPrintLast;
        int debugPrintPrecision;
        int annotateRegions;
        float regionThreshold;
        float regionOverlap;
        mutable bool result;
        mutable size_t synetMemoryUsage;

        Options(int argc, char* argv[])
            : _argc(argc)
            , _argv(argv)
            , result(false)
            , synetMemoryUsage(0)
        {
            mode = GetArg("-m");
            enable = FromString<int>(GetArg("-e", "3"));
            textWeight = GetArg("-tw", "./other.txt");
            otherModel = GetArg("-om", "./other.dsc");
            otherWeight = GetArg("-ow", "./other.dat");
            synetModel = GetArg("-sm", "./synet.xml");
            synetWeight = GetArg("-sw", "./synet.bin");
            testParam = GetArg("-tp", "./param.xml");
            imageDirectory = GetArg("-id", "./image");
            imageFilter = GetArg("-if", "*.ppm");
            outputDirectory = GetArg("-od", "./output");
            repeatNumber = FromString<size_t>(GetArg("-rn", "1"));
            workThreads = FromString<size_t>(GetArg("-wt", "1"));
            testThreads = FromString<size_t>(GetArg("-tt", "0"));
            threshold = FromString<float>(GetArg("-t", "0.001"));
            logName = GetArg("-ln", "", false);
            tensorFormat = FromString<int>(GetArg("-tf", "1"));
            batchSize = FromString<int>(GetArg("-bs", "1"));
            debugPrint = FromString<int>(GetArg("-dp", "0"));
            debugPrintFirst = FromString<int>(GetArg("-dpf", "5"));
            debugPrintLast = FromString<int>(GetArg("-dpl", "2"));
            debugPrintPrecision = FromString<int>(GetArg("-dpp", "4"));
            annotateRegions = FromString<int>(GetArg("-ar", "0"));
            regionThreshold = FromString<float>(GetArg("-rt", "0.3"));
            regionOverlap = FromString<float>(GetArg("-ro", "0.5"));
            if (enable < 1 || enable > 3)
            {
                std::cout << "Parameter '-e' (enable) must be only 1, 2, 3!" << std::endl;
                ::exit(1);
            }
        }

        ~Options()
        {
            if (mode == "compare" && result)
            {
                std::stringstream ss;
                if (synetMemoryUsage)
                    ss << "Synet memory usage: " << synetMemoryUsage / (1024 * 1024) << " MB." << std::endl;
                PerformanceMeasurerStorage::s_storage.Print(ss);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
                if (enable & ENABLE_SYNET)
                    ss << SimdPerformanceStatistic();
#endif
                std::cout << ss.str();
                if (!logName.empty())
                {
                    String dir = DirectoryByPath(logName);
                    if (!DirectoryExists(dir) && !CreatePath(dir))
                    {
                        std::cout << "Can't create output directory '" << dir << "' !" << std::endl;
                        return;
                    }
                    std::ofstream log(logName.c_str());
                    if (log.is_open())
                    {
                        log << ss.str();
                        log.close();
                    }
                }
            }
        }

        bool NeedOutputDirectory() const
        {
            return debugPrint || annotateRegions;
        }

        size_t TestThreads() const
        {
            return std::max<size_t>(1, testThreads);
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


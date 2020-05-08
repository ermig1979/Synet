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

    void GenerateReport(const struct Options& options);

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
        double executionTime;
        size_t workThreads;
        size_t testThreads;
        float threshold;
        String logName;
        bool consoleSilence;
        String syncName;
        double skipThreshold;
        String textReport;
        String htmlReport;
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
        mutable size_t synetMemoryUsage, otherMemoryUsage;
        mutable String synetName, otherName, synetType, otherType;

        Options(int argc, char* argv[])
            : _argc(argc)
            , _argv(argv)
            , result(false)
            , synetMemoryUsage(0)
            , otherMemoryUsage(0)
        {
            mode = GetArg2("-m", "-mode");
            enable = FromString<int>(GetArg2("-e", "-enable", "3"));
            textWeight = GetArg("-tw", "other.txt");
            otherModel = GetArg("-om", QuantizationTest() ? "int8.xml" : "other.dsc");
            otherWeight = GetArg("-ow", QuantizationTest() ? "synet.bin" : "other.dat");
            synetModel = GetArg("-sm", "synet.xml");
            synetWeight = GetArg("-sw", "synet.bin");
            testParam = GetArg("-tp", "param.xml");
            imageDirectory = GetArg("-id", QuantizationTest() ? "" : "image");
            imageFilter = GetArg("-if", "*.*");
            outputDirectory = GetArg("-od", "output");
            repeatNumber = std::max(0, FromString<int>(GetArg("-rn", "1")));
            executionTime = FromString<double>(GetArg("-et", "10.0"));
            workThreads = FromString<size_t>(GetArg("-wt", "1"));
            testThreads = FromString<size_t>(GetArg("-tt", "0"));
            threshold = FromString<float>(GetArg("-t", "0.001"));
            logName = GetArg("-ln", "", false);
            consoleSilence = FromString<bool>(GetArg("-cs", "0"));
            syncName = GetArg("-sn", "", false);
            skipThreshold = FromString<double>(GetArg("-st", "20.0"));
            textReport = GetArg("-tr", "", false);
            htmlReport = GetArg("-hr", "", false);
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
                if(!consoleSilence)
                    std::cout << ss.str();
                if (!logName.empty())
                {
                    if (!CreateOutputDirectory(logName))
                        return;
                    std::ofstream log(logName.c_str());
                    if (log.is_open())
                    {
                        log << ss.str();
                        log.close();
                    }
                    GenerateReport(*this);
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

        bool QuantizationTest() const
        {
            return mode == "quantize";
        }

    protected:
        String GetArg(const String& name, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name }, { default_ }, exit)[0];
        }

        String GetArg2(const String& name1, const String& name2, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name1, name2 }, { default_ }, exit)[0];
        }

        Strings GetArgs(const Strings& names, const Strings& defaults = Strings(), bool exit = true)
        {
            Strings values;
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name && arg.substr(name.size(), 1) == "=")
                        values.push_back(arg.substr(name.size() + 1));
                }
            }
            if (values.empty())
            {
                if (defaults.empty() && exit)
                {
                    std::cout << "Argument '";
                    for (size_t n = 0; n < names.size(); ++n)
                        std::cout << (n ? " | " : "") << names[n];
                    std::cout << "' is absent!" << std::endl;
                    ::exit(1);
                }
                else
                    return defaults;
            }
            return values;
        }
    private:
        int _argc;
        char** _argv;
    };
}


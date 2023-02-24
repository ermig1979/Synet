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

#pragma once

#ifndef FIRST_MODEL_DEFAULT
#define FIRST_MODEL_DEFAULT "other.dsc"
#endif

#ifndef FIRST_WEIGHT_DEFAULT
#define FIRST_WEIGHT_DEFAULT "other.dat"
#endif

#ifndef SECOND_MODEL_DEFAULT
#define SECOND_MODEL_DEFAULT "synet.xml"
#endif

#ifndef SECOND_WEIGHT_DEFAULT
#define SECOND_WEIGHT_DEFAULT "synet.bin"
#endif

#ifndef IMAGE_DIRECTORY_DEFAULT
#define IMAGE_DIRECTORY_DEFAULT "image"
#endif

#include "Cpl/Args.h"

#include "TestPerformance.h"

namespace Test
{
    const int ENABLE_FIRST = 1;
    const int ENABLE_SECOND = 2;

    void GenerateReport(const struct Options& options);

    struct Options : public Cpl::ArgsParser
    {
        String mode;
        int enable;
        String textWeight;
        String firstModel;
        String firstWeight;
        String secondModel;
        String secondWeight;
        String testParam;
        String quantParam;
        String imageDirectory;
        String imageFilter;
        size_t imageBegin;
        size_t imageEnd;
        size_t binaryBegin;
        size_t binaryEnd;
        String outputDirectory;
        size_t repeatNumber;
        double executionTime;
        size_t workThreads;
        size_t testThreads;
        float compareThreshold;
        double compareQuantile;
        String logName;
        bool consoleSilence;
        bool reverseExecution;
        String syncName;
        double skipThreshold;
        String textReport;
        String htmlReport;
        int tensorFormat;
        int batchSize;
        int performanceLog;
        int debugPrint;
        int debugPrintFirst;
        int debugPrintLast;
        int debugPrintPrecision;
        int annotateRegions;
        float regionThreshold;
        float regionOverlap;
        double statFilter;
        bool bf16;
        bool saveUnoptimized;

        mutable bool result;
        mutable size_t firstMemoryUsage,  secondMemoryUsage;
        mutable String firstName, firstType, secondName, secondType, statistics;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
            , result(false)
            , secondMemoryUsage(0)
            , firstMemoryUsage(0)
        {
            mode = GetArg2("-m", "-mode");
            enable = FromString<int>(GetArg2("-e", "-enable", "3"));
            textWeight = GetArg("-tw", "other.txt");
            firstModel = GetArg("-fm", FIRST_MODEL_DEFAULT);
            firstWeight = GetArg("-fw", FIRST_WEIGHT_DEFAULT);
            secondModel = GetArg("-sm", SECOND_MODEL_DEFAULT);
            secondWeight = GetArg("-sw", SECOND_WEIGHT_DEFAULT);
            testParam = GetArg("-tp", "param.xml");
            quantParam = GetArg("-qp", "quant.xml");
            imageDirectory = GetArg("-id", IMAGE_DIRECTORY_DEFAULT);
            imageFilter = GetArg("-if", "*.*");
            imageBegin = FromString<size_t>(GetArg("-ib", "0"));
            imageEnd = FromString<size_t>(GetArg("-ie", "1000000"));
            binaryBegin = FromString<size_t>(GetArg("-bb", "0"));
            binaryEnd = FromString<size_t>(GetArg("-be", "1000000"));
            outputDirectory = GetArg("-od", "output");
            repeatNumber = std::max(0, FromString<int>(GetArg("-rn", "1")));
            executionTime = FromString<double>(GetArg("-et", "10.0"));
            workThreads = FromString<size_t>(GetArg("-wt", "1"));
            testThreads = FromString<size_t>(GetArg("-tt", "0"));
            compareThreshold = FromString<float>(GetArg("-ct", "0.001"));
            compareQuantile = FromString<double>(GetArg("-cq", "0.0"));
            logName = GetArg("-ln", "", false);
            consoleSilence = FromString<bool>(GetArg("-cs", "0"));
            reverseExecution = FromString<bool>(GetArg("-re", "0"));
            syncName = GetArg("-sn", "", false);
            skipThreshold = FromString<double>(GetArg("-st", "20.0"));
            textReport = GetArg("-tr", "", false);
            htmlReport = GetArg("-hr", "", false);
            tensorFormat = FromString<int>(GetArg("-tf", "1"));
            batchSize = FromString<int>(GetArg("-bs", "1"));
            performanceLog = FromString<int>(GetArg("-pl", "0"));
            debugPrint = FromString<int>(GetArg("-dp", "0"));
            debugPrintFirst = FromString<int>(GetArg("-dpf", "5"));
            debugPrintLast = FromString<int>(GetArg("-dpl", "2"));
            debugPrintPrecision = FromString<int>(GetArg("-dpp", "4"));
            annotateRegions = FromString<int>(GetArg("-ar", "0"));
            regionThreshold = FromString<float>(GetArg("-rt", "0.3"));
            regionOverlap = FromString<float>(GetArg("-ro", "0.5"));
            statFilter = FromString<double>(GetArg("-sf", "0.0"));
            bf16 = FromString<bool>(GetArg("-bf", "0"));
            saveUnoptimized = FromString<bool>(GetArg("-su", "0"));
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
                if (firstMemoryUsage)
                    ss << FullName(firstName, firstType) << " memory usage: " << MemoryUsageString(firstMemoryUsage, testThreads) << std::endl;
                if (secondMemoryUsage)
                    ss << FullName(secondName, secondType) << " memory usage: " << MemoryUsageString(secondMemoryUsage, testThreads) << std::endl;
                PrintPerformance(ss, statFilter);
                if(!statistics.empty())
                    ss << statistics;
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
                if (firstName == "Synet" || secondName == "Synet")
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

        static inline String FullName(const String& name, const String& type)
        {
            return name + (type.empty() ? "" : ("(" + type + ")"));
        }
    };
}


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

#include "TestCommon.h"
#include "TestUtils.h"
#include "TestOptions.h"
#include "TestPerformance.h"
#include "TestSynet.h"

namespace Test
{
    inline bool InitNetwork(const String & model, const String & weight, const Options & options, const TestParam & param, Network & network)
    {
        if (!FileExists(model))
        {
            std::cout << "File '" << model << "' is not exist!" << std::endl;
            return false;
        }

        if (!FileExists(weight))
        {
            std::cout << "File '" << weight << "' is not exist!" << std::endl;
            return false;
        }

        if (!network.Init(model, weight, options.threadNumber, param))
        {
            std::cout << "Can't load " << network.Name() << " from '" << model << "' and '" << weight << "' !" << std::endl;
            return false;
        }

        if (!(network.channels == 1 || network.channels == 3))
        {
            std::cout << "Wrong " << network.Name() << " classifier channels count '" << network.channels << " !" << std::endl;
            return false;
        }

        return true;
    }

    struct TestData
    {
        Strings path;
        Vector input;
        Vector other;
        Vector synet;
    };
    typedef std::shared_ptr<TestData> TestDataPtr;
    typedef std::vector<TestDataPtr> TestDataPtrs;

    inline bool CreateTestList(const Options & options, const TestParam & param, const Network & network, TestDataPtrs & tests)
    {
        if (!DirectoryExists(options.imageDirectory))
        {
            std::cout << "Test image directory '" << options.imageDirectory << "' is not exists!" << std::endl;
            return false;
        }
        StringList images = GetFileList(options.imageDirectory, options.imageFilter, true, false);
        images.sort();
        size_t count = images.size() / network.num;
        if (count == 0)
        {
            std::cout << "There is no one image in '" << options.imageDirectory << "' for '" << options.imageFilter << "' filter!" << std::endl;
            return false;
        }
        StringList::const_iterator name = images.begin();

        tests.clear();
        tests.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            TestDataPtr test(new TestData());
            test->path.resize(network.num);
            test->input.resize(network.GetSize());
            float * input = test->input.data();
            for (size_t j = 0; j < network.num; ++j)
            {
                test->path[j] = MakePath(options.imageDirectory, *name++);
                View original;
                if (!original.Load(test->path[j]))
                {
                    std::cout << "Can't read '" << test->path[j] << "' image!" << std::endl;
                    return false;
                }

                View converted(original.Size(), network.channels == 1 ? View::Gray8 : View::Bgr24);
                Simd::Convert(original, converted);
                View resized(Size(network.width, network.height), converted.format);
                Simd::ResizeBilinear(converted, resized);

                Views channels(network.channels);
                if (network.channels > 1)
                {
                    for (size_t i = 0; i <  network.channels; ++i)
                        channels[i].Recreate(resized.Size(), View::Gray8);
                    Simd::DeinterleaveBgr(resized, channels[0], channels[1], channels[2]);
                }
                else
                    channels[0] = resized;

                for (size_t c = 0; c < channels.size(); ++c)
                {
                    for (size_t y = 0; y < channels[c].height; ++y)
                    {
                        const uint8_t * row = channels[c].Row<uint8_t>(y);
                        ::SimdUint8ToFloat32(row, channels[c].width, &param.lower(), &param.upper(), input);
                        input += channels[c].width;
                    }
                }
            }
            tests.push_back(test);
        }

        return true;
    }

    inline bool AnnotateRegions(const Options & options, const Network & network, const String & inputPath)
    {
        View image;
        if (!image.Load(inputPath))
        {
            std::cout << "Can't read '" << inputPath << "' image!" << std::endl;
            return false;
        }
        Regions regions = network.GetRegions(image.Size(), 0.5f, 0.5f);
        uint32_t white = 0xFFFFFFFF;
        for (size_t i = 0; i < regions.size(); ++i)
        {
            const Region & region = regions[i];
            ptrdiff_t l = ptrdiff_t(region.x - region.w / 2);
            ptrdiff_t t = ptrdiff_t(region.y - region.h / 2);
            ptrdiff_t r = ptrdiff_t(region.x + region.w / 2);
            ptrdiff_t b = ptrdiff_t(region.y + region.h / 2);
            Simd::DrawRectangle(image, l, t, r, b, white);
        }
        String outputPath = MakePath(options.outputDirectory, network.Name() + "_" + GetNameByPath(inputPath));
        if (!image.Save(outputPath))
        {
            std::cout << "Can't write '" << outputPath << "' image!" << std::endl;
            return false;
        }
        return true;
    }

#ifdef SYNET_DEBUG_PRINT_ENABLE
    inline bool DebugPrint(Network & network, const Options & options, size_t i)
    {
        String path = MakePath(options.outputDirectory, network.Name() + "_t" + Synet::ValueToString(options.threadNumber) + "_i" + Synet::ValueToString(i) + ".log");
        std::ofstream log(path);
        if (log.is_open())
        {
            network.DebugPrint(log);
            log.close();
            return true;
        }
        else
        {
            std::cout << "Can't open '" << path << "' file!" << std::endl;
            return false;
        }
    }
#endif

    inline bool Compare(float a, float b, float t)
    {
        float d = ::fabs(a - b);
        return d <= t || d / std::max(::fabs(a), ::fabs(b)) <= t;
    }

    template<class OtherNetwork> bool CompareOtherAndSynet(const Options & options)
    {
        OtherNetwork otherNetwork;
        SynetNetwork synetNetwork;
        TestParamHolder testParam;

        std::cout << "Start " << otherNetwork.Name() << " and " << synetNetwork.Name() << " comparison tests :" << std::endl;

        if (!testParam.Load(options.testParam))
        {
            std::cout << "Can't load file '" << options.testParam << "' !" << std::endl;
            return false;
        }

#ifdef SYNET_OTHER_RUN        
        if (!InitNetwork(options.otherModel, options.otherWeight, options, testParam(), otherNetwork))
            return false;
#endif

#ifdef SYNET_SYNET_RUN
        if (!InitNetwork(options.synetModel, options.synetWeight, options, testParam(), synetNetwork))
            return false;
#endif

#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
        if (!(otherNetwork.num == synetNetwork.num && otherNetwork.channels == synetNetwork.channels &&
            otherNetwork.height == synetNetwork.height && otherNetwork.width == synetNetwork.width))
        {
            std::cout << "Networks are incompatible!" << std::endl;
            return false;
        }
#endif

#if defined(SYNET_ANNOTATE_REGIONS) || defined(SYNET_DEBUG_PRINT_ENABLE) || defined(SYNET_CONVERT_IMAGE)
        if (!DirectoryExists(options.outputDirectory) && !CreatePath(options.outputDirectory))
        {
            std::cout << "Can't create output directory '" << options.outputDirectory << "' !" << std::endl;
            return false;
        }
#endif

        TestDataPtrs tests;
#ifdef SYNET_OTHER_RUN 
        if (!CreateTestList(options, testParam(), otherNetwork, tests))
            return false;
#else
        if (!CreateTestList(options, testParam(), synetNetwork, tests))
            return false;
#endif

        size_t total = tests.size()*options.repeatNumber, current = 0;
        for (size_t i = 0; i < tests.size(); ++i)
        {
            TestData & test = *tests[i];
            for (size_t j = 0; j < options.repeatNumber; ++j, ++current)
            {
                std::cout << "Test progress : " << ToString(100.0*current / total, 1) << "% " << std::flush;
#ifdef SYNET_OTHER_RUN
                test.other = otherNetwork.Predict(tests[i]->input);
#endif
#ifdef SYNET_SYNET_RUN
                test.synet = synetNetwork.Predict(tests[i]->input);
#endif
                std::cout << " \r" << std::flush;
            }

#ifdef SYNET_DEBUG_PRINT_ENABLE
#ifdef SYNET_OTHER_RUN
            if (!DebugPrint(otherNetwork, options, i))
                return false;
#endif
#ifdef SYNET_SYNET_RUN                
            if (!DebugPrint(synetNetwork, options, i))
                return false;
#endif
#endif

#ifdef SYNET_ANNOTATE_REGIONS
#ifdef SYNET_OTHER_RUN
            if (!AnnotateRegions(options, otherNetwork, test.path[0]))
                return false;
#endif
#ifdef SYNET_SYNET_RUN
            if (!AnnotateRegions(options, synetNetwork, test.path[0]))
                return false;
#endif
#endif

#if defined(SYNET_OTHER_RUN) && defined(SYNET_SYNET_RUN)
            if (test.other.size() != test.synet.size())
            {
                std::cout << "Test " << i << " '" << test.path[0];
                for (size_t k = 1; k < test.path.size(); ++k)
                    std::cout << ", " << test.path[k];
                std::cout << "' is failed!" << std::endl;
                std::cout << "Size : " << test.other.size() << " != " << test.synet.size() << std::endl;
                return false;
            }

            for (size_t j = 0; j < test.synet.size(); ++j)
            {
                if (!Compare(test.other[j], test.synet[j], options.threshold))
                {
                    std::cout << "Test " << i << " '" << test.path[0];
                    for (size_t k = 1; k < test.path.size(); ++k)
                        std::cout << ", " << test.path[k];
                    std::cout << "' is failed!" << std::endl;
                    std::cout << j << " : " << test.other[j] << " != " << test.synet[j] << std::endl;
                    return false;
                }
            }
#endif        
        }

        std::cout << "Tests are finished successfully!" << std::endl << std::endl;

#ifdef SYNET_SYNET_RUN
        options.synetMemoryUsage = synetNetwork.MemoryUsage();
#endif

        return true;
    }
}



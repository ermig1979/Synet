/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2020 Yermalayeu Ihar.
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

#include "TestUtils.h"
#include "TestArgs.h"
#include "TestHtml.h"
#include "TestTable.h"
#include "TestSynet.h"

namespace Test
{
    class PerformanceHistory
    {
    public:
        struct Options : public ArgsParser
        {
            Strings inputDirectories;
            String outputDirectory;

            Options(int argc, char* argv[])
                : ArgsParser(argc, argv)
            {
                //inputDirectories = GetArgs("-id", Strings());
                outputDirectory = GetArg("-od", "output");
            }
        };

        PerformanceHistory(const Options & options)
            : _options(options)
        {
        }

        bool Run()
        {
            return true;
        }

    private:
        Options _options;

        template<class T> struct Data
        {
            T time, flops, memory;
            Data() : time(0), flops(0), memory(0) {}
        };

        struct Test
        {
            String name, desc, link;
            int batch, count, skip;
            Data<double> first, second;
            Test(const String & n = "", int b = 0) : name(n), batch(b), count(0), skip(0) {}
        };
        typedef std::vector<Test> Tests;

        struct Set
        {
            String name;
            Tests tests;
        };
        typedef std::vector<Set> Sets;

        bool LoadSync(const String& name, Set & set)
        {
            const String separator = " ";
            set.name = name;
            std::ifstream ifs(name);
            if (ifs.is_open())
            {
                while (!ifs.eof())
                {
                    String line;
                    std::getline(ifs, line);
                    Strings values = Synet::Separate(line, separator);
                    if (values.size() > 10)
                    {
                        Test test;
                        Synet::StringToValue(values[0], test.name);
                        Synet::StringToValue(values[1], test.batch);
                        Synet::StringToValue(values[2], test.first.time);
                        Synet::StringToValue(values[3], test.second.time);
                        Synet::StringToValue(values[4], test.first.flops);
                        Synet::StringToValue(values[5], test.second.flops);
                        Synet::StringToValue(values[6], test.first.memory);
                        Synet::StringToValue(values[7], test.second.memory);
                        Synet::StringToValue(values[8], test.desc);
                        Synet::StringToValue(values[9], test.link);
                        Synet::StringToValue(values[10], test.skip);
                        set.tests.push_back(test);
                    }
                }
                ifs.close();
                return true;
            }
            return false;
        }
    };
}

int main(int argc, char* argv[])
{
    Test::PerformanceHistory::Options options(argc, argv);
    Test::PerformanceHistory history(options);
    return history.Run() ? 0 : 1;
}
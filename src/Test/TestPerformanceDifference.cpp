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

#include "TestUtils.h"
#include "TestSynet.h"

#include "Cpl/Args.h"
#include "Cpl/Table.h"

namespace Test
{
    class PerformanceDifference
    {
    public:
        struct Options : public Cpl::ArgsParser
        {
            Strings inputFirsts;
            Strings inputSeconds;
            String outputDirectory;
            String reportName;
            bool autoCorrection;
            double significantDifference;

            Options(int argc, char* argv[])
                : ArgsParser(argc, argv, true)
            {
                inputFirsts = GetArgs("-if", Strings());
                inputSeconds = GetArgs("-is", Strings());
                outputDirectory = GetArg("-od", "diff");
                reportName = GetArg("-rn", "_diff");
                autoCorrection = FromString<bool>(GetArg("-ac", "1"));
                significantDifference = FromString<double>(GetArg("-sd", "0.05"));
            }
        };

        PerformanceDifference(const Options & options)
            : _options(options)
        {
        }

        bool Estimate()
        {
            if (!LoadInput())
                return false;
            if (!SetDiffMap())
                return false;
            if (!AutoCorrection())
                return false;
            if (!SetAverage())
                return false;
            if (!FillSummary())
                return false;
            if (!PrintReport())
                return false;
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
            bool operator < (const Test& other)
            {
                return name == other.name ? batch < other.batch : name < other.name;
            }        
        };
        typedef std::vector<Test> Tests;

        struct Set
        {
            String name;
            Tests tests;
        };
        typedef std::vector<Set> Sets;
        Sets _first, _second;

        typedef std::vector<double> Doubles;
        typedef std::pair<String, int> Id;
        struct Diff
        {
            Tests firsts, seconds;
            Test first, second;
        };
        typedef std::map<Id, Diff> DiffMap;
        DiffMap _full, _summ;

        bool LoadInput()
        {
            for (size_t i = 0; i < _options.inputFirsts.size(); ++i)
            {
                Set first;
                if (!Load(_options.inputFirsts[i], first))
                    return false;
                _first.push_back(first);
            }
            for (size_t i = 0; i < _options.inputSeconds.size(); ++i)
            {
                Set second;
                if (!Load(_options.inputSeconds[i], second))
                    return false;
                _second.push_back(second);
            }
            return true;
        }

        bool Load(const String& name, Set & set)
        {
            const String separator = " ";
            set.name = name;
            String path = MakePath(name, "sync.txt");
            std::ifstream ifs(path);
            if (ifs.is_open())
            {
                while (!ifs.eof())
                {
                    String line;
                    std::getline(ifs, line);
                    Strings values = Cpl::Separate(line, separator);
                    if (values.size() > 10)
                    {
                        Test test;
                        Cpl::ToVal(values[0], test.name);
                        Cpl::ToVal(values[1], test.batch);
                        Cpl::ToVal(values[2], test.first.time);
                        Cpl::ToVal(values[3], test.second.time);
                        Cpl::ToVal(values[4], test.first.flops);
                        Cpl::ToVal(values[5], test.second.flops);
                        Cpl::ToVal(values[6], test.first.memory);
                        Cpl::ToVal(values[7], test.second.memory);
                        Cpl::ToVal(values[8], test.desc);
                        Cpl::ToVal(values[9], test.link);
                        Cpl::ToVal(values[10], test.skip);
                        set.tests.push_back(test);
                    }
                }
                ifs.close();
                std::sort(set.tests.begin(), set.tests.end());
                return true;
            }
            else
            {
                std::cout << "Can't open file '" << path << "' !" << std::endl;
                return false;
            }
        }

        bool SetDiffMap()
        {
            _full.clear();
            for (size_t i = 0; i < _first.size(); ++i)
            {
                const Set& set = _first[i];
                for (size_t j = 0; j < set.tests.size(); ++j)
                {
                    const Test& test = set.tests[j];
                    _full[Id(test.name, test.batch)].firsts.push_back(test);
                }
            }
            for (size_t i = 0; i < _second.size(); ++i)
            {
                const Set& set = _second[i];
                for (size_t j = 0; j < set.tests.size(); ++j)
                {
                    const Test& test = set.tests[j];
                    _full[Id(test.name, test.batch)].seconds.push_back(test);
                }
            }
            for (DiffMap::iterator it = _full.begin(); it != _full.end();)
            {
                if (it->second.firsts.size() < _options.inputFirsts.size() ||
                    it->second.seconds.size() < _options.inputSeconds.size())
                    it = _full.erase(it);
                else
                    ++it;
            }
            return !_full.empty();
        }

        bool AutoCorrection()
        {
            if (!_options.autoCorrection)
                return true;

            Doubles first(_options.inputFirsts.size(), 0), second(_options.inputSeconds.size(), 0);
            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
            {
                for(size_t i = 0; i < first.size(); ++i)
                    first[i] += ::log(it->second.firsts[i].second.time);
                for (size_t i = 0; i < second.size(); ++i)
                    second[i] += ::log(it->second.seconds[i].second.time);
            }

            for (size_t i = 0; i < first.size(); ++i)
                first[i] = ::exp(first[i] / _full.size());
            for (size_t i = 0; i < second.size(); ++i)
                second[i] = ::exp(second[i] / _full.size());

            Doubles firstK(first.size(), 1.0), secondK(second.size(), 1.0);
            for (size_t i = 0; i < first.size(); ++i)
            {
                for (size_t j = 0; j < first.size(); ++j)
                    if (first[i] * 0.01 > first[j])
                        firstK[i] = 0.001;
                for (size_t j = 0; j < second.size(); ++j)
                    if (first[i] * 0.01 > second[j])
                        firstK[i] = 0.001;
            }
            for (size_t i = 0; i < second.size(); ++i)
            {
                for (size_t j = 0; j < first.size(); ++j)
                    if (second[i] * 0.01 > first[j])
                        secondK[i] = 0.001;
                for (size_t j = 0; j < second.size(); ++j)
                    if (second[i] * 0.01 > second[j])
                        secondK[i] = 0.001;
            }

            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
            {
                for (size_t i = 0; i < first.size(); ++i)
                    it->second.firsts[i].second.time *= firstK[i];
                for (size_t i = 0; i < second.size(); ++i)
                    it->second.seconds[i].second.time *= secondK[i];
            }

            return true;
        }

        bool SetAverage()
        {
            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
            {
                Diff& comp = it->second;
                comp.first = comp.firsts[0];
                comp.second = comp.seconds[0];

                comp.first.second.time = 0;
                comp.first.second.flops = 0;
                for (size_t i = 0, n = comp.firsts.size(); i < n; ++i)
                {
                    comp.first.second.time += comp.firsts[i].second.time / n;
                    comp.first.second.flops += comp.firsts[i].second.flops / n;
                }

                comp.second.second.time = 0;
                comp.second.second.flops = 0;
                for (size_t i = 0, n = comp.seconds.size(); i < n; ++i)
                {
                    comp.second.second.time += comp.seconds[i].second.time / n;
                    comp.second.second.flops += comp.seconds[i].second.flops / n;
                }
            }
            return true;
        }

        bool FillSummary()
        {
            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
            {
                const Test& first = it->second.first;
                const Test& second = it->second.second;
                Diff & common = _summ[Id(" Common", 0)];
                Diff & batch = _summ[Id("Batch-" + std::to_string(first.batch), first.batch)];
                batch.firsts.resize(1);
                common.firsts.resize(1);
                batch.seconds.resize(1);
                common.seconds.resize(1);
                batch.first.count++;
                common.first.count++;

                batch.first.second.time += ::log(first.second.time);
                common.first.second.time += ::log(first.second.time);

                batch.first.second.flops += ::log(first.second.flops);
                common.first.second.flops += ::log(first.second.flops);

                batch.second.second.time += ::log(second.second.time);
                common.second.second.time += ::log(second.second.time);

                batch.second.second.flops += ::log(second.second.flops);
                common.second.second.flops += ::log(second.second.flops);
            }

            for (DiffMap::iterator it = _summ.begin(); it != _summ.end(); ++it)
            {
                Diff& comp = it->second;
                comp.first.name = it->first.first;
                comp.first.batch = it->first.second;
                comp.first.second.time = comp.first.count > 0 ? ::exp(comp.first.second.time / comp.first.count) : 0;
                comp.first.second.flops = comp.first.count > 0 ? ::exp(comp.first.second.flops / comp.first.count) : 0;
                comp.second.second.time = comp.first.count > 0 ? ::exp(comp.second.second.time / comp.first.count) : 0;
                comp.second.second.flops = comp.first.count > 0 ? ::exp(comp.second.second.flops / comp.first.count) : 0;
            }
            return true;
        }

        bool PrintReport()
        {
            if (!CreateOutputDirectory(_options.outputDirectory))
                return false;
            if (!Save(MakePath(_options.outputDirectory, _options.reportName + ".html"), false))
                return false;
            if (!Save(MakePath(_options.outputDirectory, _options.reportName + ".txt"), true))
                return false;
            return true;
        }

        bool Save(const String& name, bool text)
        {
            if (name.empty())
                return true;
            if (!CreateOutputDirectory(name))
                return false;
            std::ofstream ofs(name);
            if (ofs.is_open())
            {
                Cpl::Table table(TableSize().x, TableSize().y);
                SetHeader(table);
                size_t row = 0;
                for (DiffMap::iterator it = _summ.begin(); it != _summ.end(); ++it, ++row)
                    SetCells(table, it->second, row, true);
                for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it, ++row)
                    SetCells(table, it->second, row, false);
                if (text)
                {
                    ofs << "~~~~~~~~~~~~~~~~~~~~~ Synet Performance Difference ~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
                    ofs << "Difference generation time: " + Cpl::CurrentDateTimeString() << std::endl;
                    ofs << table.GenerateText();
                }
                else
                {
                    Cpl::Html html(ofs);

                    html.WriteBegin("html", Cpl::Html::Attr(), true, true);
                    html.WriteValue("title", Cpl::Html::Attr(), "Synet Performance Difference", true);
                    html.WriteBegin("body", Cpl::Html::Attr(), true, true);

                    html.WriteValue("h1", Cpl::Html::Attr("id", "home"), "Synet Performance Difference", true);

                    html.WriteValue("h4", Cpl::Html::Attr(), String("Difference generation time: ") + Cpl::CurrentDateTimeString(), true);

                    ofs << table.GenerateHtml(html.Indent());

                    html.WriteEnd("body", true, true);
                    html.WriteEnd("html", true, true);
                }
                ofs.close();
                return true;
            }
            return false;
        }

        Size TableSize()
        {
            size_t col = 8;
            size_t row = _summ.size() + _full.size();
            return Size(col, row);
        }

        String UnitedName(const Strings& names)
        {
            if (names.size() == 1)
                return LastDirectoryByPath(names[0]);
            else
            {
                std::stringstream ss;
                ss << "(" << LastDirectoryByPath(names[0]);
                for (size_t i = 1; i < names.size(); ++i)
                    ss << ", " << LastDirectoryByPath(names[i]);
                ss << ")";
                return ss.str();
            }
        }

        void SetHeader(Cpl::Table& table)
        {
            String first = UnitedName(_options.inputFirsts);
            String second = UnitedName(_options.inputSeconds);
            size_t col = 0;
            table.SetHeader(col++, "Test", true, Cpl::Table::Center);
            table.SetHeader(col++, "Batch", true, Cpl::Table::Center);
            table.SetHeader(col++, first + ", ms", true, Cpl::Table::Center);
            table.SetHeader(col++, second + ", ms", true, Cpl::Table::Center);
            table.SetHeader(col++, "Relation", true, Cpl::Table::Center);
            table.SetHeader(col++, "Performance, GFLOPS", true, Cpl::Table::Center);
            table.SetHeader(col++, "Size, MB", true, Cpl::Table::Center);
            table.SetHeader(col++, "Description", true, Cpl::Table::Center);
        }

        void SetCells(Cpl::Table& table, const Diff& comp, size_t row, bool summary)
        {
            const Test& first = comp.first;
            const Test& second = comp.second;
            size_t col = 0;
            table.SetCell(col++, row, first.name, Cpl::Table::Black);
            table.SetCell(col++, row, first.batch ? ToString(first.batch) : String("-"));
            table.SetCell(col++, row, ToString(first.second.time, 3), Cpl::Table::Black);
            table.SetCell(col++, row, ToString(second.second.time, 3), Cpl::Table::Black);
            double relation = first.second.time / second.second.time;
            double threshold = 1.0 - _options.significantDifference;
            table.SetCell(col++, row, ToString(relation, 2), relation < threshold ? Cpl::Table::Red : Cpl::Table::Black);
            table.SetCell(col++, row, ToString(second.second.flops, 1));
            table.SetCell(col++, row, summary ? String("-") : ToString(second.second.memory, 1));
            table.SetCell(col++, row, summary ? String("-") : first.desc);
            table.SetRowProp(row, summary && row == (summary ? _summ.size() : _full.size()) - 1, summary);
        }
    };
}

int main(int argc, char* argv[])
{
    Test::PerformanceDifference::Options options(argc, argv);
    Test::PerformanceDifference difference(options);
    return difference.Estimate() ? 0 : 1;
}
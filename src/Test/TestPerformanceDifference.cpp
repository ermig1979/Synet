/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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
#include "Cpl/Log.h"

#ifdef _stat
#undef _stat
#endif

namespace Test
{
    class PerformanceDifference
    {
    public:
        struct Options : public Cpl::ArgsParser
        {
            String mode;
            Strings inputFirsts;
            Strings inputSeconds;
            String outputDirectory;
            String reportName;
            String sortType;
            double significantDifference;

            Options(int argc, char* argv[])
                : ArgsParser(argc, argv, true)
            {
                mode = GetArg("-m", "pair", true, Strings({ "pair", "bf16" }));
                inputFirsts = GetArgs("-if", Strings());
                inputSeconds = GetArgs("-is", Strings(), false);
                outputDirectory = GetArg("-od", "diff");
                reportName = GetArg("-rn", "_diff");
                sortType = GetArg("-st", "name_format_batch", true, Strings({ "name_format_batch", "time_relation" }));
                significantDifference = FromString<double>(GetArg("-sd", "0.05"));
            }

            bool PairMode() const
            {
                return mode == "pair";
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
            if (!SetAverage())
                return false;
            if (!FillSummary())
                return false;
            if (!SortBy())
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

            inline void AddLog(const Data & data)
            {
                time += ::log(data.time);
                flops += ::log(data.flops);
                memory += ::log(data.memory);
            }

            inline void ExpAvg(int count)
            {
                time = count > 0 ? ::exp(time / count) : 0;
                flops = count > 0 ? ::exp(flops / count) : 0;
                memory = count > 0 ? ::exp(memory / count) : 0;
            }
        };

        struct Test
        {
            String name, desc, link;
            int format, batch, count, skip;
            Data<double> first, second;
            Test(const String & n = "", int f = 0, int b = 0) : name(n), format(f), batch(b), count(0), skip(0) {}
            bool operator < (const Test& other)
            {
                return name == other.name ? (format == other.format ? batch < other.batch : format < other.format) : name < other.name;
            } 

            String FormatStr() const
            {
                return format < 0 ? "-" : format == 0 ? "FP32" : "BF16";
            }

            String BatchStr() const
            {
                return batch < 0 ? "-" : ToString(batch);
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
        typedef std::tuple<String, int, int> Id;
        struct Diff
        {
            Tests firsts, seconds;
            Test first, second;
        };
        typedef std::map<Id, Diff> DiffMap;
        typedef std::vector<Diff> Diffs;
        DiffMap _full, _summ;
        Diffs _sorted;

        bool LoadInput()
        {
            for (size_t i = 0; i < _options.inputFirsts.size(); ++i)
            {
                Set first;
                if (!Load(_options.inputFirsts[i], first))
                    return false;
                _first.push_back(first);
            }
            if (_options.PairMode())
            {
                for (size_t i = 0; i < _options.inputSeconds.size(); ++i)
                {
                    Set second;
                    if (!Load(_options.inputSeconds[i], second))
                        return false;
                    _second.push_back(second);
                }
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
                    if (values.size() > 11)
                    {
                        Test test;
                        int col = 0;
                        Cpl::ToVal(values[col++], test.name);
                        Cpl::ToVal(values[col++], test.format);
                        Cpl::ToVal(values[col++], test.batch);
                        Cpl::ToVal(values[col++], test.first.time);
                        Cpl::ToVal(values[col++], test.second.time);
                        Cpl::ToVal(values[col++], test.first.flops);
                        Cpl::ToVal(values[col++], test.second.flops);
                        Cpl::ToVal(values[col++], test.first.memory);
                        Cpl::ToVal(values[col++], test.second.memory);
                        Cpl::ToVal(values[col++], test.desc);
                        Cpl::ToVal(values[col++], test.link);
                        Cpl::ToVal(values[col++], test.skip);
                        set.tests.push_back(test);
                    }
                }
                ifs.close();
                //std::sort(set.tests.begin(), set.tests.end());
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
            if (_options.PairMode())
            {
                for (size_t i = 0; i < _first.size(); ++i)
                {
                    const Set& set = _first[i];
                    for (size_t j = 0; j < set.tests.size(); ++j)
                    {
                        const Test& test = set.tests[j];
                        _full[Id(test.name, test.format, test.batch)].firsts.push_back(test);
                    }
                }
                for (size_t i = 0; i < _second.size(); ++i)
                {
                    const Set& set = _second[i];
                    for (size_t j = 0; j < set.tests.size(); ++j)
                    {
                        const Test& test = set.tests[j];
                        _full[Id(test.name, test.format, test.batch)].seconds.push_back(test);
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
            }
            else
            {
                for (size_t i = 0; i < _first.size(); ++i)
                {
                    const Set& set = _first[i];
                    for (size_t j = 0; j < set.tests.size(); ++j)
                    {
                        const Test& test = set.tests[j];
                        if(test.format == 0)
                            _full[Id(test.name, -1, test.batch)].firsts.push_back(test);
                        else
                            _full[Id(test.name, -1, test.batch)].seconds.push_back(test);
                    }
                }            
                for (DiffMap::iterator it = _full.begin(); it != _full.end();)
                {
                    if (it->second.firsts.size() == 0 || it->second.seconds.size() == 0)
                        it = _full.erase(it);
                    else
                        ++it;
                }
            }
            return !_full.empty();
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

        static inline void UpdateSummary(const Test& first, const Test& second, Diff& summary)
        {
            summary.firsts.resize(1);
            summary.seconds.resize(1);
            summary.first.count++;
            summary.first.second.AddLog(first.second);
            summary.second.second.AddLog(second.second);
        }

        bool FillSummary()
        {
            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
            {
                const Test& first = it->second.first;
                const Test& second = it->second.second;
                if (_options.PairMode())
                {
                    Diff& common = _summ[Id("  Common", -1, -1)];
                    Diff& format = _summ[Id(first.format ? "BF16-Common" : "FP32-Common", first.format, -1)];
                    Diff& batch = _summ[Id((first.format ? "BF16-Batch-" : "FP32-Batch-") + std::to_string(first.batch), first.format, first.batch)];
                    UpdateSummary(first, second, batch);
                    UpdateSummary(first, second, format);
                    UpdateSummary(first, second, common);
                }
                else
                {
                    Diff& common = _summ[Id("  Common", -1, -1)];
                    Diff& batch = _summ[Id((first.format ? "Batch-" : "Batch-") + std::to_string(first.batch), -1, first.batch)];
                    UpdateSummary(first, second, batch);
                    UpdateSummary(first, second, common);
                }
            }
            for (DiffMap::iterator it = _summ.begin(); it != _summ.end(); ++it)
            {
                Diff& comp = it->second;
                comp.first.name = std::get<0>(it->first);
                comp.first.format = std::get<1>(it->first);
                comp.first.batch = std::get<2>(it->first);
                comp.first.second.ExpAvg(comp.first.count);
                comp.second.second.ExpAvg(comp.first.count);
            }
            return true;
        }

        bool SortBy()
        {
            _sorted.reserve(_full.size());
            for (DiffMap::iterator it = _full.begin(); it != _full.end(); ++it)
                _sorted.push_back(it->second);
            if(_options.sortType == "time_relation")
                std::sort(_sorted.begin(), _sorted.end(), [](const Diff& a, const Diff& b) { return a.first.second.time / a.second.second.time < b.first.second.time / b.second.second.time; });
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
                for (size_t i = 0; i < _sorted.size(); ++i, ++row)
                    SetCells(table, _sorted[i], row, false);
                if (text)
                {
                    ofs << "~~~~~~~~~~~~~~~~~~~~~ " << CapionStr() << " ~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
                    ofs << GenTimeStr() << std::endl;
                    ofs << SourceStr() << std::endl;
                    ofs << TestNumStr() << std::endl;
                    ofs << table.GenerateText();
                }
                else
                {
                    Cpl::Html html(ofs);

                    html.WriteBegin("html", Cpl::Html::Attr(), true, true);
                    html.WriteValue("title", Cpl::Html::Attr(), CapionStr(), true);
                    html.WriteBegin("body", Cpl::Html::Attr(), true, true);

                    html.WriteValue("h1", Cpl::Html::Attr("id", "home"), CapionStr(), true);

                    html.WriteValue("h4", Cpl::Html::Attr(), GenTimeStr(), true);
                    html.WriteValue("h4", Cpl::Html::Attr(), SourceStr(), true);
                    html.WriteValue("h4", Cpl::Html::Attr(), TestNumStr(), true);

                    ofs << table.GenerateHtml(html.Indent());

                    html.WriteEnd("body", true, true);
                    html.WriteEnd("html", true, true);
                }
                ofs.close();
                return true;
            }
            return false;
        }

        String CapionStr() const
        {
            return String("Synet Performance Difference");
        }

        String GenTimeStr() const
        {
            return String("Report generation time: ") + Cpl::CurrentDateTimeString();
        }

        String SourceStr() const
        {
            std::stringstream ss;
            ss << "Generated on base: ";
            for (size_t i = 0; i < _options.inputFirsts.size(); ++i)
                ss << _options.inputFirsts[i] << " (" << _first[i].tests.size() << " tests) ";
            if (_options.inputSeconds.size())
            {
                ss << "and ";
                for (size_t i = 0; i < _options.inputSeconds.size(); ++i)
                    ss << _options.inputSeconds[i] << " (" << _second[i].tests.size() << " tests) ";
            }
            return ss.str();
        }

        String TestNumStr() const
        {
            return String("Relevant tests to compare: ") + ToString(_full.size());
        }

        Size TableSize()
        {
            size_t col = _options.PairMode() ? 9 : 11;
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
            size_t col = 0;
            table.SetHeader(col++, "Test", true, Cpl::Table::Center);
            if (_options.PairMode())
            {
                String first = UnitedName(_options.inputFirsts);
                String second = UnitedName(_options.inputSeconds);
                table.SetHeader(col++, "Format", true, Cpl::Table::Center);
                table.SetHeader(col++, "Batch", true, Cpl::Table::Center);
                table.SetHeader(col++, first + ", ms", true, Cpl::Table::Center);
                table.SetHeader(col++, second + ", ms", true, Cpl::Table::Center);
                table.SetHeader(col++, "Relation", true, Cpl::Table::Center);
                table.SetHeader(col++, "Performance, GFLOPS", true, Cpl::Table::Center);
                table.SetHeader(col++, "Size, MB", true, Cpl::Table::Center);
            }
            else
            {
                table.SetHeader(col++, "Batch", true, Cpl::Table::Center);
                table.SetHeader(col++, "FP32 time, ms", true, Cpl::Table::Center);
                table.SetHeader(col++, "BF16 time, ms", true, Cpl::Table::Center);
                table.SetHeader(col++, "Time relation", true, Cpl::Table::Center);
                table.SetHeader(col++, "FP32 perf, GFLOPS", true, Cpl::Table::Center);
                table.SetHeader(col++, "BF16 perf, GFLOPS", true, Cpl::Table::Center);
                table.SetHeader(col++, "FP32 size, MB", true, Cpl::Table::Center);
                table.SetHeader(col++, "BF16 size, MB", true, Cpl::Table::Center);
                table.SetHeader(col++, "Memory relation", true, Cpl::Table::Center);
            }
            table.SetHeader(col++, "Description", true, Cpl::Table::Center);
        }

        void SetCells(Cpl::Table& table, const Diff& comp, size_t row, bool summary)
        {
            const Test& first = comp.first;
            const Test& second = comp.second;
            size_t col = 0;
            table.SetCell(col++, row, first.name, Cpl::Table::Black);
            if (_options.PairMode())
            {
                table.SetCell(col++, row, first.FormatStr(), Cpl::Table::Black);
                table.SetCell(col++, row, first.BatchStr(), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(first.second.time, 3), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(second.second.time, 3), Cpl::Table::Black);
                double relation = first.second.time / second.second.time;
                double threshold = 1.0 - _options.significantDifference;
                table.SetCell(col++, row, ToString(relation, 2), relation < threshold ? Cpl::Table::Red : Cpl::Table::Black);
                table.SetCell(col++, row, ToString(second.second.flops, 1));
                table.SetCell(col++, row, summary ? String("-") : ToString(second.second.memory, 1));
            }
            else
            {
                table.SetCell(col++, row, first.BatchStr(), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(first.second.time, 3), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(second.second.time, 3), Cpl::Table::Black);
                double timeR = first.second.time / second.second.time;
                double timeT = 1.0 - _options.significantDifference;
                table.SetCell(col++, row, ToString(timeR, 2), timeR < timeT ? Cpl::Table::Red : Cpl::Table::Black);
                table.SetCell(col++, row, ToString(first.second.flops, 1), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(second.second.flops, 1), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(first.second.memory, 1), Cpl::Table::Black);
                table.SetCell(col++, row, ToString(second.second.memory, 1), Cpl::Table::Black);
                double memoryR = first.second.memory / second.second.memory;
                double memoryT = 1.0 - _options.significantDifference;
                table.SetCell(col++, row, ToString(memoryR, 2), memoryR < memoryT ? Cpl::Table::Red : Cpl::Table::Black);
            }
            table.SetCell(col++, row, summary ? String("-") : first.desc);
            table.SetRowProp(row, summary && row == (summary ? _summ.size() : _full.size()) - 1, summary);
        }
    };
}

int main(int argc, char* argv[])
{
    Test::PerformanceDifference::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    Test::PerformanceDifference difference(options);

    return difference.Estimate() ? 0 : 1;
}
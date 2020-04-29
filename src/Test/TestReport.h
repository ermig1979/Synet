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

#include "TestTable.h"
#include "TestOptions.h"

namespace Test
{
	class Report
	{
	public:
		Report(const Options& options)
			: _options(options)
			, _separator(" ")
		{
			LoadSync(_options.syncName);
			AddCurrent();
			SaveSync(_options.syncName);
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
				FillSummary();
				Table table(TableSize());
				SetHeader(table);
				SetCells(table, _summary, 0, true);
				SetCells(table, _tests, _summary.size(), false);
				if (text)
				{
					ofs << "~~~~~~~~~~~~~~~~~~~~~ Synet Performance Report ~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
					ofs << "Test generation time: " + CurrentDateTimeString() << std::endl;
					ofs << "Number of test threads: " << _options.testThreads << std::endl;
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
					ofs << SystemInfo() << std::endl;
					Simd::PrintInfo(ofs);
#endif
					ofs << table.GenerateText();
				}
				else
				{
					Html html(ofs);

					html.WriteBegin("html", Html::Attr(), true, true);
					html.WriteValue("title", Html::Attr(), "Synet Performance Report", true);
					html.WriteBegin("body", Html::Attr(), true, true);

					html.WriteValue("h1", Html::Attr("id", "home"), "Synet Performance Report", true);

					html.WriteValue("h4", Html::Attr(), String("Test generation time: ") + CurrentDateTimeString(), true);
					html.WriteValue("h4", Html::Attr(), String("Number of test threads: ") + ToString(_options.testThreads), true);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
					html.WriteValue("h4", Html::Attr(), SystemInfo(), true);
					html.WriteBegin("h4", Html::Attr(), true, true);
					Simd::PrintInfo(ofs);
					html.WriteEnd("h4", true, true);
#endif

					ofs << table.GenerateHtml(html.Indent());

					html.WriteEnd("body", true, true);
					html.WriteEnd("html", true, true);
				}
				ofs.close();
				return true;
			}			
			return false;
		}

	private:
		struct Test
		{
			String name, desc, link;
			int batch, count, skip;
			double other, synet, flops, memory;
			Test(const String & n = "", int b = 0) : name(n), batch(b), count(0), skip(0), other(0), synet(0), flops(0), memory(0) {}
		};
		typedef std::vector<Test> Tests;

		const Options& _options;
		String _separator;
		Tests _tests, _summary;

		bool LoadSync(const String & name)
		{
			std::ifstream ifs(name);
			if (ifs.is_open())
			{
				while (!ifs.eof())
				{
					String line;
					std::getline(ifs, line);
					Strings values = Synet::Separate(line, _separator);
					if (values.size() > 8)
					{
						Test test;
						Synet::StringToValue(values[0], test.name);
						Synet::StringToValue(values[1], test.batch);
						Synet::StringToValue(values[2], test.other);
						Synet::StringToValue(values[3], test.synet);
						Synet::StringToValue(values[4], test.flops);
						Synet::StringToValue(values[5], test.memory);
						Synet::StringToValue(values[6], test.desc);
						Synet::StringToValue(values[7], test.link);
						Synet::StringToValue(values[8], test.skip);
						_tests.push_back(test);
					}
				}
				ifs.close();
				return true;
			}
			return false;
		}

		bool SaveSync(const String& name)
		{
			std::ofstream ofs(name);
			if (ofs.is_open())
			{
				for (size_t i = 0; i < _tests.size(); ++i)
				{
					ofs << _tests[i].name << _separator;
					ofs << _tests[i].batch << _separator;
					ofs << _tests[i].other << _separator;
					ofs << _tests[i].synet << _separator;
					ofs << _tests[i].flops << _separator;
					ofs << _tests[i].memory << _separator;
					ofs << _tests[i].desc << _separator;
					ofs << _tests[i].link << _separator;
					ofs << _tests[i].skip << _separator;
					ofs << std::endl;
				}
				ofs.close();
				return true;
			}
			return false;
		}

		void AddCurrent()
		{
			Test test;
			test.name = TestName(_options.logName);
			test.batch = _options.batchSize;
			test.other = NetworkPredictPm(_options.otherName).Average() * 1000.0 / test.batch;
			test.synet = NetworkPredictPm("Synet").Average() * 1000.0 / test.batch;
			test.flops = NetworkPredictPm("Synet").GFlops();
			test.memory = _options.synetMemoryUsage / 1024.0 / 1024.0;
			test.desc = TestDesc(_options.synetModel);
			test.link = GetNameByPath(_options.logName);
			test.skip = test.synet > test.other * _options.skipThreshold ? 2 : (test.synet * _options.skipThreshold < test.other ? 1 : 0);
			_tests.push_back(test);
		}

		String TestName(const String & path)
		{
			String name = GetNameByPath(path);
			size_t beg = name.find("_") + 1, end = name.rfind(String("_t"));
			return name.substr(beg, end - beg);
		}

		String NetworkPredictName(const String & framework, bool emulation)
		{
			std::stringstream ss;
#ifdef _MSC_VER
			ss << "Test::";
			ss << framework;
			ss << "Network::Predict";
#else
			ss << "virtual const Vectors& Test::";
			ss << framework;
			ss << "Network::Predict(const Vectors&)";
#endif
			if (emulation)
				ss << " { batch emulation } ";
			return ss.str();
		}

		PerformanceMeasurer NetworkPredictPm(String framework)
		{
			framework = WithoutSymbol(framework, ' ');
			PerformanceMeasurer result = PerformanceMeasurerStorage::s_storage.GetCombined(NetworkPredictName(framework, false));
			if (result.Average() == 0)
				result = PerformanceMeasurerStorage::s_storage.GetCombined(NetworkPredictName(framework, true));
			return result;
		}

		String TestDesc(const String & path)
		{
			String desc;
			Synet::NetworkParamHolder model;
			if (model.Load(path))
			{
				std::set<Synet::LayerType> layers;
				for (size_t i = 0; i < model().layers().size(); ++i)
					layers.insert(model().layers()[i].type());
				if (layers.find(Synet::LayerTypeConvolution) != layers.end()) desc.push_back('C');
				if (layers.find(Synet::LayerTypePooling) != layers.end()) desc.push_back('P');
				if (layers.find(Synet::LayerTypeMergedConvolution) != layers.end()) desc.push_back('G');
				if (layers.find(Synet::LayerTypeInnerProduct) != layers.end()) desc.push_back('I');
				if (layers.find(Synet::LayerTypeDetectionOutput) != layers.end()) desc.push_back('D');
			}
			return desc;
		}

		Size TableSize()
		{
			size_t cols = _options.otherName.size() ? 8 : 6;
			size_t rows = _summary.size() + _tests.size();
			return Size(cols, rows);
		}

		void SetHeader(Table& table)
		{
			size_t col = 0;
			table.SetHeader(col++, "Test", true, Table::Center);
			table.SetHeader(col++, "Batch", true, Table::Center);
			if(_options.otherName.size())
				table.SetHeader(col++, _options.otherName + ", ms", true, Table::Center);
			table.SetHeader(col++, "Synet, ms", true, Table::Center);
			if (_options.otherName.size())
				table.SetHeader(col++, _options.otherName + " / Synet", true, Table::Center);
			table.SetHeader(col++, "Synet, GFLOPS", true, Table::Center);
			table.SetHeader(col++, "Synet, MB", true, Table::Center);
			table.SetHeader(col++, "Description", true, Table::Center);
		}

		void SetCells(Table& table, const Tests & tests, size_t row, bool summary)
		{
			for (size_t i = 0; i < tests.size(); ++i, ++row)
			{
				const Test& test = tests[i];
				size_t col = 0;
				table.SetCell(col++, row, test.name, Table::Black, test.link);
				table.SetCell(col++, row, test.batch ? ToString(test.batch) : String("-"));
				if (_options.otherName.size())
					table.SetCell(col++, row, ToString(test.other, 3), test.skip == 1 ? Table::Red : Table::Black);
				table.SetCell(col++, row, ToString(test.synet, 3), test.skip == 2 ? Table::Red : Table::Black);
				if (_options.otherName.size())
				{
					double relation = test.synet != 0 ? test.other / test.synet : 0.0;
					table.SetCell(col++, row, ToString(relation, 2), relation < 1.00 ? Table::Red : Table::Black);
				}
				table.SetCell(col++, row, ToString(test.flops, 1));
 				table.SetCell(col++, row, summary ? String("-") : ToString(test.memory, 1));
				table.SetCell(col++, row, summary ? String("-") : test.desc);
				table.SetRowProp(row, summary && i == tests.size() - 1, summary);
			}
		}

		void FillSummary(Test & summary)
		{
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				const Test& test = _tests[i];
				if ((summary.batch && test.batch != summary.batch) || test.skip)
					continue;
				summary.count++;
				if (_options.otherName.size())
					summary.other += ::log(test.other);
				summary.synet += ::log(test.synet);
				summary.flops += ::log(test.flops);
			}
			if (_options.otherName.size())
				summary.other = summary.count > 0 ? ::exp(summary.other / summary.count) : 0;
			summary.synet = summary.count > 0 ? ::exp(summary.synet / summary.count) : 0;
			summary.flops = summary.count > 0 ? ::exp(summary.flops / summary.count) : 0;
		}

		void FillSummary()
		{
			typedef std::set<int> Set;
			Set batch;
			for (size_t i = 0; i < _tests.size(); ++i)
				batch.insert(_tests[i].batch);
			_summary.clear();
			_summary.push_back(Test("Common", 0));
			for (Set::const_iterator it = batch.begin(); it != batch.end(); ++it)
				_summary.push_back(Test(String("Batch-") + ToString(*it), *it));
			for (size_t i = 0; i < _summary.size(); ++i)
				FillSummary(_summary[i]);
		}
	};

	void GenerateReport(const Options& options)
	{
		if (options.syncName.empty() || !CreateOutputDirectory(options.syncName))
			return;
		Report report(options);
		report.Save(options.textReport, true);
		report.Save(options.htmlReport, false);
	}
}



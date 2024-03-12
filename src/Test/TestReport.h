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

#pragma once

#include "TestOptions.h"

#include "Cpl/Table.h"

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
				Cpl::Table table(TableSize().x, TableSize().y);
				SetHeader(table);
				SetCells(table, _summary, 0, true);
				SetCells(table, _tests, _summary.size(), false);
				if (text)
				{
					ofs << "~~~~~~~~~~~~~~~~~~~~~ Synet Performance Report ~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
					ofs << "Test generation time: " + Cpl::CurrentDateTimeString() << std::endl;
					ofs << "Synet version: " + Synet::Version() << std::endl;
					ofs << "Number of test threads: " << _options.testThreads << std::endl;
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
					Simd::PrintInfo(ofs);
#endif
					ofs << table.GenerateText();
				}
				else
				{
					Cpl::Html html(ofs);

					html.WriteBegin("html", Cpl::Html::Attr(), true, true);
					html.WriteValue("title", Cpl::Html::Attr(), "Synet Performance Report", true);
					html.WriteBegin("body", Cpl::Html::Attr(), true, true);

					html.WriteValue("h1", Cpl::Html::Attr("id", "home"), "Synet Performance Report", true);

					html.WriteValue("h4", Cpl::Html::Attr(), String("Test generation time: ") + Cpl::CurrentDateTimeString(), true);
					html.WriteValue("h4", Cpl::Html::Attr(), String("Synet version: ") + Synet::Version(), true);
					html.WriteValue("h4", Cpl::Html::Attr(), String("Number of test threads: ") + ToString(_options.testThreads), true);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
					html.WriteBegin("h4", Cpl::Html::Attr(), true, true);
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
		template<class T> struct Data
		{
			T time, flops, memory;
			Data() : time(0), flops(0), memory(0) {}
		};

		struct Test
		{
			String name, desc, link;
			int batch, count, skip, bf16;
			Data<double> first, second;
			Test(const String & n = "", int b = 0, int bf = 0) : name(n), batch(b), count(0), skip(0), bf16(bf) {}
		};
		typedef std::vector<Test> Tests;

		const Options& _options;
		String _separator;
		Tests _tests, _summary;
		Data<bool> _other, _synet;

		bool LoadSync(const String & name)
		{
			std::ifstream ifs(name);
			if (ifs.is_open())
			{
				while (!ifs.eof())
				{
					String line;
					std::getline(ifs, line);
					Strings values = Cpl::Separate(line, _separator);
					if (values.size() > 10)
					{
						Test test;
						Cpl::ToVal(values[0], test.name);
						Cpl::ToVal(values[1], test.bf16);
						Cpl::ToVal(values[2], test.batch);
						Cpl::ToVal(values[3], test.first.time);
						Cpl::ToVal(values[4], test.second.time);
						Cpl::ToVal(values[5], test.first.flops);
						Cpl::ToVal(values[6], test.second.flops);
						Cpl::ToVal(values[7], test.first.memory);
						Cpl::ToVal(values[8], test.second.memory);
						Cpl::ToVal(values[9], test.desc);
						Cpl::ToVal(values[10], test.link);
						Cpl::ToVal(values[11], test.skip);
						_tests.push_back(test);
					}
				}
				ifs.close();
				return true;
			}
			return false;
		}

		void Update(const Data<double>& value, Data<bool>& enable)
		{
			enable.time = enable.time || value.time > 0;
			enable.flops = enable.flops || value.flops > 0;
			enable.memory = enable.memory || value.memory > 0;
		}

		bool SaveSync(const String& name)
		{
			std::ofstream ofs(name);
			if (ofs.is_open())
			{
				for (size_t i = 0; i < _tests.size(); ++i)
				{
					ofs << _tests[i].name << _separator;
					ofs << _tests[i].bf16 << _separator;
					ofs << _tests[i].batch << _separator;
					ofs << _tests[i].first.time << _separator;
					ofs << _tests[i].second.time << _separator;
					ofs << _tests[i].first.flops << _separator;
					ofs << _tests[i].second.flops << _separator;
					ofs << _tests[i].first.memory << _separator;
					ofs << _tests[i].second.memory << _separator;
					ofs << _tests[i].desc << _separator;
					ofs << _tests[i].link << _separator;
					ofs << _tests[i].skip << _separator;
					ofs << std::endl;
					Update(_tests[i].first, _other);
					Update(_tests[i].second, _synet);
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
			test.bf16 = _options.bf16;
			test.first.time = NetworkPredictPm(_options.firstName, _options.firstType).Average() / test.batch;
			test.second.time = NetworkPredictPm(_options.secondName, _options.secondType).Average() / test.batch;
			test.first.flops = NetworkPredictPm(_options.firstName, _options.firstType).GFlops();
			test.second.flops = NetworkPredictPm(_options.secondName, _options.secondType).GFlops();
			test.first.memory = _options.firstMemoryUsage / 1024.0 / 1024.0;
			test.second.memory = _options.secondMemoryUsage / 1024.0 / 1024.0;
			test.desc = TestDesc(_options.secondModel);
			test.link = GetNameByPath(_options.logName);
			test.skip = test.second.time > test.first.time * _options.skipThreshold ? 2 : 
				(test.second.time * _options.skipThreshold < test.first.time ? 1 : 0);
			_tests.push_back(test);
		}

		String TestName(const String & path)
		{
			String name = GetNameByPath(path);
			size_t beg = name.find("_") + 1, end = name.rfind(String("_t"));
			return name.substr(beg, end - beg);
		}

		String NetworkPredictName(const String & framework, const String& description)
		{
			std::stringstream ss;
#ifdef _MSC_VER
			ss << "Test::";
			ss << framework;
			ss << "Network::Predict";
#else
			ss << "virtual const Tensors& Test::";
			ss << framework;
			ss << "Network::Predict(const Tensors&)";
#endif
			if (!description.empty())
				ss << "{ " << description << " }";
			return ss.str();
		}

		Cpl::PerformanceMeasurer NetworkPredictPm(String framework, String type)
		{
			framework = WithoutSymbol(framework, ' ');
			Cpl::PerformanceMeasurer result = Cpl::PerformanceStorage::Global().Merged(NetworkPredictName(framework, type));
			if (result.Average() == 0)
				result = Cpl::PerformanceStorage::Global().Merged(NetworkPredictName(framework, "batch emulation"));
			return result;
		}

		String TestDesc(const String & path)
		{
			String desc;
			Synet::NetworkParamHolder model;
			if (model.Load(path))
			{
				for (size_t i = 0; i < model().layers().size(); ++i)
				{
					const Synet::LayerParam& layer = model().layers()[i];
					if (layer.type() == Synet::LayerTypeInput)
					{
						Shape dim = layer.input().shape()[0].dim();
						if (layer.input().shape()[0].format() == Synet::TensorFormatNhwc && dim.size() == 4)
							dim = Synet::Shp(dim[0], dim[3], dim[1], dim[2]);
						for (size_t d = 0; d < dim.size(); ++d)
						{
							if (d)
								desc.push_back('x');
							desc = desc + Cpl::ToStr(dim[d]);
						}
						desc.push_back('-');
						break;
					}
				}
				std::set<Synet::LayerType> layers;
				for (size_t i = 0; i < model().layers().size(); ++i)
					layers.insert(model().layers()[i].type());
				if (layers.find(Synet::LayerTypeConvolution) != layers.end()) desc.push_back('C');
				if (layers.find(Synet::LayerTypePooling) != layers.end()) desc.push_back('P');
				if (layers.find(Synet::LayerTypeMergedConvolution) != layers.end()) desc.push_back('G');
                if (layers.find(Synet::LayerTypeInnerProduct) != layers.end()) desc.push_back('I');
                if (layers.find(Synet::LayerTypeTensorIterator) != layers.end()) desc.push_back('T');
				if (layers.find(Synet::LayerTypeDetectionOutput) != layers.end()) desc.push_back('D');
                if (layers.find(Synet::LayerTypeYolo) != layers.end()) desc.push_back('Y');
                if (layers.find(Synet::LayerTypeRegion) != layers.end()) desc.push_back('R');
            }
			return desc;
		}

		Size TableSize()
		{
			size_t col = 4;
			if (_other.time)
				col++;
			if (_synet.time)
				col++;
			if (_other.time && _synet.time)
				col++;
			if (_other.flops)
				col++;
			if (_synet.flops)
				col++;
			if (_other.memory)
				col++;
			if (_synet.memory)
				col++;
			size_t row = _summary.size() + _tests.size();
			return Size(col, row);
		}

		void SetHeader(Cpl::Table& table)
		{
			String first = Options::FullName(_options.firstName, _options.firstType);
			String second = Options::FullName(_options.secondName, _options.secondType);
			size_t col = 0;
			table.SetHeader(col++, "Test", true, Cpl::Table::Center);
			table.SetHeader(col++, "Format", true, Cpl::Table::Center);
			table.SetHeader(col++, "Batch", true, Cpl::Table::Center);
			if (_other.time)
				table.SetHeader(col++, first + ", ms", true, Cpl::Table::Center);
			if (_synet.time)
				table.SetHeader(col++, second + ", ms", true, Cpl::Table::Center);
			if (_other.time && _synet.time)
				table.SetHeader(col++, first + " / " + second, true, Cpl::Table::Center);
			if (_other.flops)
				table.SetHeader(col++, first + ", GFLOPS", true, Cpl::Table::Center);
			if (_synet.flops)
				table.SetHeader(col++, second + ", GFLOPS", true, Cpl::Table::Center);
			if (_other.memory)
				table.SetHeader(col++, first + ", MB", true, Cpl::Table::Center);
			if (_synet.memory)
				table.SetHeader(col++, second + ", MB", true, Cpl::Table::Center);
			table.SetHeader(col++, "Description", true, Cpl::Table::Center);
		}

		void SetCells(Cpl::Table& table, const Tests & tests, size_t row, bool summary)
		{
			for (size_t i = 0; i < tests.size(); ++i, ++row)
			{
				const Test& test = tests[i];
				size_t col = 0;
				if (test.link.empty())
					table.SetCell(col++, row, test.name, Cpl::Table::Black);
				else
					table.SetCell(col++, row, test.name + (test.bf16 == 0 ? String("_fp32_") : String("_bf16_")) + ToString(test.batch), Cpl::Table::Black, test.link);
				table.SetCell(col++, row, test.bf16 < 0 ? String("-") : test.bf16 == 0 ? String("FP32") : String("BF16"));
				table.SetCell(col++, row, test.batch ? ToString(test.batch) : String("-"));
				if (_other.time)
					table.SetCell(col++, row, ToString(test.first.time, 3), test.skip == 1 ? Cpl::Table::Red : Cpl::Table::Black);
				if (_synet.time)
					table.SetCell(col++, row, ToString(test.second.time, 3), test.skip == 2 ? Cpl::Table::Red : Cpl::Table::Black);
				if (_other.time && _synet.time)
				{
					double relation = test.second.time != 0 ? test.first.time / test.second.time : 0.0;
					table.SetCell(col++, row, ToString(relation, 2), relation < 1.00 ? Cpl::Table::Red : Cpl::Table::Black);
				}
				if (_other.flops)
					table.SetCell(col++, row, ToString(test.first.flops, 1));
				if (_synet.flops)
					table.SetCell(col++, row, ToString(test.second.flops, 1));
				if (_other.memory)
					table.SetCell(col++, row, summary ? String("-") : ToString(test.first.memory, 1));
				if (_synet.memory)
					table.SetCell(col++, row, summary ? String("-") : ToString(test.second.memory, 1));
				table.SetCell(col++, row, summary ? String("-") : test.desc);
				table.SetRowProp(row, summary && i == tests.size() - 1, summary);
			}
		}

		void FillSummary(Test & summary)
		{
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				const Test& test = _tests[i];
				if (test.skip)
					continue;
				if (summary.batch && test.batch != summary.batch)
					continue;
				if (summary.bf16 >= 0 && test.bf16 != summary.bf16)
					continue;
				summary.count++;
				if (_other.time)
					summary.first.time += ::log(test.first.time);
				if (_synet.time)
					summary.second.time += ::log(test.second.time);
				if (_other.flops)
					summary.first.flops += ::log(test.first.flops);
				if (_synet.flops)
					summary.second.flops += ::log(test.second.flops);
			}
			if (_other.time)
				summary.first.time = summary.count > 0 ? ::exp(summary.first.time / summary.count) : 0;
			if (_synet.time)
				summary.second.time = summary.count > 0 ? ::exp(summary.second.time / summary.count) : 0;
			if (_other.flops)
				summary.first.flops = summary.count > 0 ? ::exp(summary.first.flops / summary.count) : 0;
			if (_synet.flops)
				summary.second.flops = summary.count > 0 ? ::exp(summary.second.flops / summary.count) : 0;
		}

		void FillSummary()
		{
			typedef std::set<int> Set;
			Set batch, bf16;
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				batch.insert(_tests[i].batch);
				bf16.insert(_tests[i].bf16);
			}
			_summary.clear();
			if(bf16.size() > 1)
				_summary.push_back(Test("Common", 0, -1));
			for (Set::const_iterator f = bf16.begin(); f != bf16.end(); ++f)
			{
				String format = *f ? String("BF16") : String("FP32");
				_summary.push_back(Test(format + "-Common", 0, *f));
				for (Set::const_iterator b = batch.begin(); b != batch.end(); ++b)
					_summary.push_back(Test(format + "-Batch-" + ToString(*b), *b, *f));
			}
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



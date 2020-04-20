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

		bool Save(const String& name, bool html)
		{
			if (name.empty())
				return true;
			if (!CreateOutputDirectory(name))
				return false;
			std::ofstream ofs(name);
			if (ofs.is_open())
			{
				Table table(GetTableSize());
				FillTable(table);
				if(html)
					ofs << table.GenerateHtml();
				else
					ofs << table.GenerateText();
				ofs.close();
				return true;
			}			
			return false;
		}

	private:
		struct Test
		{
			String name;
			int batch;
			double other, synet, flops, memory;
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
					if (values.size() > 5)
					{
						Test test;
						Synet::StringToValue(values[0], test.name);
						Synet::StringToValue(values[1], test.batch);
						Synet::StringToValue(values[2], test.other);
						Synet::StringToValue(values[3], test.synet);
						Synet::StringToValue(values[4], test.flops);
						Synet::StringToValue(values[5], test.memory);
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
			test.name = GetTestName(_options.logName);
			test.batch = _options.batchSize;
			test.other = GetNetworkPredictPm(_options.otherName).Average() * 1000.0;
			test.synet = GetNetworkPredictPm("Synet").Average() * 1000.0;
			test.flops = GetNetworkPredictPm("Synet").GFlops();
			test.memory = _options.synetMemoryUsage / 1024.0 / 1024.0;
			_tests.push_back(test);
		}

		String GetTestName(const String & path)
		{
			String name = GetNameByPath(path);
			size_t beg = name.find("_") + 1, end = name.rfind(String("_t"));
			return name.substr(beg, end - beg);
		}

		PerformanceMeasurer GetNetworkPredictPm(const String & framework)
		{
			std::stringstream ss;
			ss << "virtual const Vectors& Test::";
			ss << WithoutSymbol(framework, ' ');
			ss << "Network::Predict(const Vectors&)";
			return PerformanceMeasurerStorage::s_storage.GetCombined(ss.str());
		}

		Size GetTableSize()
		{
			size_t cols = 7;
			size_t rows = _tests.size() + _summary.size();
			return Size(cols, rows);
		}

		void FillTable(Table & table)
		{
			size_t col = 0;
			table.SetHeader(col++, "Test", true);
			table.SetHeader(col++, "Batch", true);
			table.SetHeader(col++, _options.otherName + ", ms", true);
			table.SetHeader(col++, "Synet, ms" , true);
			table.SetHeader(col++, _options.otherName + ", Gflops", true);
			table.SetHeader(col++, "Synet, Gflops", true);
			table.SetHeader(col++, String("Synet / ") + _options.otherName, true);

			for (size_t i = 0; i < _tests.size(); ++i)
			{
				size_t col = 0;
				table.SetCell(col++, i, _tests[i].name);
				table.SetCell(col++, i, ToString(_tests[i].batch));
				table.SetCell(col++, i, ToString(_tests[i].other / _tests[i].batch, 3));
				table.SetCell(col++, i, ToString(_tests[i].synet / _tests[i].batch, 3));
				table.SetCell(col++, i, ToString(_tests[i].flops * _tests[i].synet / _tests[i].other, 1));
				table.SetCell(col++, i, ToString(_tests[i].flops, 1));
				table.SetCell(col++, i, ToString(_tests[i].other / _tests[i].synet, 2));
			}
			//summary
		}
	};

	void GenerateReport(const Options& options)
	{
		if (options.syncName.empty() || !CreateOutputDirectory(options.syncName))
			return;
		Report report(options);
		report.Save(options.textReport, false);
		report.Save(options.htmlReport, true);
	}
}



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

#include "TestPrecision.h"

namespace Test
{
	class ClassificationPrecision : public Precision
	{
	public:
		ClassificationPrecision(const Options& options)
			: Precision(options)
		{
		}

	private:

		struct Test
		{
			String name, path;
			Floats current, control;
		};
		typedef std::vector<Test> Tests;
		Tests _tests;

		typedef std::set<String> StringSet;
		StringSet _list;

		bool LoadListFile()
		{
			_list.clear();
			String path = _options.testList;
			std::ifstream ifs(path);
			if (ifs.is_open())
			{
				while (!ifs.eof())
				{
					String name;
					ifs >> name;
					if (!name.empty())
						_list.insert(name);
				}
			}
			else
			{
				CPL_LOG_SS(Error, "Can't open list file '" << path << "' !");
				if (!_options.generateIndex)
					return false;
			}
			return true;
		}

		bool SaveListFile()
		{
			String path = _options.testList;
			std::ofstream ofs(path);
			if (!ofs.is_open())
				SYNET_ERROR("Can't open file '" << path << "' !");
			for (size_t i = 0; i < _tests.size(); ++i)
				ofs << _tests[i].name << std::endl;
			return true;
		}

		String PrintResume(size_t number, double precision, double error, double threshold)
		{
			std::stringstream ss;
			ss << "Number: " << number << ", precision: " << ToString(precision * 100, 2);
			ss << " %, error: " << ToString(error * 100, 2) << " %, threshold: " << ToString(threshold, 3) << std::endl;
			return ss.str();
		}

		bool LoadIndexFile()
		{
			if (_param().index().type() != "ClassificationTextV1")
				return false;
			String path = MakePath(_options.imageDirectory, _param().index().name());
			std::ifstream ifs(path);
			if (!ifs.is_open())
				SYNET_ERROR("Can't open file '" << path << "' !");
			_tests.clear();
			size_t size;
			ifs >> size;
			if (size == 0)
				SYNET_ERROR("Wrong size: " << size << " !");
			while (!ifs.eof())
			{
				Test test;
				ifs >> test.name;
				if (test.name.empty())
					break;
				test.path = MakePath(_options.imageDirectory, test.name);
				if (!FileExists(test.path))
					SYNET_ERROR("Image '" << test.path << "' is not exists!");
				test.control.resize(size);
				for (size_t i = 0; i < size; ++i)
					ifs >> test.control[i];
				if (_list.empty() || _list.find(test.name) != _list.end())
					_tests.push_back(test);
			}
			_options.testNumber = _tests.size();
			if (_options.testNumber == 0)
				SYNET_ERROR("Test list is empty!");
			return true;
		}

		bool SaveIndexFile()
		{
			if (_param().index().type() != "ClassificationTextV1")
				return false;
			String path = MakePath(_options.imageDirectory, _param().index().name());
			std::ofstream ofs(path);
			if (!ofs.is_open())
				SYNET_ERROR("Can't open file '" << path << "' !");
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				const Test& t = _tests[i];
				if(i == 0)
					ofs << t.control.size() << std::endl;
				ofs << t.name << " ";
				for (size_t k = 0; k < t.control.size(); ++k)
					ofs << t.control[k] << " ";
				ofs << std::endl;
			}
			return true;
		}

		virtual bool LoadTestList()
		{
			if (!LoadListFile())
				return false;
			if (_options.generateIndex)
			{
				if (!GenerateIndex(_tests))
					return false;
			}
			else
			{
				if (!LoadIndexFile())
					return false;
			}
			return true;
		}

		virtual bool PerformBatch(size_t thread, size_t current, size_t batch, size_t& progress)
		{
			Thread & t = _threads[thread];
			for (size_t b = 0; b < batch; ++b)
			{
				const Test & test = _tests[current + b];
				if (!SetInput(test.path, t.input[0], b, NULL))
					return false;
			}
			for(int i = 0; i < _options.repeatNumber; ++i, progress += batch)
				t.output = t.network->Predict(t.input);
			size_t size = t.output[0].Size(1);
			const float* data = t.output[0].CpuData();
			for (size_t b = 0; b < batch; ++b, data += size)
			{
				Test& test = _tests[current + b];
				test.current.assign(data, data + size);
				if (_options.generateIndex)
					test.control = test.current;
			}
			return true;
		}

		String PrintResume(size_t number, double min, double max, double precision, double average, double sigma, double divMin, double divMax)
		{
			std::stringstream ss;
			ss << "Number: " << number << " [" << ToString(min, 1) << " .. " << ToString(max, 1);
			ss << "], precision: " << ToString(precision * 100, 2);
			ss << " %; error avg: " << ToString(average, 2) << ", sigma: " << ToString(sigma, 2);
			ss << ", min: " << ToString(divMin, 2) << ", max: " << ToString(divMax, 2);
			return ss.str();
		}

		virtual bool ProcessResult()
		{
			if (_options.generateIndex)
			{
				if (!SaveIndexFile())
					return false;
			}
			SaveListFile();
			if (_tests[0].control.size() == 1)
			{
				float min = FLT_MAX, max = -FLT_MAX, sum = 0, sqsum = 0, divMin, divMax = 0;
				for (size_t i = 0; i < _tests.size(); ++i)
				{
					min = std::min(min, _tests[i].control[0]);
					max = std::max(max, _tests[i].control[0]);
					float diff = _tests[i].current[0] - _tests[i].control[0];
					sum += diff;
					sqsum += diff * diff;
					divMin = std::min(divMin, diff);
					divMax = std::max(divMax, diff);
				}
				double average = sum / _tests.size();
				double sigma = sqrt(sqsum / _tests.size() - average * average);
				double precision = 1.0 - sigma / (max == min ? max : max - min);
				_options.resume = PrintResume(_tests.size(), min, max, precision, average, sigma, divMin, divMax);
			}
			else
			{
				_options.resume = "Can't print resume!";
				return false;
			}
			return true;
		}
	};
}



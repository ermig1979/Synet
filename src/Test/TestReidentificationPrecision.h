/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
	class ReidentificationPrecision : public Precision
	{
	public:
		ReidentificationPrecision(const Options& options)
			: Precision(options)
		{
		}

	private:
		struct Object
		{
			String name, path;
			Tensor desc;
		};

		struct Test
		{
			Object objects[2];
			float distance;
		};
		typedef std::vector<Test> Tests;
		typedef std::shared_ptr<Test> TestPtr;
		typedef std::vector<TestPtr> TestPtrs;
		
		Tests _tests;

		bool SetPath(const String& name, int index, String& path)
		{
			path = MakePath(_options.imageDirectory, name + "_" + ToString(index, 4) + ".jpg");
			if(!FileExists(path))
				SYNET_ERROR("Test image '" << path << "' is not exist!");
			return true;
		}

		virtual bool LoadTestList()
		{
			if (!DirectoryExists(_options.imageDirectory))
				SYNET_ERROR("Image directory '" << _options.imageDirectory << "' is not exist!");
			if (!FileExists(_options.testList))
				SYNET_ERROR("Test list file '" << _options.testList << "' is not exist!");
			std::ifstream ifs(_options.testList);
			if (!ifs.is_open())
				SYNET_ERROR("Can't open test list file '" << _options.testList << "'!");
			String line;
			if (!std::getline(ifs, line))
				SYNET_ERROR("Can't read 1-st file line!");
			size_t size = FromString<int>(line);
			if (size > USHRT_MAX)
				SYNET_ERROR("Wrong test size " << size << " !");
			_tests.resize(size * 2);
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				if (!std::getline(ifs, line))
					SYNET_ERROR("Can't read " << i + 2 << " file line!");
				std::stringstream ss(line);
				int first, second;
				if (i < size)
				{
					ss >> _tests[i].objects[0].name >> first >> second;
					_tests[i].objects[1].name = _tests[i].objects[0].name;
				}
				else
					ss >> _tests[i].objects[0].name >> first >> _tests[i].objects[1].name >> second;
				if (_tests[i].objects[0].name.empty() || _tests[i].objects[1].name.empty() || first == 0 || second == 0)
					SYNET_ERROR("Can't parse " << i + 2 << " file line!");
				if (!SetPath(_tests[i].objects[0].name, first, _tests[i].objects[0].path))
					return false;
				if (!SetPath(_tests[i].objects[1].name, second, _tests[i].objects[1].path))
					return false;
			}
			_options.testNumber = _tests.size();
			ifs.close();
			return true;
		}

		bool CalculateObjectDescriptors(Test* tests, size_t batch, size_t index, size_t thread, size_t & repeat, size_t & progress)
		{
			Thread& t = _threads[thread];
			for (size_t b = 0; b < batch; ++b)
			{
				const Object & o = tests[b].objects[index];
				if (!SetInput(o.path, t.input[0], b, NULL))
					return false;
			}
			for (int i = 0; i < _options.repeatNumber; ++i, ++repeat)
			{
				t.output = t.network->Predict(t.input);
				if (repeat & 1)
					progress += batch;
			}
			size_t size = t.output[0].Size(1);
			for (size_t b = 0; b < batch; ++b)
			{
				Object& o = tests[b].objects[index];
				o.desc.Reshape(Shp(size));
				memcpy(o.desc.CpuData(), t.output[0].CpuData() + b * size, o.desc.RawSize());
			}
			return true;
		}

		bool CalculateDistances(Test * tests, size_t batch)
		{
			for (size_t b = 0; b < batch; ++b)
			{
				Object* o = tests[b].objects;
				SimdCosineDistance32f(o[0].desc.CpuData(), o[1].desc.CpuData(), o[0].desc.Size(), &tests[b].distance);
			}
			return true;
		}

		virtual bool PerformBatch(size_t thread, size_t current, size_t batch, size_t& progress)
		{
			bool result = true;
			Test* tests = _tests.data() + current;
			size_t repeat = 0;
			result = result && CalculateObjectDescriptors(tests, batch, 0, thread, repeat, progress);
			result = result && CalculateObjectDescriptors(tests, batch, 1, thread, repeat, progress);
			result = result && CalculateDistances(tests, batch);
			return result;
		}

		String PrintResume(size_t number, double precision, double threshold)
		{
			std::stringstream ss;
			ss << "Number: " << number << ", precision: " << ToString(precision * 100, 2);
			ss << " %, threshold: " << ToString(threshold, 3);
			return ss.str();
		}

		virtual bool ProcessResult()
		{
			typedef std::pair<float, int> Pair;
			std::vector<Pair> tests(_tests.size());
			int negatives = 0, positives = 0;
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				tests[i].first = _tests[i + 0].distance;
				if (_tests[i].objects[0].name == _tests[i].objects[1].name)
					tests[i].second = +1, positives++;
				else
					tests[i].second = -1, negatives++;
				tests[i].second = _tests[i].objects[0].name == _tests[i].objects[1].name ? 1 : -1;
			}
			std::sort(tests.begin(), tests.end(), [](const Pair& a, const Pair& b) {return a.first < b.first; });
			size_t idx = tests.size(), num = negatives, max = num;
			for (size_t i = 0; i < tests.size(); ++i)
			{
				num += tests[i].second;
				if (num > max)
				{
					max = num;
					idx = i;
				}
			}
			if (idx >= tests.size() - 1)
				SYNET_ERROR("Can't process result!");
			_options.resume = PrintResume(tests.size(), double(max) / tests.size(), (tests[idx].first + tests[idx + 1].first) / 2);
			return true;
		}
	};
}



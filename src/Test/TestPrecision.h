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

#pragma once

#include "TestCommon.h"
#include "TestArgs.h"
#include "TestSynet.h"
#ifdef SYNET_TEST_FIRST_RUN
#include "TestInferenceEngine.h"
#endif
#include "TestImage.h"

namespace Test
{
	class Precision
	{
	public:
		struct Options : public ArgsParser
		{
			String mode;
			String framework;
			String testModel;
			String testWeight;
			String testParam;
			String testList;
			String imageDirectory;
			String outputDirectory;
			String logName;
			bool consoleSilence;
			int testThreads;

			mutable volatile bool result;
			mutable size_t memoryUsage, testNumber;
			mutable double precision, threshold;

			Options(int argc, char* argv[])
				: ArgsParser(argc, argv)
				, result(true)
				, memoryUsage(0)
			{
				mode = GetArg("-m", "reidentification");
				framework = GetArg("-f", "synet");
				testModel = GetArg("-tm", "sy_fp32.xml");
				testWeight = GetArg("-tw", "sy_fp32.bin");
				testParam = GetArg("-tp", "param.xml");
				testList = GetArg("-tl", "pairs1.txt");
				imageDirectory = GetArg("-id", "image");
				outputDirectory = GetArg("-od", "output");
				logName = GetArg("-ln", "", false);
				consoleSilence = FromString<bool>(GetArg("-cs", "0"));
				testThreads = FromString<int>(GetArg("-tt", "1"));
			}

			~Options()
			{
				if (result)
				{
					std::stringstream ss;

					ss << "Test info: (" << Description() << ")." << std::endl;
					ss << "Precision: " << ToString(precision * 100, 2) << " %, threshold: " << ToString(threshold, 3) << std::endl;
					if (memoryUsage)
						ss << "Memory usage: " << memoryUsage / (1024 * 1024) << " MB." << std::endl;
					ss << SystemInfo() << std::endl;
					PerformanceMeasurerStorage::s_storage.Print(ss);
#if defined(SYNET_SIMD_LIBRARY_ENABLE)
					if (framework == "synet")
						ss << SimdPerformanceStatistic();
#endif
					if (!consoleSilence)
						std::cout << ss.str() << std::endl;
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
					}
				}
			}

			bool NeedOutputDirectory() const
			{
				return false;
			}

			size_t TestThreads() const
			{
				return Synet::RestrictRange<size_t>(testThreads, 1, std::thread::hardware_concurrency());
			}

			String Description() const
			{
				std::stringstream ss;
				ss << "framework: " << framework;
				ss << ", test: " << GetNameByPath(DirectoryByPath(testModel));
				ss << ", model: " << GetNameByPath(testModel);
				ss << ", list: " << GetNameByPath(testList);
				ss << ", threads: " << TestThreads();
				return ss.str();
			}
		};

		Precision(const Options& options)
			: _options(options)
			, _progressMessageSizeMax(0)
		{

		}

		bool Run()
		{
			PrintStartMessage();
			if (!LoadTestParam())
				return false;
			if (!CreateDirectories())
				return false;
			if (!InitNetwork())
				return false;
			if (!LoadTestList())
				return false;
			if (!PerformTests())
				return false;
			if (!ProcessResult())
				return false;
			return PrintFinishMessage();
		}

	private:
		struct Object
		{
			String name, path;
			Tensors input;
			Tensor desc;
		};
		typedef std::shared_ptr<Object> ObjectPtr;

		struct Test
		{
			Object first, second;
			float distance;
		};
		typedef std::vector<Test> Tests;
		typedef std::shared_ptr<Test> TestPtr;
		typedef std::vector<TestPtr> TestPtrs;
		
		struct Thread
		{
			size_t begin, end, current;
			std::thread thread;
			Thread() : begin(0), end(0), current(0) {}
		};
		typedef std::vector<Thread> Threads;

		const Options& _options;
		TestParamHolder _param;
		NetworkPtrs _networks;
		Tests _tests;
		Threads _threads;
		size_t _progressMessageSizeMax;

		void PrintStartMessage() const
		{
			std::cout << "Start test (" << _options.Description() << "):" << std::endl;
		}

		bool PrintFinishMessage() const
		{
			std::cout << ExpandRight("Test is finished.", _progressMessageSizeMax) << std::endl << std::endl;
			return true;
		}

		bool LoadTestParam()
		{
			if (!_param.Load(_options.testParam))
			{
				std::cout << "Can't load file '" << _options.testParam << "' !" << std::endl;
				return false;
			}
			return true;
		}

		bool CreateDirectories()
		{
			if (_options.NeedOutputDirectory() && !DirectoryExists(_options.outputDirectory) && !CreatePath(_options.outputDirectory))
			{
				std::cout << "Can't create output directory '" << _options.outputDirectory << "' !" << std::endl;
				return false;
			}
			return true;
		}

		bool InitNetwork()
		{
			if (!FileExists(_options.testModel))
			{
				std::cout << "Model file '" << _options.testModel << "' is not exist!" << std::endl;
				return false;
			}
			if (!FileExists(_options.testWeight))
			{
				std::cout << "Weight file '" << _options.testWeight << "' is not exist!" << std::endl;
				return false;
			}
			_networks.resize(_options.testThreads);
			for (size_t i = 0; i < _networks.size(); ++i)
			{
				if (_options.framework == "synet")
					_networks[i] = std::make_shared<SynetNetwork>();
#ifdef SYNET_TEST_FIRST_RUN
				else if (_options.framework == "inference_engine")
					_networks[i] = std::make_shared<InferenceEngineNetwork>();
#endif
				else
				{
					std::cout << "Unknown framework: " << _options.framework << "!" << std::endl;
					return false;
				}
				Network::Options options(_options.outputDirectory, 1, true, 1, 0, 0.5f);
				if (!_networks[i]->Init(_options.testModel, _options.testWeight, options, _param()))
				{
					std::cout << "Can't load " << _networks[i]->Name() << " from '" << _options.testModel << "' and '" << _options.testWeight << "' !" << std::endl;
					return false;
				}
				Shape shape = _networks[i]->SrcShape(0);
				if (shape.size() != 4 || (shape[1] != 3 && shape[1] != 1))
				{
					std::cout << "Wrong " << _networks[i]->Name() << " classifier input shape: " << Synet::Detail::DebugPrint(shape) << " !" << std::endl;
					return false;
				}
			}
			return true;
		}

		bool SetPath(const String& name, int index, String& path)
		{
			path = MakePath(_options.imageDirectory, name + "_" + ToString(index, 4) + ".jpg");
			if(!FileExists(path))
			{
				std::cout << "Test image '" << path << "' is not exist!" << std::endl;
				return false;
			}
			return true;
		}

		bool LoadTestList()
		{
			if (!DirectoryExists(_options.imageDirectory))
			{
				std::cout << "Image directory '" << _options.imageDirectory << "' is not exist!" << std::endl;
				return false;
			}
			if (!FileExists(_options.testList))
			{
				std::cout << "Test list file '" << _options.testList << "' is not exist!" << std::endl;
				return false;
			}
			std::ifstream ifs(_options.testList);
			if (!ifs.is_open())
			{
				std::cout << "Can't open test list file '" << _options.testList << "'!" << std::endl;
				return false;
			}
			String line;
			if (!std::getline(ifs, line))
			{
				std::cout << "Can't read 1-st file line!" << std::endl;
				return false;
			}
			size_t size = FromString<int>(line);
			if (size > USHRT_MAX)
			{
				std::cout << "Wrong test size " << size << " !" << std::endl;
				return false;
			}
			_tests.resize(size * 2);
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				if (!std::getline(ifs, line))
				{
					std::cout << "Can't read " << i + 2 << " file line!" << std::endl;
					return false;
				}
				std::stringstream ss(line);
				int first, second;
				if (i < size)
				{
					ss >> _tests[i].first.name >> first >> second;
					_tests[i].second.name = _tests[i].first.name;
				}
				else
					ss >> _tests[i].first.name >> first >> _tests[i].second.name >> second;
				if (_tests[i].first.name.empty() || _tests[i].second.name.empty() || first == 0 || second == 0)
				{
					std::cout << "Can't parse " << i + 2 << " file line!" << std::endl;
					return false;
				}
				if (!SetPath(_tests[i].first.name, first, _tests[i].first.path))
					return false;
				if (!SetPath(_tests[i].second.name, second, _tests[i].second.path))
					return false;
			}
			ifs.close();
			return true;
		}

		bool CalculateFaceDescriptor(Object& object, size_t thread)
		{
			TEST_PERF_FUNC();

			View original;
			if (!LoadImage(object.path, original))
			{
				std::cout << "Can't read '" << object.path << "' image!" << std::endl;
				return false;
			}
			Shape shape = _networks[thread]->SrcShape(0);
			object.input.resize(1, Tensor(shape));
			Floats lower = _param().lower(), upper = _param().upper();
			if (lower.size() == 1)
				lower.resize(shape[1], lower[0]);
			if (upper.size() == 1)
				upper.resize(shape[1], upper[0]);

			View converted(original.Size(), shape[1] == 1 ? View::Gray8 : View::Bgr24);
			Simd::Convert(original, converted);
			View resized(Size(shape[3], shape[2]), converted.format);
			Simd::Resize(converted, resized, SimdResizeMethodArea);

			Views channels(shape[1]);
			if (shape[1] > 1)
			{
				for (size_t i = 0; i < shape[1]; ++i)
					channels[i].Recreate(resized.Size(), View::Gray8);
				Simd::DeinterleaveBgr(resized, channels[0], channels[1], channels[2]);
			}
			else
				channels[0] = resized;
			float * input = object.input[0].CpuData();
			for (size_t c = 0; c < channels.size(); ++c)
			{
				for (size_t y = 0; y < channels[c].height; ++y)
				{
					const uint8_t* row = channels[c].Row<uint8_t>(y);
					::SimdUint8ToFloat32(row, channels[c].width, &lower[c], &upper[c], input);
					input += channels[c].width;
				}
			}
			object.desc.Clone(_networks[thread]->Predict(object.input)[0]);
			return true;
		}

		bool CalculateDistance(Test& test)
		{
			SimdCosineDistance32f(test.first.desc.CpuData(), test.second.desc.CpuData(), test.first.desc.Size(), &test.distance);
			return true;
		}

		static void ThreadTask(Precision * precision, size_t thread)
		{
			size_t& current = precision->_threads[thread].current;
			volatile bool& result = precision->_options.result;
			for (; current < precision->_threads[thread].end && result; current++)
			{
				result = result && precision->CalculateFaceDescriptor(precision->_tests[current].first, thread);
				result = result && precision->CalculateFaceDescriptor(precision->_tests[current].second, thread);
				precision->CalculateDistance(precision->_tests[current]);
			}
			if (!result)
				std::cout << "Error at " << current << " test!" << std::endl;
		}

		String ProgressString(size_t current, size_t total)
		{
			std::stringstream progress;
			progress << "Test progress : " << ToString(100.0 * current / total, 1) << "% ";
			_progressMessageSizeMax = std::max(_progressMessageSizeMax, progress.str().size());
			return progress.str();
		}

		bool PerformTests()
		{
			size_t current = 0, total = _tests.size();
			_threads.resize(_options.testThreads);
			size_t part = Synet::DivHi(_tests.size(), _threads.size());
			for (size_t t = 0; t < _threads.size(); ++t)
			{
				_threads[t].begin = t * part;
				_threads[t].current = t * part;
				_threads[t].end = std::min((t + 1) * part, _tests.size());
				_threads[t].thread = std::thread(ThreadTask, this, t);
			}

			while (current < total && _options.result)
			{
				current = 0;
				for (size_t t = 0; t < _threads.size(); ++t)
					current += _threads[t].current - _threads[t].begin;
				std::cout << ProgressString(current, total) << std::flush;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				std::cout << " \r" << std::flush;
			}

			for (size_t t = 0; t < _threads.size(); ++t)
			{
				if (_threads[t].thread.joinable())
					_threads[t].thread.join();
				_options.memoryUsage += _networks[t]->MemoryUsage();
			}

			return _options.result;
		}

		bool ProcessResult()
		{
			typedef std::pair<float, int> Pair;
			std::vector<Pair> tests(_tests.size());
			int negatives = 0, positives = 0;
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				tests[i].first = _tests[i + 0].distance;
				if (_tests[i].first.name == _tests[i].second.name)
					tests[i].second = +1, positives++;
				else
					tests[i].second = -1, negatives++;
				tests[i].second = _tests[i].first.name == _tests[i].second.name ? 1 : -1;
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
			{
				std::cout << "Can't process result!" << std::endl;
				return false;
			}
			_options.testNumber = tests.size();
			_options.precision = double(max) / tests.size();
			_options.threshold = (tests[idx].first + tests[idx + 1].first) / 2;
			return true;
		}
	};
}



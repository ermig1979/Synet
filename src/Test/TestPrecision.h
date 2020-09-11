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
			int batchSize;

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
				batchSize = FromString<int>(GetArg("-bs", "1"));
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

			size_t BatchSize() const
			{
				return Synet::Max<size_t>(batchSize, 1);
			}

			String Description() const
			{
				std::stringstream ss;
				ss << "framework: " << framework;
				ss << ", test: " << GetNameByPath(DirectoryByPath(testModel));
				ss << ", model: " << GetNameByPath(testModel);
				ss << ", list: " << GetNameByPath(testList);
				ss << ", threads: " << TestThreads();
				ss << ", batch: " << BatchSize();
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
		
		struct Thread
		{
			NetworkPtr network;
			Tensors input, output;
			size_t begin, end, current;
			std::thread thread;
			Thread() : begin(0), end(0), current(0) {}
		};
		typedef std::vector<Thread> Threads;

		const Options& _options;
		TestParamHolder _param;
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
			_threads.resize(_options.testThreads);
			for (size_t t = 0; t < _threads.size(); ++t)
			{
				NetworkPtr& network = _threads[t].network;
				if (_options.framework == "synet")
					network = std::make_shared<SynetNetwork>();
#ifdef SYNET_TEST_FIRST_RUN
				else if (_options.framework == "inference_engine")
					network = std::make_shared<InferenceEngineNetwork>();
#endif
				else
				{
					std::cout << "Unknown framework: " << _options.framework << "!" << std::endl;
					return false;
				}
				Network::Options options(_options.outputDirectory, 1, true, _options.batchSize, 0, 0.5f);
				if (!network->Init(_options.testModel, _options.testWeight, options, _param()))
				{
					std::cout << "Can't load " << network->Name() << " from '" << _options.testModel << "' and '" << _options.testWeight << "' !" << std::endl;
					return false;
				}
				Shape shape = network->SrcShape(0);
				if (shape.size() != 4 || (shape[1] != 3 && shape[1] != 1))
				{
					std::cout << "Wrong " << network->Name() << " classifier input shape: " << Synet::Detail::DebugPrint(shape) << " !" << std::endl;
					return false;
				}
				_threads[t].input.resize(1, Tensor(shape));
				if (_param().lower().size() == 1)
					_param().lower().resize(shape[1], _param().lower()[0]);
				if (_param().upper().size() == 1)
					_param().upper().resize(shape[1], _param().upper()[0]);
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
					ss >> _tests[i].objects[0].name >> first >> second;
					_tests[i].objects[1].name = _tests[i].objects[0].name;
				}
				else
					ss >> _tests[i].objects[0].name >> first >> _tests[i].objects[1].name >> second;
				if (_tests[i].objects[0].name.empty() || _tests[i].objects[1].name.empty() || first == 0 || second == 0)
				{
					std::cout << "Can't parse " << i + 2 << " file line!" << std::endl;
					return false;
				}
				if (!SetPath(_tests[i].objects[0].name, first, _tests[i].objects[0].path))
					return false;
				if (!SetPath(_tests[i].objects[1].name, second, _tests[i].objects[1].path))
					return false;
			}
			ifs.close();
			return true;
		}

		bool SetInput(const String & path, Tensor & input, size_t index)
		{
			TEST_PERF_FUNC();

			View original;
			if (!LoadImage(path, original))
			{
				std::cout << "Can't read '" << path << "' image!" << std::endl;
				return false;
			}

			const Shape& shape = input.Shape();
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

			float* ptr = input.CpuData(Shp(index, 0, 0, 0));
			for (size_t c = 0; c < channels.size(); ++c)
			{
				for (size_t y = 0; y < channels[c].height; ++y)
				{
					const uint8_t* row = channels[c].Row<uint8_t>(y);
					::SimdUint8ToFloat32(row, channels[c].width, &_param().lower()[c], &_param().upper()[c], ptr);
					ptr += channels[c].width;
				}
			}

			return true;
		}

		bool CalculateFaceDescriptors(Test* tests, size_t batch, size_t index, size_t thread)
		{
			TEST_PERF_FUNC();

			Thread& t = _threads[thread];
			for (size_t b = 0; b < batch; ++b)
			{
				const Object & o = tests[b].objects[index];
				if (!SetInput(o.path, t.input[0], b))
					return false;
			}
			t.output = t.network->Predict(t.input);
			size_t size = t.output[0].Size(1);
			for (size_t b = 0; b < batch; ++b)
			{
				Object& o = tests[b].objects[index];
				o.desc.Reshape(Shp(size));
				memcpy(o.desc.CpuData(), t.output[0].CpuData() + b * size, o.desc.RawSize());
			}
			return true;
		}

		void CalculateDistances(Test * tests, size_t batch)
		{
			for (size_t b = 0; b < batch; ++b)
			{
				Object* o = tests[b].objects;
				SimdCosineDistance32f(o[0].desc.CpuData(), o[1].desc.CpuData(), o[0].desc.Size(), &tests[b].distance);
			}
		}

		static void ThreadTask(Precision * precision, size_t thread)
		{
			size_t& current = precision->_threads[thread].current;
			size_t end = precision->_threads[thread].end;
			volatile bool& result = precision->_options.result;
			for (; current < end && result;)
			{
				size_t batch = std::min(precision->_options.BatchSize(), end - current);
				Test* tests = precision->_tests.data() + current;
				result = result && precision->CalculateFaceDescriptors(tests, batch, 0, thread);
				result = result && precision->CalculateFaceDescriptors(tests, batch, 1, thread);
				precision->CalculateDistances(tests, batch);
				current += batch;
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
				_options.memoryUsage += _threads[t].network->MemoryUsage();
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



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

#include "TestCommon.h"
#include "TestSynet.h"
#ifdef SYNET_TEST_FIRST_RUN
#if defined(SYNET_TEST_OPENVINO_API)
#include "TestOpenVino.h"
#else
#include "TestInferenceEngine.h"
#endif
#endif

#include "Cpl/Args.h"

namespace Test
{
	class Precision
	{
	public:
		struct Options : public Cpl::ArgsParser
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
			int repeatNumber;
			float ratioVariation;
			float compareOverlap;
			bool adaptiveThreshold;
			int performanceLog;
			bool annotateRegions;
			bool generateIndex;
			double statFilter;

			mutable volatile bool result;
			mutable size_t memoryUsage, testNumber;
			mutable String resume;

			Options(int argc, char* argv[])
				: ArgsParser(argc, argv, true)
				, result(true)
				, memoryUsage(0)
				, testNumber(0)
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
				batchSize = Synet::Max(FromString<int>(GetArg("-bs", "1")), 1);
				repeatNumber = Synet::Max(FromString<int>(GetArg("-rn", "1")), 1);
				ratioVariation = FromString<float>(GetArg("-rv", "0.500"));
				compareOverlap = FromString<float>(GetArg("-co", "0.5"));
				adaptiveThreshold = FromString<bool>(GetArg("-at", "1"));
				performanceLog = FromString<int>(GetArg("-pl", "0"));
				annotateRegions = FromString<bool>(GetArg("-ar", "0"));
				generateIndex = FromString<bool>(GetArg("-gi", "0"));
				statFilter = FromString<double>(GetArg("-sf", "0.0"));
			}

			~Options()
			{
				if (result)
				{
					std::stringstream ss;
					ss << resume << std::endl;
					if (memoryUsage)
						ss << "Memory usage: " << MemoryUsageString(memoryUsage, testThreads) << std::endl;
					PrintPerformance(ss, statFilter);
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
				return annotateRegions;
			}

			size_t TestThreads() const
			{
				return Synet::RestrictRange<size_t>(testThreads, 1, std::thread::hardware_concurrency());
			}

			String Description() const
			{
				std::stringstream ss;
				ss << "mode: " << mode;
				ss << ", framework: " << framework;
				ss << ", test: " << GetNameByPath(DirectoryByPath(testModel));
				ss << ", model: " << GetNameByPath(testModel);
				ss << ", list: " << GetNameByPath(testList);
				ss << ", threads: " << TestThreads();
				ss << ", batch: " << batchSize;
				ss << ", repeats: " << repeatNumber;
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

	protected:

		struct Thread
		{
			NetworkPtr network;
			Tensors input, output;
			size_t begin, end, current, progress;
			std::thread thread;
			Thread() : begin(0), end(0), current(0), progress(0) {}
		};
		typedef std::vector<Thread> Threads;

		const Options& _options;
		TestParamHolder _param;
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
				SYNET_ERROR("Can't create output directory '" << _options.outputDirectory << "' !");
			return true;
		}

		bool InitNetwork()
		{
			if (!FileExists(_options.testModel))
				SYNET_ERROR("Model file '" << _options.testModel << "' is not exist!");
			if (!FileExists(_options.testWeight))
				SYNET_ERROR("Weight file '" << _options.testWeight << "' is not exist!");
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
					SYNET_ERROR("Unknown framework: " << _options.framework << "!");
				Network::Options options(_options.outputDirectory, 1, true, _options.batchSize, _options.performanceLog, 0, 0.5f, false);
				if (!network->Init(_options.testModel, _options.testWeight, options, _param()))
					SYNET_ERROR("Can't load " << network->Name() << " from '" << _options.testModel << "' and '" << _options.testWeight << "' !");
				Shape shape = network->SrcShape(0);
				if (shape.size() != 4 || (shape[1] != 3 && shape[1] != 1))
					SYNET_ERROR("Wrong " << network->Name() << " classifier input shape: " << Synet::Detail::DebugPrint(shape) << " !");
				_threads[t].input.resize(1, Tensor(shape));
				if (_param().lower().size() == 1)
					_param().lower().resize(shape[1], _param().lower()[0]);
				if (_param().upper().size() == 1)
					_param().upper().resize(shape[1], _param().upper()[0]);
			}
			return true;
		}

		template<class Test> bool GenerateIndex(std::vector<Test> & tests)
		{
			StringList names = GetFileList(_options.imageDirectory, "*.jpg", true, false);
			tests.clear();
			if (names.empty())
				SYNET_ERROR("Directory '" << _options.imageDirectory << "' is empty!");
			for (StringList::const_iterator it = names.begin(); it != names.end(); ++it)
			{
				Test test;
				test.name = *it;
				test.path = MakePath(_options.imageDirectory, test.name);
				if (ExtensionByPath(test.name) == "jpg")
					tests.push_back(test);
			}
			_options.testNumber = tests.size();
			return true;
		}

		bool SetInput(const String & path, Tensor & input, size_t index, Size * pSize)
		{
			CPL_PERF_FUNC();

			View original;
			if (!LoadImage(path, original))
				SYNET_ERROR("Can't read '" << path << "' image!");
			if (pSize)
				*pSize = original.Size();

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

		String ProgressString(size_t current, size_t total)
		{
			std::stringstream progress;
			progress << "Test progress : " << ToString(100.0 * current / total, 1) << "% ";
			_progressMessageSizeMax = std::max(_progressMessageSizeMax, progress.str().size());
			return progress.str();
		}

		static void ThreadTask(Precision* precision, size_t thread)
		{
			size_t& current = precision->_threads[thread].current;
			size_t& progress = precision->_threads[thread].progress;
			size_t end = precision->_threads[thread].end;
			volatile bool& result = precision->_options.result;
			for (; current < end && result;)
			{
				size_t batch = Synet::Min<size_t>(precision->_options.batchSize, end - current);
				result = precision->PerformBatch(thread, current, batch, progress);
				current += batch;
			}
			if (!result)
				CPL_LOG_SS(Error, "Error at " << current << " test!");
		}

		bool PerformTests()
		{
			size_t current = 0, total = _options.testNumber * _options.repeatNumber;
			size_t part = Synet::DivHi(_options.testNumber, _threads.size());
			for (size_t t = 0; t < _threads.size(); ++t)
			{
				_threads[t].begin = t * part;
				_threads[t].current = t * part;
				_threads[t].end = std::min((t + 1) * part, _options.testNumber);
				_threads[t].thread = std::thread(ThreadTask, this, t);
			}

			while (current < total && _options.result)
			{
				current = 0;
				for (size_t t = 0; t < _threads.size(); ++t)
					current += _threads[t].progress;
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

		virtual bool LoadTestList() = 0;
		virtual bool PerformBatch(size_t thread, size_t current, size_t batch, size_t & progress) = 0;
		virtual bool ProcessResult() = 0;
	};
}



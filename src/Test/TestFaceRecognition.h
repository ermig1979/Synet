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

#include "TestCommon.h"
#include "TestArgs.h"
#include "TestImage.h"
#ifdef SYNET_TEST_FIRST_RUN
#include "TestInferenceEngine.h"
#endif
#include "TestSynet.h"

namespace Test
{
	class FaceRecognition
	{
	public:
		struct Options : public ArgsParser
		{
			String framework;
			String testModel;
			String testWeight;
			String testParam;
			String testList;
			String imageDirectory;
			String outputDirectory;
			bool result;

			Options(int argc, char* argv[])
				: ArgsParser(argc, argv)
				, result(false)
			{
				framework = GetArg("-f", "synet");
				testModel = GetArg("-tm", "sy_fp32.xml");
				testWeight = GetArg("-tw", "sy_fp32.bin");
				testParam = GetArg("-tp", "param.xml");
				testList = GetArg("-tl", "pairs10.txt");
				imageDirectory = GetArg("-id", "image");
				outputDirectory = GetArg("-od", "output");
			}

			bool NeedOutputDirectory() const
			{
				return false;
			}
		};

		FaceRecognition(const Options& options)
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
			return PrintFinishMessage();
		}

	private:
		struct Face
		{
			String name, path;
			Tensors input;
			Tensor desc;
		};
		typedef std::shared_ptr<Face> FacePtr;

		struct Test
		{
			Face first, second;
			float distance;
		};
		typedef std::vector<Test> Tests;
		typedef std::shared_ptr<Test> TestPtr;
		typedef std::vector<TestPtr> TestPtrs;

		const Options& _options;
		TestParamHolder _param;
		NetworkPtr _network;
		Tests _tests;
		size_t _progressMessageSizeMax;

		void PrintStartMessage() const
		{
			std::cout << "Start test for " << _options.testModel << " model :" << std::endl;
		}

		bool PrintFinishMessage() const
		{
			std::cout << ExpandRight("Test is finished successfully!", _progressMessageSizeMax) << std::endl << std::endl;
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
			if (_options.framework == "synet")
				_network = std::make_shared<SynetNetwork>();
#ifdef SYNET_TEST_FIRST_RUN
			else if(_options.framework == "inference_engine")
				_network = std::make_shared<InferenceEngineNetwork>();
#endif
			else
			{
				std::cout << "Unknown framework: " << _options.framework << "!" << std::endl;
				return false;
			}
			Network::Options options(_options.outputDirectory, 1, true, 1, 0, 0.5f);
			if (!_network->Init(_options.testModel, _options.testWeight, options, _param()))
			{
				std::cout << "Can't load " << _network->Name() << " from '" << _options.testModel << "' and '" << _options.testWeight << "' !" << std::endl;
				return false;
			}
			Shape shape = _network->SrcShape(0);
			if (shape.size() != 4 || (shape[1] != 3 && shape[1] != 1))
			{
				std::cout << "Wrong " << _network->Name() << " classifier input shape: " << Synet::Detail::DebugPrint(shape) << " !" << std::endl;
				return false;
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

		bool CalculateFaceDescriptor(Face & face)
		{
			View original;
			if (!LoadImage(face.path, original))
			{
				std::cout << "Can't read '" << face.path << "' image!" << std::endl;
				return false;
			}
			Shape shape = _network->SrcShape(0);
			face.input.resize(1, Tensor(shape));
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
			float * input = face.input[0].CpuData();
			for (size_t c = 0; c < channels.size(); ++c)
			{
				for (size_t y = 0; y < channels[c].height; ++y)
				{
					const uint8_t* row = channels[c].Row<uint8_t>(y);
					::SimdUint8ToFloat32(row, channels[c].width, &lower[c], &upper[c], input);
					input += channels[c].width;
				}
			}
			face.desc.Clone(_network->Predict(face.input)[0]);
			return true;
		}

		bool CalculateDistance(Test& test)
		{
			SimdCosineDistance32f(test.first.desc.CpuData(), test.second.desc.CpuData(), test.first.desc.Size(), &test.distance);
			return true;
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
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				std::cout << ProgressString(i, _tests.size()) << std::flush;
				if (!(CalculateFaceDescriptor(_tests[i].first) && CalculateFaceDescriptor(_tests[i].second) && CalculateDistance(_tests[i])))
				{
					std::cout << "Error at " << i << " test!" << std::endl;
					return false;
				}
				std::cout << "\r" << std::flush;
			}
			return true;
		}
	};
}



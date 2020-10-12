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

#include "TestPrecision.h"

namespace Test
{
	class DetectionPrecision : public Precision
	{
	public:
		DetectionPrecision(const Options& options)
			: Precision(options)
		{
		}

	private:

		struct Test
		{
			String name, path;
			Regions detected, control;
		};

		typedef std::vector<Test> Tests;
		typedef std::shared_ptr<Test> TestPtr;
		typedef std::vector<TestPtr> TestPtrs;

		Tests _tests;

		bool ParseIndexFile()
		{
			String path = MakePath(_options.imageDirectory, _options.indexFile);
			std::ifstream ifs(path);
			if (!ifs.is_open())
			{
				std::cout << "Can't open file '" << path << "' !" << std::endl;
				return false;
			}
			while (!ifs.eof())
			{
				Test test;
				ifs >> test.name;
				test.path = MakePath(_options.imageDirectory, test.name);
				if (!FileExists(test.path))
				{
					std::cout << "Image '" << test.path << "' is not exists!" << std::endl;
					return false;
				}
				size_t number;
				ifs >> number;
				for (size_t i = 0; i < number; ++i)
				{
					Region region;
					ifs >> region.x >> region.y >> region.w >> region.h >> region.id;
					for (size_t j = 0, stub; j < 5; ++j)
						ifs >> stub;
					if(region.id == 0)
						test.control.push_back(region);
				}
				_tests.push_back(test);
			}
			return true;
		}

		virtual bool LoadTestList()
		{
			if (!ParseIndexFile())
				return false;

			return true;
		}

		virtual bool PerformBatch(size_t thread, size_t current, size_t batch)
		{
			return true;
		}

		virtual bool ProcessResult()
		{
			return true;
		}
	};
}



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
			bool skip;
			String name, path;
			Regions current, control;
		};

		typedef std::vector<Test> Tests;
		typedef std::shared_ptr<Test> TestPtr;
		typedef std::vector<TestPtr> TestPtrs;
		typedef std::set<String> StringSet;

		Tests _tests;
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
				std::cout << "Can't open list file '" << path << "' !" << std::endl;
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
			{
				std::cout << "Can't open file '" << path << "' !" << std::endl;
				return false;
			}
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				if (_tests[i].skip)
					continue;
				ofs << _tests[i].name << std::endl;
			}
			return true;
		}

		bool LoadTextIndexFile()
		{
			String path = MakePath(_options.imageDirectory, _param().index().name());
			std::ifstream ifs(path);
			if (!ifs.is_open())
			{
				std::cout << "Can't open file '" << path << "' !" << std::endl;
				return false;
			}
			_tests.clear();
			while (!ifs.eof())
			{
				Test test;
				ifs >> test.name;
				if (test.name.empty())
					break;
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
					region.x += int(region.w) / 2;
					region.y += int(region.h) / 2;
					for (size_t j = 0, stub; j < 5; ++j)
						ifs >> stub;
					bool add = false;
					for (size_t j = 0; j < _param().index().ids().size(); ++j)
						if (_param().index().ids()[j].id() == region.id)
							add = true;
					if(add)
						test.control.push_back(region);
				}
				if(_list.empty() || _list.find(test.name) != _list.end())
					_tests.push_back(test);
			}
			_options.testNumber = _tests.size();
			return true;
		}

		typedef Synet::Xml::XmlBase<char> XmlBase;
		typedef Synet::Xml::XmlNode<char> XmlNode;

		template<class T> static bool Convert(const XmlBase * src, T& dst)
		{
			if (src == NULL)
				return false;
			Synet::StringToValue(src->Value(), dst);
			return true;
		}

		bool LoadRegion(const XmlNode & object, Region& region)
		{
			XmlNode * pName = object.FirstNode("name");
			if (pName == NULL)
			{
				std::cout << "Can't find <name> node!" << std::endl;
				return false;
			}
			region.id = -1;
			for (size_t i = 0; i < _param().index().ids().size(); ++i)
				if (_param().index().ids()[i].name() == pName->Value())
					region.id = _param().index().ids()[i].id();
			if(region.id == -1)
				return false;
			XmlNode * pBndbox = object.FirstNode("bndbox");
			if (pBndbox == NULL)
			{
				std::cout << "Can't find <bndbox> node!" << std::endl;
				return false;
			}
			int xmin, ymin, xmax, ymax;
			if (!Convert(pBndbox->FirstNode("xmin"), xmin))
				return false;
			if (!Convert(pBndbox->FirstNode("ymin"), ymin))
				return false;			
			if (!Convert(pBndbox->FirstNode("xmax"), xmax))
				return false;
			if (!Convert(pBndbox->FirstNode("ymax"), ymax))
				return false;
			region.x = float(xmin + xmax) * 0.5f;
			region.y = float(ymin + ymax) * 0.5f;
			region.w = float(xmax - xmin);
			region.h = float(ymax - ymin);
			return true;
		}

		bool LoadXmlIndexFile(Test& test)
		{
			String original = WithoutExtension(test.path) + ".xml";
			String corrected = WithoutExtension(test.path) + "_.xml";
			String path = FileExists(corrected) ? corrected : original;
			std::ifstream ifs(path.c_str());
			if (ifs.is_open())
			{
				using namespace Synet::Xml;
				File<char> file(ifs);
				XmlDocument<char> doc;
				try
				{
					doc.Parse<0>(file.Data());
				}
				catch (std::exception& e)
				{
					std::cout << "Can't parse xml '" << path << "' file! There is an exception: " << e.what() << std::endl;
					return false;
				}
				XmlNode * pAnnotation = doc.FirstNode("annotation");
				if (pAnnotation == NULL)
				{
					std::cout << "Can't find <annotation> node!" << std::endl;
					return false;
				}
				XmlNode* pObject = pAnnotation->FirstNode("object");
				while(pObject)
				{
					Region region;
					if(LoadRegion(*pObject, region))
						test.control.push_back(region);
					pObject = pObject->NextSibling();
				}
				ifs.close();
			}
			else
			{
				std::cout << "Can't open '" << path << "' index file!" << std::endl;
				return false;
			}
			return true;
		}

		bool LoadXmlIndexFiles()
		{
			_tests.clear();
			for (StringSet::const_iterator it = _list.begin(); it != _list.end(); ++it)
			{
				Test test;
				test.name = *it;
				test.path = MakePath(_options.imageDirectory, test.name);
				if (LoadXmlIndexFile(test))
					_tests.push_back(test);
				else
					return false;
			}
			_options.testNumber = _tests.size();
			return true;
		}

		bool LoadIndex()
		{
			if (_param().index().type() == "DetectionTextV1")
				return LoadTextIndexFile();
			else if(_param().index().type() == "DetectionXmlFilesV1")
				return LoadXmlIndexFiles();
			else
				return false;
		}

		bool SaveTextIndexFile()
		{
			String path = MakePath(_options.imageDirectory, _param().index().name());
			std::ofstream ofs(path);
			if (!ofs.is_open())
			{
				std::cout << "Can't open file '" << path << "' !" << std::endl;
				return false;
			}
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				const Test& t = _tests[i];
				ofs << t.name << std::endl;
				ofs << t.control.size() << std::endl;
				for (size_t k = 0; k < t.control.size(); ++k)
				{
					const Region& r = t.control[k];
					ofs << int(r.x) - int(r.w) / 2 << " ";
					ofs << int(r.y) - int(r.h) / 2 << " ";
					ofs << int(r.w) << " " << int(r.h) << " ";
					ofs << r.id << " 0 0 0 0 " << k << std::endl;
				}
			}
			return true;
		}

		bool SaveIndex()
		{
			if (_param().index().type() == "DetectionTextV1")
				return SaveTextIndexFile();
			else if (_param().index().type() == "DetectionXmlFilesV1")
				return true;
			else
				return false;
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
				if (!LoadIndex())
					return false;
			}
			return true;
		}

		Size ToSize(const Shape& shape) const
		{
			return Size(shape[3], shape[2]);
		}

		bool BadRatio(const Size & netSize, const Size& imgSize) const
		{
			float netRatio = float(netSize.x) / float(netSize.y);
			float imgRatio = float(imgSize.x) / float(imgSize.y);
			float ratioVariation = Synet::Max(netRatio, imgRatio) / Synet::Min(netRatio, imgRatio) - 1.0f;
			return ratioVariation > _options.ratioVariation;
		}

		virtual bool PerformBatch(size_t thread, size_t current, size_t batch)
		{
			if (_options.batchSize != 1)
			{
				std::cout << "Batch size can be only 1 for detection tests!" << std::endl;
				return false;
			}

			Thread& t = _threads[thread];
			Size netSize = ToSize(t.input[0].Shape());
			Test & test = _tests[current];
			Size imgSize;
			if (!SetInput(test.path, t.input[0], 0, &imgSize))
				return false;

			test.skip = BadRatio(netSize, imgSize);
			if (test.skip)
				return true;

			for (int i = 0; i < _options.repeatNumber; ++i)
				t.output = t.network->Predict(t.input);
			test.current = t.network->GetRegions(imgSize, _options.thresholdConfidence, _options.thresholdOverlap);
			if (_options.generateIndex)
				test.control = test.current;

			return true;
		}

		void Annotate(const Region& region, size_t index, uint32_t color, int width, const Simd::Font * font, View& image)
		{
			ptrdiff_t l = ptrdiff_t(region.x - region.w / 2);
			ptrdiff_t t = ptrdiff_t(region.y - region.h / 2);
			ptrdiff_t r = ptrdiff_t(region.x + region.w / 2);
			ptrdiff_t b = ptrdiff_t(region.y + region.h / 2);
			Simd::DrawRectangle(image, l, t, r, b, color, width);
			if(font)
				font->Draw(image, std::to_string(index), Size(l, t), color);
		}

		bool Annotate(const Test& test, const Simd::Font& font)
		{
			View image;
			if (!LoadImage(test.path, image))
			{
				std::cout << "Can't read '" << test.path << "' image!" << std::endl;
				return false;
			}
			for (size_t j = 0; j < test.control.size(); ++j)
				Annotate(test.control[j], j, 0xFFFF0000, 2, NULL, image);
			for (size_t j = 0; j < test.current.size(); ++j)
				Annotate(test.current[j], j, 0xFF00FF00, 1, &font, image);
			String path = MakePath(_options.outputDirectory, GetNameByPath(test.name));
			if (!SaveImage(image, path))
			{
				std::cout << "Can't write '" << path << "' image!" << std::endl;
				return false;
			}
			return true;
		}

		String PrintResume(size_t number, double precision, double error, double threshold)
		{
			std::stringstream ss;
			ss << "Number: " << number << ", precision: " << ToString(precision * 100, 2);
			ss << " %, error: " << ToString(error * 100, 2) << " %, threshold: " << ToString(threshold, 3);
			return ss.str();
		}

		virtual bool ProcessResult()
		{
			if (_options.generateIndex)
			{
				if (!SaveIndex())
					return false;
			}
			Simd::Font font(20);
			SaveListFile();
			typedef std::pair<float, int> Pair;
			std::vector<Pair> detections;
			size_t total = 0;
			for (size_t i = 0; i < _tests.size(); ++i)
			{
				const Test& test = _tests[i];
				total += test.control.size();
				for (size_t j = 0; j < test.current.size(); ++j)
				{
					int type = -1;
					for (size_t k = 0; k < test.control.size(); ++k)
					{
						float overlap = Synet::Overlap(test.current[j], test.control[k]);
						if (overlap > _options.thresholdOverlap && test.current[j].id == test.control[k].id)
							type = 1;
					}
					detections.push_back(Pair(test.current[j].prob, type));
				}
				if(_options.annotateRegions)
					Annotate(test, font);
			}
			std::sort(detections.begin(), detections.end(), [](const Pair& a, const Pair& b) {return a.first > b.first; });
			int idx = 0, max = 0, pos = 0, neg = 0, posMax = 0, negMax = 0;
			for (size_t i = 0; i < detections.size(); ++i)
			{
				switch (detections[i].second)
				{
				case -1: 
					neg++;
					break;
				case 1:
					pos++;
					break;
				}
				if (pos - neg > max)
				{
					max = pos - neg;
					idx = (int)i;
					posMax = pos;
					negMax = neg;
				}
			}
			double threshold;
			if (detections.size())
			{
				threshold = detections[idx].first / 2.0f;
				if (idx < detections.size() - 1)
					threshold += detections[idx + 1].first / 2.0f;
				else
					threshold += _options.thresholdConfidence / 2.0f;
			}
			else
				threshold = _options.thresholdConfidence;
			_options.resume = PrintResume(total, double(posMax) / total, double(negMax) / total, threshold);
			return true;
		}
	};
}



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
#include "TestUtils.h"
#include "TestImage.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Test
{
	bool LoadImage(const String& path, View& view)
	{
		String ext = ToLower(ExtensionByPath(path));
		if (ext == "ppm" || ext == "pgm")
			return view.Load(path);
		else
		{
			bool result = false;
			int x, y, c;
			stbi_uc* data = stbi_load(path.c_str(), &x, &y, &c, STBI_rgb);
			if (data)
			{
				if (c == 1)
				{
					view.Recreate(x, y, View::Gray8);
					Simd::Convert(View(x, y, x * c, View::Gray8, data), view);
					result = true;
				}
				if (c == 3)
				{
					view.Recreate(x, y, View::Bgra32);
					Simd::Convert(View(x, y, x * c, View::Bgr24, data), view);
					result = true;
				}
				if (c == 4)
				{
					view.Recreate(x, y, View::Bgra32);
					Simd::Convert(View(x, y, x * c, View::Bgra32, data), view);
					result = true;
				}
				stbi_image_free(data);
			}
			return result;
		}
	}

	bool SaveImage(const View& view, const String& path)
	{
		String ext = ToLower(ExtensionByPath(path));
		if (ext == "ppm" || ext == "pgm")
			return view.Save(path);
		else if (ext == "jpg" || ext == "jpeg")
		{
			View bgr;
			if (view.format == View::Bgr24)
				bgr = view;
			else
			{
				bgr.Recreate(view.width, view.height, View::Bgr24, NULL, 1);
				Simd::Convert(view, bgr);
			}
			return stbi_write_jpg(path.c_str(), (int)bgr.width, (int)bgr.height, 3, bgr.data, 100) != 0;
		}
		else if (ext == "png")
			return stbi_write_png(path.c_str(), (int)view.width, (int)view.height, (int)view.ChannelCount(), view.data, (int)view.stride) != 0;
		else
			return false;
	}
}



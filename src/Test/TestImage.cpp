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
#include "TestUtils.h"
#include "TestImage.h"

#ifndef SYNET_TEST_STB_EXTERNAL
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#ifndef SYNET_TEST_STB_EXTERNAL
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"

namespace Test
{
	bool LoadImage(const String& path, View& view)
	{
		return view.Load(path);
	}

	bool SaveImage(const View& view, const String& path)

	{
		String ext = ToLower(ExtensionByPath(path));
		if (ext == "pgm")
			return view.Save(path, SimdImageFilePgmBin);
		else if (ext == "ppm")
			return view.Save(path, SimdImageFilePpmBin);
		else if (ext == "png")
			return view.Save(path, SimdImageFilePng);
		else if (ext == "jpg" || ext == "jpeg")
			return view.Save(path, SimdImageFileJpeg, 95);
		else
			return false;
	}
}



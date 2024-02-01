/*
* Use samples for Synet Framework (http://github.com/ermig1979/Synet).
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
#ifndef SYNET_SIMD_LIBRARY_ENABLE
#define SYNET_SIMD_LIBRARY_ENABLE
#endif
#include "Synet/Network.h"
#include "Synet/Converters/InferenceEngine.h"
#include "Simd/SimdDrawing.hpp"

typedef Synet::Network Net;
typedef Synet::View View;
typedef Synet::Shape Shape;
typedef Synet::Region<float> Region;
typedef std::vector<Region> Regions;

int main(int argc, char* argv[])
{
    Synet::ConvertInferenceEngineToSynet("ie_fd.xml", "ie_fd.bin", 
		true, "synet.xml", "synet.bin");

    Net net;
    net.Load("synet.xml", "synet.bin");

    net.Reshape(256, 256, 1);

    Shape shape = net.NchwShape();

    View original;
    original.Load("faces_0.ppm");

    View resized(shape[3], shape[2], original.format);
    Simd::Resize(original, resized, ::SimdResizeMethodArea);

    net.SetInput(resized, 0.0f, 255.0f);

    net.Forward();

    Regions faces = net.GetRegions(original.width, original.height, 0.5f, 0.5f);
    uint32_t white = 0xFFFFFFFF;
    for (size_t i = 0; i < faces.size(); ++i)
    {
        const Region & face = faces[i];
        ptrdiff_t l = ptrdiff_t(face.x - face.w / 2);
        ptrdiff_t t = ptrdiff_t(face.y - face.h / 2);
        ptrdiff_t r = ptrdiff_t(face.x + face.w / 2);
        ptrdiff_t b = ptrdiff_t(face.y + face.h / 2);
        Simd::DrawRectangle(original, l, t, r, b, white);
    }
    original.Save("annotated_faces_0.ppm");

    return 0;
}
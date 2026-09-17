/*
* Use samples for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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
#include "Synet/Decoders/Detection.h"
#include "Simd/SimdDrawing.hpp"

typedef Synet::Network Net;
typedef Synet::View View;
typedef Synet::Shape Shape;
typedef Synet::Region<float> Region;
typedef std::vector<Region> Regions;
typedef Synet::DetOutDecoder Decoder;

int main(int argc, char* argv[])
{
    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    CPL_LOG_SS(Info, "Synet face detection example:");

    Net net;
    if (!net.Load("face_detector.xml", "face_detector.bin"))
        SYNET_ERROR("Can't load model files: face_detector.xml and face_detector.bin !");

    Shape shape = net.NchwShape();

    View original;
    if (!original.Load("faces.jpg", View::Bgra32))
        SYNET_ERROR("Can't load test image: faces.jpg !");

    View resized(shape[3], shape[2], original.format);
    Simd::Resize(original, resized, ::SimdResizeMethodArea);

    net.SetInput(resized, 0.0f, 255.0f);

    net.Forward();

    Decoder decoder;
    Regions faces = decoder.GetRegions(net, original.width, original.height, 0.5f, 0.5f)[0];
    uint32_t color = 0xFF00FF00;
    for (size_t i = 0; i < faces.size(); ++i)
    {
        const Region & face = faces[i];
        ptrdiff_t l = ptrdiff_t(face.x - face.w / 2);
        ptrdiff_t t = ptrdiff_t(face.y - face.h / 2);
        ptrdiff_t r = ptrdiff_t(face.x + face.w / 2);
        ptrdiff_t b = ptrdiff_t(face.y + face.h / 2);
        Simd::DrawRectangle(original, l, t, r, b, color, 2);
    }

    if(!original.Save("annotated_faces.jpg"))
        SYNET_ERROR("Can't save annotated image: annotated_faces.jpg !");

    CPL_LOG_SS(Info, "Annotated image saved to 'annotated_faces.jpg'.");

    return 0;
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"

namespace Synet
{
    template<class T> bool LoadBinaryData(const String& path, std::vector<T> & data)
    {
        std::ifstream ifs(path.c_str(), std::ofstream::binary);
        if (!ifs.is_open())
            return false;
        size_t beg = ifs.tellg();
        ifs.seekg(0, std::ios::end);
        size_t end = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        size_t size = (end - beg) / sizeof(T);
        data.resize(size);
        ifs.read((char*)data.data(), size * sizeof(T));
        ifs.close();
        return true;
    }

    template<class T> bool SaveBinaryData(const std::vector<T> & data, const String& path)
    {
        std::ofstream ofs(path.c_str(), std::ofstream::binary);
        if (!ofs.is_open())
            return false;
        ofs.write((const char*)data.data(), data.size() * sizeof(T));
        bool result = (bool)ofs;
        ofs.close();
        return result;
    }
}
/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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
    struct VersionInfo
    {
        int major, minor, release;
        String original, date, branch, revision;

        VersionInfo(const String& version)
            : original(version)
            , major(0)
            , minor(0)
            , release(0)
        {
            size_t beg = 0, end = 0;
            end = original.find('.', beg);
            if (end == String::npos)
                return;
            Cpl::ToVal(version.substr(beg, end - beg), major);
            beg = end + 1;
            end = original.find('.', beg);
            if (end == String::npos)
                return;
            Cpl::ToVal(version.substr(beg, end - beg), minor);
            beg = end + 1;
            end = original.find('.', beg);
            if (end == String::npos)
                return;
            Cpl::ToVal(version.substr(beg, end - beg), release);
            beg = end + 1;
            end = original.find('.', beg);
            if (end == String::npos)
                return;
            date = version.substr(beg, end - beg);
            beg = end + 1;
            end = original.find('-', beg);
            if (end == String::npos)
                return;
           branch = version.substr(beg, end - beg);
           beg = end + 1;
           end = version.length();
           revision = version.substr(beg, end - beg);
        }

        bool LessThan(const VersionInfo& other) const
        {
            if (major < other.major)
                return true;
            if (major == other.major && minor < other.minor)
                return true;
            if (major == other.major && minor == other.minor && release < other.release)
                return true;
            return false;
        }
    };
}
/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#endif

#ifdef __linux__
#include <unistd.h>
#include <dirent.h>
#endif

namespace Test
{
    template <class T> inline String ToString(const T & val)
    {
        std::stringstream ss;
        ss << val;
        return ss.str();
    }

    inline String ToString(double value, int precision)
    {
        std::stringstream ss;
        ss << std::setprecision(precision) << std::fixed << value;
        return ss.str();
    }

    template <class T> inline T FromString(const String & str)
    {
        std::stringstream ss(str);
        T t;
        ss >> t;
        return t;
    }

    inline String MakePath(const String & a, const String & b)
    {
#ifdef _MSC_VER
        return a + (a[a.size() - 1] == '\\' ? "" : "\\") + b;
#else
        return a + (a[a.size() - 1] == '/' ? "" : "/") + b;
#endif
    }

    bool CreatePath(const String & path)
    {
#ifdef _MSC_VER
        return std::tr2::sys::create_directories(std::tr2::sys::path(path));
#else
        return std::system((String("mkdir -p ") + path).c_str()) == 0;
#endif
    }

    inline bool FileExists(const String & path)
    {
#ifdef _MSC_VER
        DWORD fileAttribute = ::GetFileAttributes(path.c_str());
        return (fileAttribute != INVALID_FILE_ATTRIBUTES);
#else
        return (::access(path.c_str(), F_OK) != -1);
#endif	//_MSC_VER
    }

    inline String GetNameByPath(const String & path_)
    {
#ifdef _MSC_VER
        std::tr2::sys::path path(path_);

#if _MSC_VER >= 1900
        return path.filename().string();
#else
        return path.filename();
#endif	//_MSC_VER >= 1900

#elif defined(__unix__)
        size_t pos = path_.find_last_of("/");
        if (pos == std::string::npos)
            return path_;
        else
            return path_.substr(pos + 1);
#else
        std::cerr << "GetNameByPath: Is not implemented yet!\n";
        return "";
#endif
    }

    inline bool DirectoryExists(const String & path)
    {
#ifdef _MSC_VER
        DWORD fileAttribute = GetFileAttributes(path.c_str());
        return ((fileAttribute != INVALID_FILE_ATTRIBUTES) &&
            (fileAttribute & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
        DIR * dir = opendir(path.c_str());
        if (dir != NULL)
        {
            ::closedir(dir);
            return true;
        }
        else
            return false;
#endif	//_MSC_VER
    }

    inline StringList GetFileList(const String & directory, const String & filter, bool files, bool directories)
    {
        std::list<String> names;
#ifdef _MSC_VER
        ::WIN32_FIND_DATA fd;
        ::HANDLE hFind = ::FindFirstFile(MakePath(directory, filter).c_str(), &fd);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                String name = fd.cFileName;
                if (files && !(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                    names.push_back(fd.cFileName);
                if (directories && (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && name != "." && name != "..")
                    names.push_back(name);
            } while (::FindNextFile(hFind, &fd));
            ::FindClose(hFind);
        }
#else
        DIR * dir = ::opendir(directory.c_str());
        if (dir != NULL)
        {
            struct dirent * drnt;
            while ((drnt = ::readdir(dir)) != NULL)
            {
                String name = drnt->d_name;
                if (name == "." || name == "..")
                    continue;
                if (files && drnt->d_type != DT_DIR)
                    names.push_back(String(drnt->d_name));
                if (directories && drnt->d_type == DT_DIR)
                    names.push_back(String(drnt->d_name));
            }
            ::closedir(dir);
        }
        else
            std::cout << "There is an error during (" << errno << ") opening '" << directory << "' !" << std::endl;
#endif
        return names;
    }
}

#ifdef _MSC_VER
inline int setenv(const char * name, const char * value, int overwrite)
{
    int errcode = 0;
    if (!overwrite) {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}
#endif


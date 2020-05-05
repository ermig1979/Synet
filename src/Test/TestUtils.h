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

#include <locale>

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
    template <class T> inline String ToString(const T& val)
    {
        std::stringstream ss;
        ss << val;
        return ss.str();
    }

    inline String ToString(int value, int width)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(width) << value;
        return ss.str();
    }

    inline String ToString(double value, int precision)
    {
        std::stringstream ss;
        ss << std::setprecision(precision) << std::fixed << value;
        return ss.str();
    }

    template <class T> inline T FromString(const String& str)
    {
        std::stringstream ss(str);
        T t;
        ss >> t;
        return t;
    }

    inline String ExpandLeft(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        ss << value;
        return ss.str();
    }

    inline String ExpandRight(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        ss << value;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    inline String ExpandBoth(const String& value, size_t count)
    {
        count = std::max(count, value.size());
        std::stringstream ss;
        for (size_t i = 0, left = (count - value.size()) / 2; i < left; i++)
            ss << " ";
        ss << value;
        for (size_t i = ss.str().size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    inline String MakePath(const String& a, const String& b)
    {
#ifdef _MSC_VER
        return a + (a[a.size() - 1] == '\\' ? "" : "\\") + b;
#else
        return a + (a[a.size() - 1] == '/' ? "" : "/") + b;
#endif
    }

    inline bool CreatePath(const String& path)
    {
#if defined(_MSC_VER) && _MSC_VER < 1900
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
#if _MSC_VER < 1900
        std::tr2::sys::path path(path_);
        return path.filename().string();
#else
        std::filesystem::path path(path_);
        return path.filename().string();
#endif
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

    inline String DirectoryByPath(const String & path)
    {
#ifdef WIN32
        size_t pos = path.find_last_of("\\");
        if (pos == std::string::npos)
            return path;
        else
            return path.substr(0, pos);
#elif defined(__unix__)
        size_t pos = path.find_last_of("/");
        if (pos == std::string::npos)
            return path;
        else
            return path.substr(0, pos);
#else
        std::cerr << "GetNameByPath: Is not implemented yet!\n";
        return "";
#endif
    }

    inline String ExtensionByPath(const String& path)
    {
        String name = GetNameByPath(path);
        size_t pos = path.find_last_of(".");
        if (pos == std::string::npos)
            return String();
        else
            return path.substr(pos + 1);
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

    inline bool CreateOutputDirectory(const String& path)
    {
        String directory = DirectoryByPath(path);
        if (directory != "" && directory != "." && directory != path)
        {
            if (!DirectoryExists(directory) && !CreatePath(directory))
            {
                std::cout << "Can't create output directory '" << directory << "' !" << std::endl;
                return false;
            }
        }
        return true;
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

    inline void SortDetectionOutput(float * data, size_t size)
    {
        struct Tmp { float val[7]; };
        std::vector<Tmp> tmp(size / 7);
        for(size_t i = 0; i < tmp.size(); ++i)
            for (size_t j = 0; j < 7; ++j)
                tmp[i].val[j] = data[i*7 + j];
        std::sort(tmp.begin(), tmp.end(), [](const Tmp & t1, const Tmp & t2) 
        {
            const float thr = 0.01f;
            if (abs(t1.val[2] - t2.val[2]) > thr)
                return t1.val[2] > t2.val[2];
            else if (abs(t1.val[3] - t2.val[3]) > thr)
                return t1.val[3] > t2.val[3];
            else
                return t1.val[4] > t2.val[4];
        });
        for (size_t i = 0; i < tmp.size(); ++i)
            for (size_t j = 0; j < 7; ++j)
                data[i*7 + j] = tmp[i].val[j];
    }

    inline String ToLower(const String& src)
    {
        std::locale loc;
        String dst(src);
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] = std::tolower(dst[i], loc);
        return dst;
    }

    inline String WithoutSymbol(const String& src, char symbol = ' ')
    {
        String dst;
        dst.reserve(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            if(src[i] != symbol)
                dst.push_back(src[i]);
        return dst;
    }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
    inline String CurrentDateTimeString()
    {
        std::time_t t;
        std::time(&t);
        std::tm* tm = ::localtime(&t);
        std::stringstream ss;
        ss << ToString(tm->tm_year + 1900, 4) << "."
            << ToString(tm->tm_mon + 1, 2) << "."
            << ToString(tm->tm_mday, 2) << " "
            << ToString(tm->tm_hour, 2) << ":"
            << ToString(tm->tm_min, 2) << ":"
            << ToString(tm->tm_sec, 2);
        return ss.str();
    }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    inline String SystemInfo()
    {
        String cpu = "Unknown", mem = "Unknown";
#ifdef __linux__
        ::FILE* c = ::popen("lscpu | grep 'Model name:' | sed -r 's/Model name:\\s{1,}//g'", "r");
        if (c)
        {
            char buf[PATH_MAX];
            while (::fgets(buf, PATH_MAX, c));
            cpu = buf;
            cpu = cpu.substr(0, cpu.find('\n'));
            ::pclose(c);
        }
        ::FILE* m = ::popen("grep MemTotal /proc/meminfo | awk '{print $2 / 1024}'", "r");
        if (m)
        {
            char buf[PATH_MAX];
            while (::fgets(buf, PATH_MAX, m));
            mem = buf;
            mem = mem.substr(0, mem.find('\n'));
            ::pclose(m);
        }
#endif
        return String("CPU: ") + cpu + ", Memory: " + mem + " MB";
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


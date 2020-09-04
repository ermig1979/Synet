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

#include "TestCommon.h"

namespace Test
{
    struct ArgsParser
    {
        ArgsParser(int argc, char* argv[])
            : _argc(argc)
            , _argv(argv)
        {
        }

    protected:
        String GetArg(const String& name, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name }, { default_ }, exit)[0];
        }

        String GetArg2(const String& name1, const String& name2, const String& default_ = String(), bool exit = true)
        {
            return GetArgs({ name1, name2 }, { default_ }, exit)[0];
        }

        Strings GetArgs(const String& name, const Strings& defaults = Strings(), bool exit = true)
        {
            return GetArgs(Strings({ name }), defaults, exit);
        }

        Strings GetArgs(const Strings& names, const Strings& defaults = Strings(), bool exit = true)
        {
            Strings values;
            for (int a = 1; a < _argc; ++a)
            {
                String arg = _argv[a];
                for (size_t n = 0; n < names.size(); ++n)
                {
                    const String& name = names[n];
                    if (arg.substr(0, name.size()) == name && arg.substr(name.size(), 1) == "=")
                        values.push_back(arg.substr(name.size() + 1));
                }
            }
            if (values.empty())
            {
                if (defaults.empty() && exit)
                {
                    std::cout << "Argument '";
                    for (size_t n = 0; n < names.size(); ++n)
                        std::cout << (n ? " | " : "") << names[n];
                    std::cout << "' is absent!" << std::endl;
                    ::exit(1);
                }
                else
                    return defaults;
            }
            return values;
        }

        String AppName() const
        {
            return _argv[0];
        }

    private:
        int _argc;
        char** _argv;
    };
}


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

#include "TestUtils.h"

namespace Test
{
    struct Html
    {
        struct Attribute
        {
            String name, value;
            Attribute(const String& n = String(), const String& v = String())
                : name(n)
                , value(v)
            {
            }
        };
        typedef std::vector<Attribute> Attributes;

        static Attributes Attr()
        {
            return Attributes();
        }

        static Attributes Attr(
            const String& name0, const String& value0)
        {
            Attributes attrbutes;
            attrbutes.push_back(Attribute(name0, value0));
            return attrbutes;
        }

        static Attributes Attr(
            const String& name0, const String& value0,
            const String& name1, const String& value1)
        {
            Attributes attrbutes;
            attrbutes.push_back(Attribute(name0, value0));
            attrbutes.push_back(Attribute(name1, value1));
            return attrbutes;
        }

        static Attributes Attr(
            const String& name0, const String& value0,
            const String& name1, const String& value1,
            const String& name2, const String& value2)
        {
            Attributes attrbutes;
            attrbutes.push_back(Attribute(name0, value0));
            attrbutes.push_back(Attribute(name1, value1));
            attrbutes.push_back(Attribute(name2, value2));
            return attrbutes;
        }

        Html(std::ostream& stream, size_t indent = 0)
            : _stream(stream)
            , _indent(indent)
            , _line(true)
        {
        }

        void WriteIndent()
        {
            const String INDENT = "  ";
            for (size_t i = 0; i < _indent; ++i)
                _stream << INDENT;
        }

        void WriteAtribute(const Attribute& attribute)
        {
            _stream << " " << attribute.name << "=\"" << attribute.value << "\"";
        }

        void WriteBegin(const String& name, const Attributes& attributes, bool indent, bool line)
        {
            if (_line)
                WriteIndent();
            _stream << "<" << name;
            for (size_t i = 0; i < attributes.size(); ++i)
                WriteAtribute(attributes[i]);
            _stream << ">";
            if (line)
                _stream << std::endl;
            if (indent)
                _indent++;
            _line = line;
        }

        void WriteEnd(const String& name, bool indent, bool line)
        {
            if (indent)
            {
                _indent--;
                if (_line)
                    WriteIndent();
            }
            _stream << "</" << name << ">";
            if (line)
                _stream << std::endl;
            _line = line;
        }

        void WriteValue(const String& name, const Attributes& attributes, const String& value, bool line)
        {
            WriteBegin(name, attributes, false, false);
            _stream << value;
            WriteEnd(name, false, line);
        }

        void WriteText(const String& text, bool indent, bool line)
        {
            if (indent && _line)
                WriteIndent();
            _stream << text;
            if (line)
                _stream << std::endl;
            _line = line;
        }

        size_t Indent() const
        {
            return _indent;
        }

    private:
        std::ostream& _stream;
        size_t _indent;
        bool _line;
    };
}



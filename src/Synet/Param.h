/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"

namespace Synet
{
    template<class T> struct Param
    {
        typedef T Type;

        Param()
        {
            _value = Default();
        }        
        
        const Type & operator () () const { return _value; }
        Type & operator () () { return _value; }

        virtual bool Node() const { return false; }
        virtual Type Default() const { return Type(); }
        virtual String ToString() const { return ""; }
        virtual String Name() const = 0;
        virtual size_t Size() const = 0;

        template <class Saver> bool Save(Saver & saver) const
        {
            if (Node())
            {
                for (const Param * child = Begin(); child < End(); child = Next(child))
                {
                    saver.WriteBegin(child->Name(), child->Node());
                    if (!child->Save(saver))
                        return false;
                    saver.WriteEnd(child->Name(), child->Node());
                }
            }
            else
            {
                saver.WriteValue(ToString());
            }
            return true;
        }

        template <class Loader> bool Load(Loader & loader)
        {
            if (Node())
            {
                for (Param * child = Begin(); child < End(); child = Next(child))
                {
                    if (!child->Load(loader))
                        return false;
                }
            }
            else
                return loader.Load(Name(), _value, Default());
            return true;
        }

    protected:
        Type _value;

        Param * Begin() const { return (Param*)(&_value); }
        Param * Next(const Param * param) const { return (Param*)((char*)param + param->Size()); }
        const Param * End() const { return (Param *)((char*)this + Size()); }
    };

    template<class T> String ToString(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

#define SYNET_ELEM_PARAM(type, name, value) \
struct __Param_##name : public Synet::Param<type> \
{ \
__Param_##name() { _value = Default(); } \
virtual type Default() const { return value; } \
virtual Synet::String Name() const { return #name; } \
virtual size_t Size() const { return sizeof(__Param_##name); } \
virtual Synet::String ToString() const { return Synet::ToString((*this)()); } \
} name;

#define SYNET_NODE_PARAM(type, name) \
struct __Param_##name : public Synet::Param<type> \
{ \
virtual Synet::String Name() const { return #name; } \
virtual size_t Size() const { return sizeof(__Param_##name); } \
virtual bool Node() const { return true; } \
} name;

#define SYNET_ROOT_CLASS(type, name) \
struct name : public Synet::Param<type> \
{ \
virtual Synet::String Name() const { return #name; } \
virtual size_t Size() const { return sizeof(name); } \
virtual bool Node() const { return true; } \
};

namespace Synet
{
    struct XmlSaver
    {
        XmlSaver()
            : _os(std::cout)
            , _indent(0)
        {
            WriteRoot();
        }

        bool WriteBegin(const Synet::String & name, bool node)
        {
            WriteIndent();
            _os << "<" << name << ">";
            if (node)
            {
                _os << std::endl;
                _indent += INDENT;
            }
            return true;
        }

        bool WriteValue(const Synet::String & value)
        {
            _os << value;
            return true;
        }

        bool WriteEnd(const Synet::String & name, bool node)
        {
            if (node)
            {
                _indent -= INDENT;
                WriteIndent();
            }
            _os << "</" << name << ">" << std::endl;
            return true;
        }

    private:
        void WriteRoot()
        {
            _os << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" << std::endl;
        }

        void WriteIndent()
        {
            for (size_t i = 0; i < _indent; ++i)
                _os << " ";
        }

        const size_t INDENT = 2;
        std::ostream & _os;
        size_t _indent;
    };
}
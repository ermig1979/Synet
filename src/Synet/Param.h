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
        
        const Type & operator () () const { return _value; }
        Type & operator () ()  { return _value; }
        const String & Name() const { return _name; }

        virtual Type Default() const { return T(); }

        template <class Saver> bool Save(Saver & saver, bool full) const
        {
            saver.WriteBegin(Name(), _mode != Value);
            switch (_mode)
            {
            case Value:
                saver.WriteValue(ToString());
                break;
            case Struct:
                for (const Param * child = StructBegin(); child < StructEnd(); child = StructNext(child))
                {
                    if (full || child->Changed())
                    {
                        if (!child->Save(saver, full))
                            return false;
                    }
                }
                break;
            case Vector:
                for (const Param * item = VectorBegin(); item < VectorEnd(); item = VectorNext(item))
                {
                    saver.WriteBegin(ItemName(), _mode != Value);
                    for (const Param * child = item, *childEnd = VectorNext(item); child < childEnd; child = StructNext(child))
                    {
                        if (full || child->Changed())
                        {
                            if (!child->Save(saver, full))
                                return false;
                        }
                    }
                    saver.WriteEnd(ItemName(), _mode != Value);
                }
                break;
            }
            saver.WriteEnd(Name(), _mode != Value);
            return true;
        }

        template <class Loader> bool Load(Loader & loader)
        {
            //if (Node())
            //{
            //    for (Param * child = Begin(); child < End(); child = Next(child))
            //    {
            //        if (!child->Load(loader))
            //            return false;
            //    }
            //}
            //else
            //    return loader.Load(Name(), _value, Default());
            return true;
        }

    protected:
        enum Mode
        {
            Value,
            Struct,
            Vector,
        };

        Mode _mode;
        String _name;
        Type _value;

        Param(Mode mode, const String & name)
            : _mode(mode)
            , _name(name)
        {
        }
        
        virtual size_t TotalSize() const = 0;
        virtual size_t ChildSize() const { return 0; }
        virtual String ToString() const { return ""; }
        virtual String & ItemName() const { return "item"; }

        Param * StructBegin() const { return (Param*)(&_value); }
        Param * StructNext(const Param * param) const { return (Param*)((char*)param + param->TotalSize()); }
        Param * StructEnd() const { return (Param *)((char*)this + TotalSize()); }

        Param * VectorBegin() const { return (*(std::vector<Param>*)&_value).data(); }
        Param * VectorNext(const Param * param) const { return (Param*)((char*)param + ChildSize()); }
        Param * VectorEnd() const { return (*(std::vector<Param>*)&_value).data() + (*(std::vector<Param>*)&_value).size(); }

        virtual bool Changed() const
        {
            for (const Param * child = StructBegin(); child < StructEnd(); child = StructNext(child))
            {
                if (child->Changed())
                    return true;
            }
            return false;
        }
    };

    template<class T> String ToString(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template<> inline String ToString<std::vector<int>>(const std::vector<int> & values)
    {
        std::stringstream ss;
        for (size_t i = 0; i < values.size(); ++i)
            ss << (i ? " " : "") << values[i];
        return ss.str();
    }
}

#define SYNET_PARAM_VALUE(type, name, value) \
struct Param_##name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
Param_##name() : Base(Base::Value, #name) { _value = Default(); } \
virtual type Default() const { return value; } \
virtual size_t TotalSize() const { return sizeof(Param_##name); } \
virtual Synet::String ToString() const { return Synet::ToString((*this)()); } \
virtual bool Changed() const { return Default() != _value; } \
} name;

#define SYNET_PARAM_STRUCT(type, name) \
struct Param_##name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
Param_##name() : Base(Base::Struct, #name) {} \
virtual size_t TotalSize() const { return sizeof(Param_##name); } \
} name;

#define SYNET_PARAM_VECTOR(type, name) \
struct Param_##name : public Synet::Param<std::vector<type>> \
{ \
typedef Synet::Param<std::vector<type>> Base; \
Param_##name() : Base(Base::Vector, #name) {} \
virtual size_t TotalSize() const { return sizeof(Param_##name); } \
virtual size_t ChildSize() const { return sizeof(type); } \
virtual bool Changed() const { return !_value.empty(); } \
} name;

#define SYNET_PARAM_ROOT(type, name) \
struct name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
name() : Base(Base::Struct, #name) {} \
virtual size_t TotalSize() const { return sizeof(name); } \
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

        XmlSaver(std::ostream & os)
            : _os(os)
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
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
#include "Synet/Xml.h"

namespace Synet
{
    template<class T> struct Param
    {
        typedef T Type;
        
        const Type & operator () () const { return _value; }
        Type & operator () ()  { return _value; }

        const String & Name() const { return _name; }

        virtual Type Default() const { return T(); }

        virtual bool Changed() const
        {
            for (const Param * child = StructBegin(); child < StructEnd(); child = StructNext(child))
            {
                if (child->Changed())
                    return true;
            }
            return false;
        }

        bool Save(std::ostream & os, bool full) const
        {
            Xml::XmlDocument<char> doc;
            this->Save(doc, &doc, full);
            Xml::Print(os, doc);
            return true;
        }

        bool Save(const String & path, bool full) const
        {
            bool result = false;
            std::ofstream ofs(path.c_str());
            if (ofs.is_open())
            {
                result = this->Save(ofs, full);
                ofs.close();
            }
            return result;
        }

        bool Load(std::istream & is)
        {
            Xml::File<char> file(is);
            Xml::XmlDocument<char> doc;
            try
            {
                doc.Parse<0>(file.Data());
            }
            catch (std::exception e)
            {
                return false;
            }
            return this->Load(&doc);
        }

        bool Load(const String & path) const
        {
            bool result = false;
            std::ifstream ifs(path.c_str());
            if (ifs.is_open())
            {
                result = this->Load(ifs);
                ifs.close();
            }
            return result;
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
        size_t _size, _item;
        Type _value;

        Param(Mode mode, const String & name, size_t size, size_t item = 0)
            : _mode(mode)
            , _name(name)
            , _size(size)
            , _item(item)
        {
        }
        
        virtual String ToString() const { return ""; }
        virtual void ToValue(const String & string) {}
        virtual void Resize(size_t size) {}
        String ItemName() const { return "item"; }

        Param * StructBegin() const { return (Param*)(&_value); }
        Param * StructNext(const Param * param) const { return (Param*)((char*)param + param->_size); }
        Param * StructEnd() const { return (Param *)((char*)this + this->_size); }

        Param * VectorBegin() const { return (*(std::vector<Param>*)&_value).data(); }
        Param * VectorNext(const Param * param) const { return (Param*)((char*)param + this->_item); }
        Param * VectorEnd() const { return (*(std::vector<Param>*)&_value).data() + (*(std::vector<Param>*)&_value).size(); }

        bool Load(Xml::XmlNode<char> * xmlParent)
        {
            Xml::XmlNode<char> * xmlCurrent = xmlParent->FirstNode(this->Name().c_str());
            if (xmlCurrent)
            {
                switch (_mode)
                {
                case Value:
                    this->FromString(xmlCurrent->Value());
                    break;
                case Struct:
                    for (Param * paramChild = this->StructBegin(); paramChild < this->StructEnd(); paramChild = this->StructNext(paramChild))
                    {
                        if (!paramChild->Load(xmlCurrent))
                            return true;
                    }
                    break;
                case Vector:
                    this->Resize(Xml::CountChildren(xmlCurrent));
                    Xml::XmlNode<char> * xmlItem = xmlCurrent->FirstNode();
                    for (Param * paramItem = this->VectorBegin(); paramItem < this->VectorEnd(); paramItem = this->VectorNext(paramItem))
                    {
                        if (ItemName() != xmlItem->Name())
                            return false;
                        const Param * paramChildEnd = this->VectorNext(paramItem);
                        for (Param * paramChild = this->StructBegin(); paramChild < paramChildEnd; paramChild = this->StructNext(paramChild))
                        {
                            if (!paramChild->Load(xmlItem))
                                return true;
                        }
                        xmlItem = xmlItem->NextSibling();
                    }
                    break;
                }
            }
            return true;
        }

        void Save(Xml::XmlDocument<char> & xmlDoc, Xml::XmlNode<char> * xmlParent, bool full) const
        {
            Xml::XmlNode<char> * xmlCurrent = xmlDoc.AllocateNode(Xml::NodeElement, xmlDoc.AllocateString(this->Name().c_str()));
            switch (_mode)
            {
            case Value:
                xmlCurrent->Value(xmlDoc.AllocateString(this->ToString().c_str()));
                break;
            case Struct:
                for (const Param * paramChild = this->StructBegin(); paramChild < this->StructEnd(); paramChild = this->StructNext(paramChild))
                {
                    if (full || paramChild->Changed())
                        paramChild->Save(xmlDoc, xmlCurrent, full);
                }
                break;
            case Vector:
                for (const Param * paramItem = this->VectorBegin(); paramItem < this->VectorEnd(); paramItem = this->VectorNext(paramItem))
                {
                    Xml::XmlNode<char> * xmlItem = xmlDoc.AllocateNode(Xml::NodeElement, xmlDoc.AllocateString(ItemName().c_str()));
                    const Param * paramChildEnd = this->VectorNext(paramItem);
                    for (const Param * paramChild = paramItem; paramChild < paramChildEnd; paramChild = this->StructNext(paramChild))
                    {
                        if (full || paramChild->Changed())
                            paramChild->Save(xmlDoc, xmlItem, full);
                    }
                    xmlCurrent->AppendNode(xmlItem);
                }
                break;
            }
            xmlParent->AppendNode(xmlCurrent);
        }
    };

    template<class T> inline String ValueToString(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template<class T> inline String ValueToString(const std::vector<T> & values)
    {
        std::stringstream ss;
        for (size_t i = 0; i < values.size(); ++i)
            ss << (i ? " " : "") << values[i];
        return ss.str();
    }

    template<class T> inline void StringToValue(const String & string, T & value)
    {
        std::stringstream ss(string);
        ss >> value;
    }

    template<class T> inline void StringToValue(const String & string, std::vector<T> & values)
    {
        std::stringstream ss(string);
        values.clear();
        while (!ss.eof())
        {
            String item;
            ss >> item;
            if (item.size())
            {
                T value;
                StringToValue(item, value);
                values.push_back(value);
            }
        }
    }
}

#define SYNET_PARAM_VALUE(type, name, value) \
struct Param_##name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
Param_##name() : Base(Base::Value, #name, sizeof(Param_##name)) { this->_value = this->Default(); } \
virtual type Default() const { return value; } \
virtual Synet::String ToString() const { using namespace Synet; return ValueToString((*this)()); } \
virtual void ToValue(const Synet::String & string) { using namespace Synet; StringToValue(string, this->_value); } \
virtual bool Changed() const { return this->Default() != this->_value; } \
} name;

#define SYNET_PARAM_STRUCT(type, name) \
struct Param_##name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
Param_##name() : Base(Base::Struct, #name, sizeof(Param_##name)) {} \
} name;

#define SYNET_PARAM_VECTOR(type, name) \
struct Param_##name : public Synet::Param<std::vector<type>> \
{ \
typedef Synet::Param<std::vector<type>> Base; \
Param_##name() : Base(Base::Vector, #name, sizeof(Param_##name), sizeof(type)) {} \
virtual void Resize(size_t size) { this->_value.resize(size); } \
virtual bool Changed() const { return !this->_value.empty(); } \
} name;

#define SYNET_PARAM_ROOT(type, name) \
struct name : public Synet::Param<type> \
{ \
typedef Synet::Param<type> Base; \
name() : Base(Base::Struct, #name, sizeof(name)) {} \
};

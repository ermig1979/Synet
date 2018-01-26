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
#include "Synet/Param.h"

#if defined(SYNET_PROTOBUF_ENABLE)

#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>

namespace google 
{
    namespace protobuf 
    {
        class LIBPROTOBUF_EXPORT Xml 
        {
        public:
            static inline void Save(const Message & message, std::ostream & os)
            {
                Printer().Save(message, os);
            }

            static void Save(const Message& message, Synet::Xml::XmlDocument<char> & doc)
            {
                Printer().Save(message, doc);
            }

            class LIBPROTOBUF_EXPORT Printer 
            {
            public:
                Printer()
                {
                }

                ~Printer()
                {

                }

                void Save(const Message & message, std::ostream & os)
                {
                    Synet::Xml::XmlDocument<char> doc;
                    MessageToDOM(message, &doc);
                    os << doc;
                }

                void Save(const Message & message, Synet::Xml::XmlDocument<char> & doc)
                {
                    Synet::Xml::XmlNode<char> * xmlDeclaration = doc.AllocateNode(Synet::Xml::NodeDeclaration);
                    xmlDeclaration->AppendAttribute(doc.AllocateAttribute("version", "1.0"));
                    xmlDeclaration->AppendAttribute(doc.AllocateAttribute("encoding", "utf-8"));
                    doc.AppendNode(xmlDeclaration);

                    Synet::Xml::XmlNode<char> * rootNode = doc.AllocateNode(Synet::Xml::NodeElement, message.GetDescriptor()->name().c_str());
                    doc.AppendNode(rootNode);

                    SaveMessage(message, doc, rootNode);
                }

            private:
                void SaveMessage(const Message & message, Synet::Xml::XmlDocument<char> & doc, Synet::Xml::XmlNode<char> * node)
                {
                    const Reflection * reflection = message.GetReflection();
                    vector<const FieldDescriptor*> fields;
                    reflection->ListFields(message, &fields);
                    for (unsigned int i = 0; i < fields.size(); i++) 
                        SaveField(message, reflection, fields[i], doc, node);
                }

                void SaveField(const Message & message, const Reflection * reflection, const FieldDescriptor * field,
                    Synet::Xml::XmlDocument<char> & doc, Synet::Xml::XmlNode<char> * node)
                {
                    int count = 0;
                    if (field->is_repeated()) 
                        count = reflection->FieldSize(message, field);
                    else if (reflection->HasField(message, field)) 
                        count = 1;
                    for (int j = 0; j < count; ++j) 
                    {
                        int fieldIndex = j;
                        if (!field->is_repeated()) 
                            fieldIndex = -1;
                        SaveValue(message, reflection, field, fieldIndex, doc, node);
                    }
                }

                Sybet::String FieldName(const Message & message, const Reflection * reflection, const FieldDescriptor * field)
                {
                    if (field->is_extension()) 
                    {
                        if (field->containing_type()->options().message_set_wire_format()
                            && field->type() == FieldDescriptor::TYPE_MESSAGE
                            && field->is_optional()
                            && field->extension_scope() == field->message_type()) 
                        {
                            return field->message_type()->full_name();
                        }
                        else 
                            return field->full_name();
                    }
                    else 
                    {
                        if (field->type() == FieldDescriptor::TYPE_GROUP) 
                            return field->message_type()->name();
                        else 
                            return field->name();
                    }
                }

                static inline const char * TrueString() { return "true"; }
                static inline const char * FalseString() { return "false"; }

                void SaveValue(const Message & message, const Reflection* reflection, const FieldDescriptor * field,
                    int fieldIndex, Synet::Xml::XmlDocument<char> & doc, Synet::Xml::XmlNode<char> * node)
                {
                    assert(field->is_repeated() || fieldIndex == -1);

                    switch (field->cpp_type())
                    {
                    case FieldDescriptor::CPPTYPE_INT32:
                    {
                        int32_t value = field->is_repeated() ? reflection->GetRepeatedInt32(message, field, fieldIndex) :  reflection->GetInt32(message, field); 
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value)); 
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_INT64:
                    {
                        int64_t value = field->is_repeated() ? reflection->GetRepeatedInt64(message, field, fieldIndex) : reflection->GetInt64(message, field);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value));
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_UINT32:
                    {
                        uint32_t value = field->is_repeated() ? reflection->GetRepeatedUInt32(message, field, fieldIndex) : reflection->GetUInt32(message, field);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value));
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_UINT64:
                    {
                        uint64_t value = field->is_repeated() ? reflection->GetRepeatedUInt64(message, field, fieldIndex) : reflection->GetUInt64(message, field);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value));
                        node->AppendNode(stringNode);
                        break; 
                    }
                    case FieldDescriptor::CPPTYPE_FLOAT:
                    {
                        float value = field->is_repeated() ? reflection->GetRepeatedFloat(message, field, fieldIndex) : reflection->GetFloat(message, field);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value));
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_DOUBLE:
                    {
                        double value = field->is_repeated() ? reflection->GetRepeatedDouble(message, field, fieldIndex) : reflection->GetDouble(message, field);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), Synet::ValueToString(value));
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_STRING:
                    {
                        Synet::String scratch;
                        const Synet::String & value = field->is_repeated() ? reflection->GetRepeatedStringReference(
                                message, field, fieldIndex, &scratch) : reflection->GetStringReference(message, field, &scratch);
                        Synet::Xml::XmlNode<char> * stringNode = doc.AllocateNode(Synet::Xml::NodeElement,
                            FieldName(message, reflection, field).c_str(), value.c_str());
                        node->AppendNode(stringNode);
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_BOOL: 
                    {
                        if (field->is_repeated())
                        {
                            Synet::Xml::XmlNode<char> * boolNode = doc.AllocateNode(Synet::Xml::NodeElement, FieldName(message, reflection, field).c_str(), 
                                reflection->GetRepeatedBool(message, field, fieldIndex) ? TrueString() : FalseString());
                            node->AppendNode(boolNode);
                        }
                        else 
                        {
                            Synet::Xml::XmlNode<char> * boolNode = doc.AllocateNode(Synet::Xml::NodeElement, FieldName(message, reflection, field).c_str(),
                                reflection->GetBool(message, field) ? TrueString() : FalseString());
                            node->AppendNode(boolNode);
                        }
                        break;
                    }
                    case FieldDescriptor::CPPTYPE_ENUM: 
                    {
                        Synet::String value = field->is_repeated() ? reflection->GetRepeatedEnum(message, field, fieldIndex)->name() :
                            reflection->GetEnum(message, field)->name();
                        Synet::Xml::XmlNode<char> * enumNode = doc.AllocateNode(Synet::Xml::NodeElement, GetXmlFieldName(message, reflection, field).c_str(), value.c_str());
                        node->AppendNode(enumNode);
                        break;
                    }                    
                    case FieldDescriptor::CPPTYPE_MESSAGE: 
                    {
                        Synet::Xml::XmlNode<char> * messageNode = doc.AllocateNode(Synet::Xml::NodeElement, field->name().c_str());
                        node->AppendNode(messageNode);
                        SaveMessage(field->is_repeated() ? reflection->GetRepeatedMessage(message, field, fieldIndex) : 
                            reflection->GetMessage(message, field), doc, messageNode);
                        break;
                    }                    
                    default:
                        break;
                    }
                }
            };
        private:
            GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Xml);
        };
    }
}

#endif//SYNET_PROTOBUF_ENABLE
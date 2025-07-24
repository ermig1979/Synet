/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"

#if defined(SYNET_ONNXRUNTIME_ENABLE)

#include "onnx/onnx.pb.h"

#include "Cvt/OnnxRuntime/Common.h"

namespace Synet
{
    inline const onnx::AttributeProto* GetAtrribute(const onnx::NodeProto& node, const String& name)
    {
        for (size_t i = 0; i < node.attribute_size(); ++i)
            if (node.attribute(i).name() == name)
                return &node.attribute(i);
        return NULL;
    }

    template<class T> inline bool ConvertAtrributeInt(const onnx::NodeProto& node, const String& name, T& value, bool optional = false, const T& defVal = T())
    {
        const onnx::AttributeProto* attribute = GetAtrribute(node, name);
        if (attribute == NULL)
        {
            if (optional)
            {
                value = defVal;
                return true;
            }
            SYNET_ERROR("Can't find attribute '" << name << "' !");
        }
        if (attribute->type() != onnx::AttributeProto_AttributeType_INT)
            SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
        value = attribute->i();
        return true;
    }

    template<class T> inline bool ConvertAtrributeInts(const onnx::NodeProto& node, const String& name, std::vector<T>& values,
        bool optional = false, const std::vector<T>& defVals = std::vector<T>())
    {
        const onnx::AttributeProto* attribute = GetAtrribute(node, name);
        if (attribute == NULL)
        {
            if (optional)
            {
                values = defVals;
                return true;
            }
            SYNET_ERROR("Can't find attribute '" << name << "' !");
        }
        if (attribute->type() != onnx::AttributeProto_AttributeType_INTS)
            SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
        values.resize(attribute->ints_size());
        for (size_t i = 0; i < attribute->ints_size(); ++i)
            values[i] = (T)attribute->ints(i);
        return true;
    }

    inline bool ConvertAtrributeFloat(const onnx::NodeProto& node, const String& name, float& value, bool optional = false, const float& defVal = float())
    {
        const onnx::AttributeProto* attribute = GetAtrribute(node, name);
        if (attribute == NULL)
        {
            if (optional)
            {
                value = defVal;
                return true;
            }
            SYNET_ERROR("Can't find attribute '" << name << "' !");
        }
        if (attribute->type() != onnx::AttributeProto_AttributeType_FLOAT)
            SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
        value = attribute->f();
        return true;
    }

    inline bool ConvertAtrributeString(const onnx::NodeProto& node, const String& name, String& value, bool optional = false, const String& defVal = String())
    {
        const onnx::AttributeProto* attribute = GetAtrribute(node, name);
        if (attribute == NULL)
        {
            if (optional)
            {
                value = defVal;
                return true;
            }
            SYNET_ERROR("Can't find attribute '" << name << "' !");
        }
        if (attribute->type() != onnx::AttributeProto_AttributeType_STRING)
            SYNET_ERROR("Attribute '" << name << "' has wrong type " << attribute->type() << " !");
        value = attribute->s();
        return true;
    }
}

#endif
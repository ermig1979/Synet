/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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
#include "Synet/Params.h"
#include "Synet/Buffer.h"
#include "Synet/Utils/Shape.h"
#include "Synet/Utils/Math.h"
#include "Synet/Utils/DebugPrint.h"

namespace Synet
{
    struct Unknown
    {
    };

    template <class T> TensorType GetTensorType();
    template <> SYNET_INLINE TensorType GetTensorType<Unknown>() { return TensorTypeUnknown; }
    template <> SYNET_INLINE TensorType GetTensorType<float>() { return TensorType32f; }
    template <> SYNET_INLINE TensorType GetTensorType<int32_t>() { return TensorType32i; }
    template <> SYNET_INLINE TensorType GetTensorType<int8_t>() { return TensorType8i; }
    template <> SYNET_INLINE TensorType GetTensorType<uint8_t>() { return TensorType8u; }
    template <> SYNET_INLINE TensorType GetTensorType<int64_t>() { return TensorType64i; }
    template <> SYNET_INLINE TensorType GetTensorType<uint64_t>() { return TensorType64u; }
    template <> SYNET_INLINE TensorType GetTensorType<bool>() { return TensorTypeBool; }
    template <> SYNET_INLINE TensorType GetTensorType<uint16_t>() { return TensorType16b; }
    template <> SYNET_INLINE TensorType GetTensorType<int16_t>() { return TensorType16f; }

    SYNET_INLINE size_t GetTensorTypeSize(TensorType type)
    {
        switch (type)
        {
        case TensorTypeUnknown: return 0;
        case TensorType32f: return 4;
        case TensorType32i: return 4;
        case TensorType8i: return 1;
        case TensorType8u: return 1;
        case TensorType64i: return 8;
        case TensorType64u: return 8;
        case TensorTypeBool: return 1;
        case TensorType16b: return 2;
        case TensorType16f: return 2;
        default: assert(0); return 0;
        }
    }

    SYNET_INLINE String ToChar(TensorType t)
    {
        const char* tts[] = { "?", "f", "i", "u", "u", "l", "l", "~", "b", "h" };
        return String(tts[int(t) + 1]);
    }

    //-------------------------------------------------------------------------------------------------

    template<class T> class Tensor
    {
    public:
        typedef T Type;

        SYNET_INLINE Tensor()
            : _buffer(std::make_shared<Buffer>())
            , _type(TensorTypeUnknown)
            , _format(TensorFormatUnknown)
            , _size(0)
            , _const(false)
        {
        }

        SYNET_INLINE Tensor(TensorType type, const Synet::Shape& shape, const TensorFormat& format = TensorFormatUnknown)
            : _type(type)
            , _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _const(false)
        {
            Resize();
        }

        template<class U> SYNET_INLINE Tensor(TensorType type, const Synet::Shape& shape, const TensorFormat& format, const U & value, const String& name = String())
            : _type(type)
            , _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _name(name)
            , _const(false)
        {
            Resize<U>(value);
        }

        SYNET_INLINE Tensor(const uint8_t * data, size_t size, TensorType type, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown, const String & name = String())
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>(data, size))
            , _type(type)
            , _format(format)
            , _name(name)
            , _const(false)
        {
            _size = Size(0, _shape.size());
            assert(_size * TypeSize() == _buffer->size);
        }

        SYNET_INLINE ~Tensor()
        {
        }

        SYNET_INLINE void Reshape(TensorType type, const Synet::Shape& shape, const TensorFormat& format = TensorFormatUnknown)
        {
            _type = type;
            _shape = shape;
            _format = format;
            Resize();
        }

        template<class U> SYNET_INLINE void Reshape(TensorType type, const Synet::Shape& shape, const TensorFormat& format, const U& value, const String& name = String())
        {
            _type = type;
            _name = name;
            _shape = shape;
            _format = format;
            Resize<U>(value);
        }

        SYNET_INLINE void Extend(TensorType type, const Synet::Shape& shape, const TensorFormat& format = TensorFormatUnknown)
        {
            if (_type == TensorTypeUnknown)
                _type = type;
            else
                assert(_type == type);
            assert(_type != TensorTypeUnknown);
            _shape = shape;
            _format = format;
            _size = Size(0, _shape.size());
            _const = false;
            size_t size = _size * TypeSize();
            if (size > _buffer->size)
                _buffer->Resize(size);
        }

        SYNET_INLINE void Clear(bool saveType = false)
        {
            if(!saveType)
                _type = TensorTypeUnknown;
            _format = TensorFormatUnknown;
            _shape.clear();
            _size = 0;
            _const = false;
#ifdef SYNET_MALLOC_DEBUG
            size_t size = _buffer->size;
            if (size > SYNET_MALLOC_TRIM_THRESHOLD)
            {
                std::cout << "Try to free " << size / 1024 / 1024 << " MB :" << std::endl;
                PrintMemoryUsage();
            }
#endif
            _buffer = std::make_shared<Buffer>();
#ifdef SYNET_MALLOC_DEBUG
            if (size > SYNET_MALLOC_TRIM_THRESHOLD)
            {
                PrintMemoryUsage();
            }
#endif
        }

        SYNET_INLINE TensorType GetType() const
        {
            return _type;
        }

        SYNET_INLINE size_t TypeSize() const
        {
            return GetTensorTypeSize(_type);
        }

        SYNET_INLINE void SetType(TensorType type)
        {
            assert(_buffer->size == 0 && _size == 0);
            _type = type;
        }

        SYNET_INLINE TensorFormat Format() const
        {
            return _format;
        }

        SYNET_INLINE void SetFormat(const TensorFormat & format)
        {
            _format = format;
        }

        SYNET_INLINE const String & Name() const
        {
            return _name;
        }

        SYNET_INLINE void SetName(const String & name)
        {
            _name = name;
        }

        SYNET_INLINE bool Const() const
        {
            return _const;
        }

        SYNET_INLINE void SetConst(bool value)
        {
            _const = value;
        }

        SYNET_INLINE const Synet::Shape & Shape() const
        {
            return _shape;
        }

        SYNET_INLINE size_t Count() const
        {
            return _shape.size();
        }

        SYNET_INLINE size_t Index(ptrdiff_t axis) const
        {
            if (axis < 0)
                axis += _shape.size();
            return axis;
        }

        SYNET_INLINE size_t Axis(ptrdiff_t axis) const
        {
            return _shape[Index(axis)];
        }

        SYNET_INLINE size_t Size(ptrdiff_t startAxis, ptrdiff_t endAxis) const
        {
            startAxis = Index(startAxis);
            endAxis = Index(endAxis);
            assert(startAxis <= endAxis && (size_t)endAxis <= _shape.size());
            size_t size = 1;
            for (ptrdiff_t axis = startAxis; axis < endAxis; ++axis)
                size *= _shape[axis];
            return size;
        }

        SYNET_INLINE size_t Size(ptrdiff_t startAxis) const
        {
            return Size(startAxis, _shape.size());
        }

        SYNET_INLINE size_t Size() const
        {
            return _size;
        }

        SYNET_INLINE size_t Offset(const Synet::Index & index) const
        {
            assert(_shape.size() == index.size());
            size_t offset = 0;
            for (size_t axis = 0; axis < _shape.size(); ++axis)
            {
                assert(_shape[axis] > 0);
                assert(index[axis] < _shape[axis]);

                offset *= _shape[axis];
                offset += index[axis];
            }
            assert(offset < _size);
            return offset;
        }

        SYNET_INLINE size_t Offset(std::initializer_list<size_t> index) const
        {
            assert(_shape.size() == index.size());
            size_t offset = 0;
            for (const size_t * s = _shape.data(), *i = index.begin(); i < index.end(); ++s, ++i)
            {
                assert(*s > 0);
                assert(*i < *s);

                offset *= *s;
                offset += *i;
            }
            assert(offset < _size);
            return offset;
        }

        SYNET_INLINE size_t RawSize() const
        {
            return _buffer->size;
        }

        SYNET_INLINE uint8_t* RawData()
        {
            return _buffer->data;
        }

        SYNET_INLINE const uint8_t* RawData() const
        {
            return _buffer->data;
        }

        template<class U> SYNET_INLINE U* Data()
        {
            assert(_type == GetTensorType<U>() || _buffer->data == NULL);
            return (U*)_buffer->data;
        }

        template<class U> SYNET_INLINE const U* Data() const
        {
            assert(_type == GetTensorType<U>() || _buffer->data == NULL);
            return (U*)_buffer->data;
        }

        template<class U> SYNET_INLINE U* Data(const Synet::Index& index)
        {
            return Data<U>() + Offset(index);
        }

        template<class U> SYNET_INLINE const U* Data(const Synet::Index& index) const
        {
            return Data<U>() + Offset(index);
        }

        template<class U> SYNET_INLINE U* Data(std::initializer_list<size_t> index)
        {
            return Data<U>() + Offset(index);
        }

        template<class U> SYNET_INLINE const U* Data(std::initializer_list<size_t> index) const
        {
            return Data<U>() + Offset(index);
        }

        SYNET_INLINE void Share(const Tensor & tensor)
        {
            _type = tensor._type;
            _shape = tensor._shape;
            _size = tensor._size;
            _format = tensor._format;
            _name = tensor._name;
            _buffer = tensor._buffer;
            _const = tensor._const;
        }

        SYNET_INLINE void ShareAs(const Tensor & tensor, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _buffer = tensor._buffer;
            _type = tensor._type;
            if(_name.empty())
                _name = tensor._name;
            _shape = shape;
            _size = Size(0, _shape.size());
            _format = format;
            _const = tensor._const;
            assert(_size * TypeSize() <= _buffer->size);
        }

        SYNET_INLINE void ShareAs(const uint8_t * data, size_t size, TensorType type, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _buffer->Share(data, size);
            _type = type;
            _shape = shape;
            _size = Size(0, _shape.size());
            _format = format;
            _const = false;
            assert(_size * TypeSize() <= _buffer->size);
        }

        SYNET_INLINE void Clone(const Tensor & tensor)
        {
            _type = tensor._type;
            _shape = tensor._shape;
            _size = tensor._size;
            _format = tensor._format;
            _name = tensor._name;
            _const = tensor._const;
            _buffer.reset(tensor._buffer->Clone());
        }

        SYNET_INLINE bool Import(const TensorParam & param)
        {
            Reshape(param.type(), param.shape(), param.format());
            switch (param.type())
            {
            case TensorType32f: memcpy(RawData(), param.f32().data(), RawSize()); break;
            case TensorType32i: memcpy(RawData(), param.i32().data(), RawSize()); break;
            case TensorType64i: memcpy(RawData(), param.i64().data(), RawSize()); break;
            case TensorType64u: memcpy(RawData(), param.u64().data(), RawSize()); break;
            default:
                SYNET_ERROR("Can't import " << Cpl::ToStr(param.type()) << " tensor!")
            }
            _const = true;
            return true;
        }

        SYNET_INLINE bool Export(TensorParam & param) const
        {
            param.type() = _type;
            param.shape() = _shape;
            switch (_type)
            {
            case TensorType32f: param.f32().resize(Data<float>(), Data<float>() + Size()); break;
            case TensorType32i: param.i32().resize(Data<int32_t>(), Data<int32_t>() + Size()); break;
            case TensorType64i: param.i64().resize(Data<int64_t>(), Data<int64_t>() + Size()); break;
            case TensorType64u: param.u64().resize(Data<uint64_t>(), Data<uint64_t>() + Size()); break;
            default:
                SYNET_ERROR("Can't export " << Cpl::ToStr(_type) << " tensor!")
            }
            return true;
        }

        SYNET_INLINE size_t MemoryUsage() const
        {
            return _buffer->Owner() ? RawSize() : 0;
        }

        SYNET_INLINE void Capture()
        {
            _buffer->Capture();
        }

        void DebugPrint(std::ostream & os, const String & name, bool weight, size_t first, size_t last, size_t precision) const
        {
            switch (_type)
            {
            case TensorType32f: DebugPrint<float>(os, *this, name, weight, first, last, precision); break;
            case TensorType32i: DebugPrint<int32_t>(os, *this, name, weight, first, last, precision); break;
            case TensorType8i: DebugPrint<int8_t>(os, *this, name, weight, first, last, precision); break;
            case TensorType8u: DebugPrint<uint8_t>(os, *this, name, weight, first, last, precision); break;
            case TensorType64i: DebugPrint<int64_t>(os, *this, name, weight, first, last, precision); break;
            case TensorType64u: DebugPrint<uint64_t>(os, *this, name, weight, first, last, precision); break;
            case TensorTypeBool: DebugPrint<bool>(os, *this, name, weight, first, last, precision); break;
            case TensorType16b: DebugPrint<uint16_t>(os, *this, name, weight, first, last, precision); break;
            }
        }

        void DebugPrint(std::ostream& os, const Synet::Shape& shape, const TensorFormat& format, const String& name, 
            bool weight, size_t first, size_t last, size_t precision) const
        {
            if (_buffer->size)
            {
                Tensor<T> tensor;
                tensor.ShareAs(*this, shape, format);
                tensor.DebugPrint(os, name, weight, first, last, precision);
            }
        }

    private:

        template <class U> static void DebugPrint(std::ostream& os, const Tensor<T> & tensor, const String& name, bool weight, size_t first, size_t last, size_t precision)
        {
            const Synet::Shape& shape = tensor.Shape();
            TensorFormat format = tensor.Format();
            if (shape.size() == 4 && format == TensorFormatNhwc)
            {
                if (weight)
                {
                    Tensor<U> trans(tensor.GetType(), Shp(shape[3], shape[2], shape[0], shape[1]), TensorFormatNchw);
                    for (size_t y = 0; y < shape[0]; ++y)
                        for (size_t x = 0; x < shape[1]; ++x)
                            for (size_t i = 0; i < shape[2]; ++i)
                                for (size_t o = 0; o < shape[3]; ++o)
                                    trans.template Data<U>({ o, i, y, x })[0] = tensor.template Data<U>({ y, x, i, o })[0];
                    std::stringstream ss;
                    ss << name << " HWIO { ";
                    for (size_t i = 0; i < shape.size(); ++i)
                        ss << shape[i] << " ";
                    ss << "} -> ";
                    trans.DebugPrint(os, ss.str(), weight, first, last, precision);
                }
                else
                {
                    Tensor<U> trans(tensor.GetType(), Shp(shape[0], shape[3], shape[1], shape[2]), TensorFormatNchw);
                    for (size_t n = 0; n < shape[0]; ++n)
                        for (size_t c = 0; c < shape[3]; ++c)
                            for (size_t y = 0; y < shape[1]; ++y)
                                for (size_t x = 0; x < shape[2]; ++x)
                                    trans.template Data<U>({ n, c, y, x })[0] = tensor.template Data<U>({ n, y, x, c })[0];
                    std::stringstream ss;
                    ss << name << " NHWC { ";
                    for (size_t i = 0; i < shape.size(); ++i)
                        ss << shape[i] << " ";
                    ss << "} -> ";
                    trans.DebugPrint(os, ss.str(), weight, first, last, precision);
                }
                return;
            }
            os << name;
            if (weight)
                os << (format == TensorFormatNchw && shape.size() == 4 ? " OIHW" : (format == TensorFormatNhwc && shape.size() == 4 ? " HWIO" : ""));
            else
                os << (format == TensorFormatNchw ? " NCHW" : (format == TensorFormatNhwc ? " NHWC" : ""));
            Synet::DebugPrint<U>(os, tensor.template Data<U>(), tensor.Shape(), String(), tensor.Const(), first, last, precision);
        }

        template<class U> SYNET_INLINE void Resize(U value)
        {
            assert(_type != TensorTypeUnknown && _type == GetTensorType<U>());
            _size = Size(0, _shape.size());
            _buffer->Resize(_size * TypeSize());
            CpuSet(_size, value, Data<U>());
            _const = false;
        }

        SYNET_INLINE void Resize()
        {
            assert(_type != TensorTypeUnknown);
            _size = Size(0, _shape.size());
            _buffer->Resize(_size * TypeSize());
            memset(_buffer->data, 0, _buffer->size);
            _const = false;
        }

        typedef Synet::Buffer<uint8_t> Buffer;
        typedef std::shared_ptr<Buffer> BufferPtr;

        Synet::String _name;
        TensorType _type;
        TensorFormat _format;
        Synet::Shape _shape;
        size_t _size;
        BufferPtr _buffer;
        bool _const;

    public:
#if defined(SYNET_TENSOR_API_OLD)
        SYNET_DEPRECATED SYNET_INLINE Tensor(const Synet::Shape& shape, const TensorFormat& format)
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _const(false)
        {
            ResizeOld();
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor(const Synet::Shape& shape, const Type& value = Type(), const TensorFormat& format = TensorFormatUnknown, const String& name = String())
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _name(name)
            , _const(false)
        {
            ResizeOld(value);
        }

        SYNET_DEPRECATED SYNET_INLINE void Reshape(const Synet::Shape& shape, const TensorFormat& format)
        {
            _shape = shape;
            _format = format;
            ResizeOld();
        }

        SYNET_DEPRECATED SYNET_INLINE void Reshape(const Synet::Shape& shape, const Type& value = Type(), const TensorFormat& format = TensorFormatUnknown, const String& name = String())
        {
            _name = name;
            _shape = shape;
            _format = format;
            ResizeOld(value);
        }

        SYNET_DEPRECATED SYNET_INLINE void Extend(const Synet::Shape& shape, const TensorFormat& format = TensorFormatUnknown)
        {
            _shape = shape;
            _format = format;
            ExtendOld();
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<float>& As32f()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32f);
            return *(Tensor<float>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<float>& As32f() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32f);
            return *(const Tensor<float>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<int32_t>& As32i()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32i);
            return *(Tensor<int32_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<int32_t>& As32i() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32i);
            return *(const Tensor<int32_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<int8_t>& As8i()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8i);
            return *(Tensor<int8_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<int8_t>& As8i() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8i);
            return *(const Tensor<int8_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<uint8_t>& As8u()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8u);
            return *(Tensor<uint8_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<uint8_t>& As8u() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8u);
            return *(const Tensor<uint8_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<int64_t>& As64i()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType64i);
            return *(Tensor<int64_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<int64_t>& As64i() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType64i);
            return *(const Tensor<int64_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<uint64_t>& As64u()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType64u);
            return *(Tensor<uint64_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<uint64_t>& As64u() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType64u);
            return *(const Tensor<uint64_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<bool>& AsBool()
        {
            assert(_type == TensorTypeUnknown || _type == TensorTypeBool);
            return *(Tensor<bool>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<bool>& AsBool() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorTypeBool);
            return *(const Tensor<bool>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE Tensor<uint16_t>& As16b()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType16b);
            return *(Tensor<uint16_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE const Tensor<uint16_t>& As16b() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType16b);
            return *(const Tensor<uint16_t>*)this;
        }

        SYNET_DEPRECATED SYNET_INLINE uint8_t* RawCpuData()
        {
            return _buffer->data;
        }

        SYNET_DEPRECATED SYNET_INLINE const uint8_t* RawCpuData() const
        {
            return _buffer->data;
        }

        SYNET_DEPRECATED SYNET_INLINE Type* CpuData()
        {
            assert(_type == GetTensorType<Type>() || _buffer->data == NULL);
            return (Type*)_buffer->data;
        }

        SYNET_DEPRECATED SYNET_INLINE const Type* CpuData() const
        {
            assert(_type == GetTensorType<Type>() || _buffer->data == NULL);
            return (const Type*)_buffer->data;
        }

        SYNET_DEPRECATED SYNET_INLINE Type* CpuData(const Synet::Index& index)
        {
            return CpuData() + Offset(index);
        }

        SYNET_DEPRECATED SYNET_INLINE const Type* CpuData(const Synet::Index& index) const
        {
            return CpuData() + Offset(index);
        }

        SYNET_DEPRECATED SYNET_INLINE Type* CpuData(std::initializer_list<size_t> index)
        {
            return CpuData() + Offset(index);
        }

        SYNET_DEPRECATED SYNET_INLINE const Type* CpuData(std::initializer_list<size_t> index) const
        {
            return CpuData() + Offset(index);
        }

    private:
        SYNET_INLINE void ResizeOld(Type value)
        {
            _type = GetTensorType<Type>();
            assert(_type != TensorTypeUnknown);
            _size = Size(0, _shape.size());
            _buffer->Resize(_size * TypeSize());
            CpuSet(_size, value, Data<Type>());
            _const = false;
        }

        SYNET_INLINE void ResizeOld()
        {
            _type = GetTensorType<Type>();
            assert(_type != TensorTypeUnknown);
            _size = Size(0, _shape.size());
            _buffer->Resize(_size * TypeSize());
            memset(_buffer->data, 0, _buffer->size);
            _const = false;
        }

        SYNET_INLINE void ExtendOld()
        {
            if (_type == TensorTypeUnknown)
                _type = GetTensorType<Type>();
            assert(_type != TensorTypeUnknown && _type == GetTensorType<Type>());
            _size = Size(0, _shape.size());
            size_t size = _size * TypeSize();
            if (size > _buffer->size)
                _buffer->Resize(size);
            _const = false;
        }
#endif
    };

    typedef Tensor<Unknown> TensorAny;
    typedef Tensor<float> Tensor32f;
    typedef Tensor<int32_t> Tensor32i;
    typedef Tensor<int8_t> Tensor8i;
    typedef Tensor<uint8_t> Tensor8u;
    typedef Tensor<int64_t> Tensor64i;
    typedef Tensor<uint64_t> Tensor64u;
    typedef Tensor<bool> TensorBool;
}
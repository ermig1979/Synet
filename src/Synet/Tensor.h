/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Buffer.h"
#include "Synet/Utils/Math.h"

namespace Synet
{
    struct Unknown
    {
    };

    namespace Detail
    {
        template <class T> TensorType GetTensorType();
        template <> SYNET_INLINE TensorType GetTensorType<Unknown>() { return TensorTypeUnknown; }
        template <> SYNET_INLINE TensorType GetTensorType<float>() { return TensorType32f; }
        template <> SYNET_INLINE TensorType GetTensorType<int32_t>() { return TensorType32i; }
        template <> SYNET_INLINE TensorType GetTensorType<int8_t>() { return TensorType8i; }
        template <> SYNET_INLINE TensorType GetTensorType<uint8_t>() { return TensorType8u; }
    }

    template<class T> class Tensor
    {
    public:
        typedef T Type;

        SYNET_INLINE Tensor()
            : _buffer(std::make_shared<Buffer>())
            , _type(TensorTypeUnknown)
            , _format(TensorFormatUnknown)
        {
        }

        SYNET_INLINE Tensor(const Synet::Shape & shape, const TensorFormat & format)
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
        {
            Resize();
        }

        SYNET_INLINE Tensor(const Synet::Shape & shape, const Type & value = Type(), const TensorFormat & format = TensorFormatUnknown, const String & name = String())
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _name(name)
        {
            Resize(value);
        }

        SYNET_INLINE Tensor(std::initializer_list<size_t> shape, const Type & value = Type(), const TensorFormat & format = TensorFormatUnknown, const String & name = String())
            : _shape(shape.begin(), shape.end())
            , _buffer(std::make_shared<Buffer>())
            , _format(format)
            , _name(name)
        {
            Resize(value);
        }

        SYNET_INLINE Tensor(const Type * data, size_t size, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown, const String & name = String())
            : _shape(shape)
            , _buffer(std::make_shared<Buffer>(data, size))
            , _type(Detail::GetTensorType<Type>())
            , _format(format)
            , _name(name)
        {
            assert(Size(0, _shape.size()) == _buffer->size);
        }

        SYNET_INLINE ~Tensor()
        {
        }

        SYNET_INLINE void Reshape(const Synet::Shape & shape, const TensorFormat & format)
        {
            _shape = shape;
            _format = format;
            Resize();
        }

        SYNET_INLINE void Reshape(std::initializer_list<size_t> shape, const TensorFormat & format)
        {
            _shape.assign(shape.begin(), shape.end());
            _format = format;
            Resize();
        }

        SYNET_INLINE void Reshape(const Synet::Shape & shape, const Type & value = Type(), const TensorFormat & format = TensorFormatUnknown, const String & name = String())
        {
            _name = name;
            _shape = shape;
            _format = format;
            Resize(value);
        }

        SYNET_INLINE void Reshape(std::initializer_list<size_t> shape, const Type & value = Type(), const TensorFormat & format = TensorFormatUnknown, const String & name = String())
        {
            _name = name;
            _shape.assign(shape.begin(), shape.end());
            _format = format;
            Resize(value);
        }

        SYNET_INLINE void Extend(const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _shape = shape;
            _format = format;
            Extend();
        }

        SYNET_INLINE void Extend(std::initializer_list<size_t> shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _shape.assign(shape.begin(), shape.end());
            _format = format;
            Extend();
        }

        SYNET_INLINE void Clear(bool saveType = false)
        {
            if(!saveType)
                _type = TensorTypeUnknown;
            _format = TensorFormatUnknown;
            _shape.clear();
#ifdef SYNET_MALLOC_DEBUG
            size_t size = _buffer->size * sizeof(T);
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

        SYNET_INLINE Tensor<float> & As32f()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32f);
            return *(Tensor<float>*)this;
        }

        SYNET_INLINE const Tensor<float> & As32f() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32f);
            return *(const Tensor<float>*)this;
        }

        SYNET_INLINE Tensor<int32_t> & As32i()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32i);
            return *(Tensor<int32_t>*)this;
        }

        SYNET_INLINE const Tensor<int32_t> & As32i() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType32i);
            return *(const Tensor<int32_t>*)this;
        }

        SYNET_INLINE Tensor<int8_t> & As8i()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8i);
            return *(Tensor<int8_t>*)this;
        }

        SYNET_INLINE const Tensor<int8_t> & As8i() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8i);
            return *(const Tensor<int8_t>*)this;
        }

        SYNET_INLINE Tensor<uint8_t> & As8u()
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8u);
            return *(Tensor<uint8_t>*)this;
        }

        SYNET_INLINE const Tensor<uint8_t> & As8u() const
        {
            assert(_type == TensorTypeUnknown || _type == TensorType8u);
            return *(const Tensor<uint8_t>*)this;
        }

        SYNET_INLINE TensorType GetType() const
        {
            return _type;
        }

        SYNET_INLINE void SetType(TensorType type)
        {
            assert(_buffer->size == 0);
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
            return _buffer->size;
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
            return offset;
        }

        SYNET_INLINE Type * CpuData()
        {
            assert(_type == Detail::GetTensorType<Type>());
            return _buffer->data;
        }

        SYNET_INLINE const Type * CpuData() const
        {
            assert(_type == Detail::GetTensorType<Type>());
            return _buffer->data;
        }

        SYNET_INLINE Type * CpuData(const Synet::Index & index)
        {
            return CpuData() + Offset(index);
        }

        SYNET_INLINE const Type * CpuData(const Synet::Index & index) const
        {
            return CpuData() + Offset(index);
        }

        SYNET_INLINE Type * CpuData(std::initializer_list<size_t> index)
        {
            return CpuData() + Offset(index);
        }

        SYNET_INLINE const Type * CpuData(std::initializer_list<size_t> index) const
        {
            return CpuData() + Offset(index);
        }

        SYNET_INLINE void Share(const Tensor & tensor)
        {
            _type = tensor._type;
            _shape = tensor._shape;
            _format = tensor._format;
            _name = tensor._name;
            _buffer = tensor._buffer;
        }

        SYNET_INLINE void ShareAs(const Tensor & tensor, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _buffer = tensor._buffer;
            _type = tensor._type;
            _name = tensor._name;
            _shape = shape;
            _format = format;
            assert(Size(0, _shape.size()) == _buffer->size);
        }

        SYNET_INLINE void ShareAs(const Type * data, size_t size, const Synet::Shape & shape, const TensorFormat & format = TensorFormatUnknown)
        {
            _type = Detail::GetTensorType<Type>();
            _shape = shape;
            _format = format;
            _buffer->Share(data, size);
            assert(Size(0, _shape.size()) == _buffer->size);
        }

        SYNET_INLINE void Clone(const Tensor & tensor)
        {
            _type = tensor._type;
            _shape = tensor._shape;
            _format = tensor._format;
            _name = tensor._name;
            _buffer.reset(tensor._buffer.Clone());
        }

        SYNET_INLINE void Import(const TensorParam & param)
        {
            switch (param.type())
            {
            case TensorType32i:
            {
                Synet::Tensor<int32_t> & i32 = As32i();
                i32.Reshape(param.shape());
                CpuCopy(param.i32().data(), param.i32().size(), i32.CpuData());
                break;
            }
            default:
                assert(0);
            }
        }

        SYNET_INLINE void Export(TensorParam & param)
        {
            param.type() = _type;
            param.shape() = _shape;
            switch (_type)
            {
            case TensorType32f:
            {
                const Synet::Tensor<float> & f32 = As32f();
                param.f32().resize(f32.Size());
                CpuCopy(f32.CpuData(), f32.Size(), param.f32().data());
                break;
            }
            case TensorType32i:
            {
                const Synet::Tensor<int32_t> & i32 = As32i();
                param.i32().resize(i32.Size());
                CpuCopy(i32.CpuData(), i32.Size(), param.i32().data());
                break;
            }
            default:
                assert(0);
            }
        }

        SYNET_INLINE size_t MemoryUsage() const
        {
            return _buffer->Owner() ? _buffer->size * sizeof(Type) : 0;
        }

        SYNET_INLINE void Capture()
        {
            _buffer->Capture();
        }

        void DebugPrint(std::ostream & os, const String & name, bool weight = false, size_t first = 5, size_t last = 2, size_t precision = 4) const
        {
            if (_shape.size() == 4 && _format == TensorFormatNhwc)
            {
                if (weight)
                {
                    Tensor trans({ _shape[3], _shape[2], _shape[0], _shape[1] }, 0, TensorFormatNchw);
                    for (size_t y = 0; y < Axis(0); ++y)
                        for (size_t x = 0; x < Axis(1); ++x)
                            for (size_t i = 0; i < Axis(2); ++i)
                                for (size_t o = 0; o < Axis(3); ++o)
                                    trans.CpuData({ o, i, y, x })[0] = CpuData({ y, x, i, o })[0];
                    std::stringstream ss;
                    ss << name << " { ";
                    for (size_t i = 0; i < _shape.size(); ++i)
                        ss << _shape[i] << " ";
                    ss << "} HWIO -> ";
                    trans.DebugPrint(os, ss.str(), weight, first, last, precision);
                }
                else
                {
                    Tensor trans({ _shape[0], _shape[3], _shape[1], _shape[2] }, 0, TensorFormatNchw);
                    for (size_t n = 0; n < Axis(0); ++n)
                        for (size_t c = 0; c < Axis(3); ++c)
                            for (size_t y = 0; y < Axis(1); ++y)
                                for (size_t x = 0; x < Axis(2); ++x)
                                    trans.CpuData({ n, c, y, x })[0] = CpuData({ n, y, x, c })[0];
                    std::stringstream ss;
                    ss << name << " { ";
                    for (size_t i = 0; i < _shape.size(); ++i)
                        ss << _shape[i] << " ";
                    ss << "} NHWC -> ";
                    trans.DebugPrint(os, ss.str(), weight, first, last, precision);
                }
                return;
            }
            os << name << " { ";
            for (size_t i = 0; i < _shape.size(); ++i)
                os << _shape[i] << " ";
            os << "} ";
            if(weight)
                os << (_format == TensorFormatNchw && _shape.size() == 4 ? "OIHW " : (_format == TensorFormatNhwc && _shape.size() == 4 ? "HWIO " : ""));
            else
                os << (_format == TensorFormatNchw ? "NCHW " : (_format == TensorFormatNhwc ? "NHWC " : ""));
            if (_buffer->data == NULL)
                return;
            os << ValueToString(_type) << std::endl;
            switch (_type)
            {
            case TensorType32f: DebugPrint(os, As32f(), first, last, precision); break;
            case TensorType32i: DebugPrint(os, As32i(), first, last, precision); break;
            case TensorType8i: DebugPrint(os, As8i(), first, last, precision); break;
            case TensorType8u: DebugPrint(os, As8u(), first, last, precision); break;
            }        
        }

        void DebugPrint(std::ostream& os, const Synet::Shape& shape, const TensorFormat& format, const String& name, 
            bool weight = false, size_t first = 5, size_t last = 2, size_t precision = 4) const
        {
            Tensor<T> tensor;
            tensor.ShareAs(*this, shape, format);
            tensor.DebugPrint(os, name, weight, first, last, precision);
        }

    private:

        template <class U> static void DebugPrint(std::ostream& os, const Tensor<U> & tensor, size_t first, size_t last, size_t precision)
        {
            size_t n = tensor.Count();
            Synet::Shape firsts(n), lasts(n), index(n, 0);
            Strings separators(n);
            for (ptrdiff_t i = n - 1; i >= 0; --i)
            {
                if (i == n - 1)
                {
                    firsts[i] = first;
                    lasts[i] = last;
                    separators[i] = "\t";
                }
                else
                {
                    firsts[i] = std::max<size_t>(firsts[i + 1] - 1, 1);
                    lasts[i] = std::max<size_t>(lasts[i + 1] - 1, 1);
                    separators[i] = separators[i + 1] + "\n";
                }
            }
            DebugPrint(os, tensor, firsts, lasts, separators, index, 0, precision, DebugPrintPadding(tensor.CpuData(), tensor.Size(), precision));
            if (n == 1 || n == 0)
                os << "\n";
        }

        template <class U> static void DebugPrint(std::ostream & os, const Tensor<U> & tensor, 
            const Synet::Shape & firsts, const Synet::Shape & lasts, const Strings & separators, Synet::Shape index, size_t order, size_t precision, size_t padding)
        {
            if (order == tensor.Count())
            {
                Synet::DebugPrint(os, tensor.CpuData(index)[0], precision, padding);
                return;
            }
            if (firsts[order] + lasts[order] < tensor.Axis(order))
            {
                size_t lo = firsts[order], hi = tensor.Axis(order) - lasts[order];
                for (index[order] = 0; index[order] < lo; ++index[order])
                {
                    DebugPrint(os, tensor, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
                os << "..." << separators[order];
                for (index[order] = hi; index[order] < tensor.Axis(order); ++index[order])
                {
                    DebugPrint(os, tensor, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
            }
            else
            {
                for (index[order] = 0; index[order] < tensor.Axis(order); ++index[order])
                {
                    DebugPrint(os, tensor, firsts, lasts, separators, index, order + 1, precision, padding);
                    os << separators[order];
                }
            }  
        }

        SYNET_INLINE void Resize(const Type & value)
        {
            _type = Detail::GetTensorType<Type>();
            assert(_type != TensorTypeUnknown);
            size_t size = Size(0, _shape.size());
            _buffer->Resize(size);
            CpuSet(_buffer->size, value, _buffer->data);
        }

        SYNET_INLINE void Resize()
        {
            _type = Detail::GetTensorType<Type>();
            assert(_type != TensorTypeUnknown);
            size_t size = Size(0, _shape.size());
            _buffer->Resize(size);
            CpuSet(_buffer->size, Type(), _buffer->data);
            //CpuTouch(_buffer->data, _buffer->size);
        }

        SYNET_INLINE void Extend()
        {
            if(_type == TensorTypeUnknown)
                _type = Detail::GetTensorType<Type>();
            assert(_type != TensorTypeUnknown && _type == Detail::GetTensorType<Type>());
            size_t size = Size(0, _shape.size());
            if (size > _buffer->size)
                _buffer->Resize(size);
        }

        typedef Synet::Buffer<Type> Buffer;
        typedef std::shared_ptr<Buffer> BufferPtr;

        Synet::String _name;
        TensorType _type;
        TensorFormat _format;
        Synet::Shape _shape;
        BufferPtr _buffer;
    };

    template<class T> void DebugPrint(std::ostream& os, const std::vector<T>& src, const String& name, size_t first, size_t last, size_t precision)
    {
        Tensor<T> tensor;
        tensor.ShareAs(src.data(), src.size(), Shape({ src.size() }));
        tensor.DebugPrint(os, name, false, first, last, precision);
    }

    typedef Tensor<Unknown> TensorAny;
    typedef Tensor<float> Tensor32f;
    typedef Tensor<int32_t> Tensor32i;
    typedef Tensor<int8_t> Tensor8i;
    typedef Tensor<uint8_t> Tensor8u;
}
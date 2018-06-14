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
#include "Synet/Params.h"
#include "Synet/Math.h"

namespace Synet
{
    namespace Detail
    {
        template <class T> TensorType GetTensorType();
        template <> SYNET_INLINE TensorType GetTensorType<float>() { return TensorType32f; }
        template <> SYNET_INLINE TensorType GetTensorType<int32_t>() { return TensorType32i; }
    }

    template<class T> class Tensor
    {
    public:
        typedef T Type;

        SYNET_INLINE Tensor()
            : _size(0)
            , _cpuData(std::make_shared<Vector>())
            , _type(TensorTypeUnknown)
        {
        }

        SYNET_INLINE Tensor(const Synet::Shape & shape, const Type & value = Type(), const String & name = String())
            : _shape(shape)
            , _cpuData(std::make_shared<Vector>())
            , _name(name)
        {
            Resize(value);
        }

        SYNET_INLINE Tensor(std::initializer_list<size_t> shape, const Type & value = Type(), const String & name = String())
            : _shape(shape.begin(), shape.end())
            , _cpuData(std::make_shared<Vector>())
            , _name(name)
        {
            Resize(value);
        }

        SYNET_INLINE ~Tensor()
        {
        }

        SYNET_INLINE void Reshape(const Synet::Shape & shape, const Type & value = Type(), const String & name = String())
        {
            _name = name;
            _shape = shape;
            Resize(value);
        }

        SYNET_INLINE void Reshape(std::initializer_list<size_t> shape, const Type & value = Type(), const String & name = String())
        {
            _name = name;
            _shape.assign(shape.begin(), shape.end());
            Resize(value);
        }

        SYNET_INLINE void Extend(const Synet::Shape & shape)
        {
            _shape = shape;
            Extend();
        }

        SYNET_INLINE void Extend(std::initializer_list<size_t> shape)
        {
            _shape.assign(shape.begin(), shape.end());
            Extend();
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

        SYNET_INLINE TensorType GetType() const
        {
            return _type;
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
            return _cpuData->data();
        }

        SYNET_INLINE const Type * CpuData() const
        {
            assert(_type == Detail::GetTensorType<Type>());
            return _cpuData->data();
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
            _name = tensor._name;
            _size = tensor._size;
            _cpuData = tensor._cpuData;
            SetDebugPtr();
        }

        SYNET_INLINE void ShareAs(const Tensor & tensor, const Synet::Shape & shape)
        {
            _type = tensor._type;
            _shape = shape;
            _size = Size(0, _shape.size());
            assert(_size == tensor._size);
            _cpuData = tensor._cpuData;
            SetDebugPtr();
        }

        SYNET_INLINE void Clone(const Tensor & tensor)
        {
            _type = tensor._type;
            _shape = tensor._shape;
            _name = tensor._name;
            _size = tensor._size;
            _cpuData(std::make_shared<Vector>(tensor._cpuData->begin(), tensor._cpuData->end()));
            SetDebugPtr();
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
                param.f32().resize(_size);
                const Synet::Tensor<float> & f32 = As32f();
                CpuCopy(f32.CpuData(), _size, param.f32().data());
                break;
            }
            case TensorType32i:
            {
                param.i32().resize(_size);
                const Synet::Tensor<int32_t> & i32 = As32i();
                CpuCopy(i32.CpuData(), _size, param.i32().data());
                break;
            }
            default:
                assert(0);
            }
        }

#ifdef SYNET_DEBUG_PRINT_ENABLE
        void DebugPrint(std::ostream & os, const String & name, size_t first = 5, size_t last = 2) const
        {
            os << name << " { ";
            for (size_t i = 0; i < _shape.size(); ++i)
                os << _shape[i] << " ";
            os << "} " << std::endl;

            if (_size == 0)
                return;

            size_t n = _shape.size();
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
            DebugPrint(os, firsts, lasts, separators, index, 0);
            if (n == 1 || n == 0)
                os << "\n";
        }
#endif

    private:

#ifdef SYNET_DEBUG_PRINT_ENABLE
        void DebugPrint(std::ostream & os, const Synet::Shape & firsts, const Synet::Shape & lasts, const Strings & separators, Synet::Shape index, size_t order) const
        {
            if (order == _shape.size())
            {
                std::cout << std::fixed << std::setprecision(4);
                os << *CpuData(index);
                return;
            }
            if (firsts[order] + lasts[order] < _shape[order])
            {
                size_t lo = firsts[order], hi = _shape[order] - lasts[order];
                for (index[order] = 0; index[order] < lo; ++index[order])
                {
                    DebugPrint(os, firsts, lasts, separators, index, order + 1);
                    os << separators[order];
                }
                os << "..." << separators[order];
                for (index[order] = hi; index[order] < _shape[order]; ++index[order])
                {
                    DebugPrint(os, firsts, lasts, separators, index, order + 1);
                    os << separators[order];
                }
            }
            else
            {
                for (index[order] = 0; index[order] < _shape[order]; ++index[order])
                {
                    DebugPrint(os, firsts, lasts, separators, index, order + 1);
                    os << separators[order];
                }
            }  
        }
#endif

        SYNET_INLINE void Resize(const Type & value)
        {
            _type = Detail::GetTensorType<Type>();
            _size = Size(0, _shape.size());
            _cpuData->resize(_size, value);
            SetDebugPtr();
        }

        SYNET_INLINE void Extend()
        {
            if(_type == TensorTypeUnknown)
                _type = Detail::GetTensorType<Type>();
            assert(_type == Detail::GetTensorType<Type>());
            _size = Size(0, _shape.size());
            if (_size > _cpuData->size())
                _cpuData->resize(_size);
            SetDebugPtr();
        }

#if defined(_DEBUG) && defined(_MSC_VER)
        const Type * _ptr;

        SYNET_INLINE void SetDebugPtr()
        {
            _ptr = _cpuData->data();
        }
#else
        SYNET_INLINE void SetDebugPtr()
        {
        }
#endif

#if defined(SYNET_SIMD_LIBRARY_ENABLE)
        typedef std::vector<Type, Simd::Allocator<Type>> Vector;
#else
        typedef std::vector<Type, std::allocator<Type>> Vector;
#endif
        typedef std::shared_ptr<Vector> VectorPtr;

        Synet::String _name;
        TensorType _type;
        Synet::Shape _shape;
        size_t _size;
        VectorPtr _cpuData;
    };
}
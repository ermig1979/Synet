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
    template<class T, template<class> class Allocator = std::allocator> class Tensor
    {
    public:
        typedef T Type;

        SYNET_INLINE Tensor()
            : _size(0)
        {
            _data.reset(new Vector());
        }

        SYNET_INLINE Tensor(const Synet::Shape & shape, const Type & value = Type())
            : _shape(shape)
        {
            _size = Size(0, _shape.size());
            _data.reset(new Vector(_size, value));
        }

        SYNET_INLINE Tensor(std::initializer_list<size_t> shape, const Type & value = Type())
            : _shape(shape.begin(), shape.end())
        {
            _size = Size(0, _shape.size());
            _data.reset(new Vector(_size, value));
        }

        SYNET_INLINE ~Tensor()
        {
        }

        SYNET_INLINE void Reshape(const Synet::Shape & shape, const Type & value = Type())
        {
            _shape = shape;
            _size = Size(0, _shape.size());
            _data->resize(_size, value);
        }

        SYNET_INLINE void Reshape(std::initializer_list<size_t> shape, const Type & value = Type())
        {
            _shape.assign(shape.begin(), shape.end());
            _size = Size(0, _shape.size());
            _data->resize(_size, value);
        }

        SYNET_INLINE void Extend(const Synet::Shape & shape)
        {
            _shape = shape;
            _size = Size(0, _shape.size());
            if(_size > _data->size())
                _data->resize(_size);
        }

        SYNET_INLINE void Extend(std::initializer_list<size_t> shape)
        {
            _shape.assign(shape.begin(), shape.end());
            _size = Size(0, _shape.size());
            if (_size > _data->size())
                _data->resize(_size);
        }

        SYNET_INLINE const Synet::Shape & Shape() const
        {
            return _shape;
        }

        SYNET_INLINE size_t Count() const
        {
            return _shape.size();
        }

        SYNET_INLINE size_t Axis(size_t axis) const
        {
            return _shape[axis];
        }

        SYNET_INLINE size_t Size(size_t startAxis, size_t endAxis) const
        {
            assert(startAxis <= endAxis && endAxis <= _shape.size());

            size_t size = 1;
            for (size_t axis = startAxis; axis < endAxis; ++axis)
                size *= _shape[axis];
            return size;
        }

        SYNET_INLINE size_t Size(size_t startAxis) const
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
            for (const size_t * s = _shape.data(), * i = index.begin(); i < index.end(); ++s, ++i)
            {
                assert(*s > 0);
                assert(*i < *s);

                offset *= *s;
                offset += *i;
            }
            return offset;
        }

        SYNET_INLINE Type * Data()
        {
            return _data->data();
        }

        SYNET_INLINE const Type * Data() const
        {
            return _data->data();
        }

        SYNET_INLINE Type * Data(const Synet::Index & index)
        {
            return _data->data() + Offset(index);
        }

        SYNET_INLINE const Type * Data(const Synet::Index & index) const
        {
            return _data->data() + Offset(index);
        }

        SYNET_INLINE Type * Data(std::initializer_list<size_t> index)
        {
            return _data->data() + Offset(index);
        }

        SYNET_INLINE const Type * Data(std::initializer_list<size_t> index) const
        {
            return _data->data() + Offset(index);
        }

        SYNET_INLINE void Share(const Tensor & tensor)
        {
            _data = tensor._data;
        }

    private:
        typedef std::vector<Type, Allocator<Type> > Vector;
        typedef std::shared_ptr<Vector> VectorPtr;

        Synet::Shape _shape;
        size_t _size;
        VectorPtr _data;
    };
}
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
        typedef std::vector<size_t> Shape;
        typedef std::vector<size_t> Index;

        inline Tensor()
        {
        }

        inline Tensor(const Shape & shape, const Type & value = Type())
            : _shape(shape)
        {
            _data.resize(Size(), value);
        }

        inline ~Tensor()
        {
        }

        inline void Reshape(const Shape & shape, const Type & value = Type())
        {
            _shape = shape;            
            _data.resize(Size(), value);
        }

        inline const Shape & GetShape() const
        {
            return _shape;
        }

        inline size_t AxisCount() const
        {
            return _shape.size();
        }

        inline size_t AxisSize(size_t axis) const
        {
            return _shape[axis];
        }

        inline size_t Size(size_t startAxis, size_t endAxis) const
        {
            assert(startAxis < endAxis && endAxus <= _shape.size());

            size_t size = 1;
            for (size_t axis = startAxis; axis < endAxis; ++axis)
                size *= _shape[axis];
            return size;
        }

        inline size_t Size() const
        {
            return Size(0, _shape.size());
        }

        inline Type * Data()
        {
            return _data.data();
        }

        inline const Type * Data() const
        {
            return _data.data();
        }

        size_t Offset(const Index & index) const
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

        inline Type * DataAt(const Index & index)
        {
            return _data.data() + Offset(index);
        }

        inline const Type * DataAt(const Index & index) const
        {
            return _data.data() + Offset(index);
        }

    private:
        typedef std::vector<Type, Allocator<Type> > Vector;

        Shape _shape;
        Vector _data;
    };
}
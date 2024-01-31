/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
    namespace Detail
    {
        SYNET_INLINE void * Allocate(size_t size)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return ::SimdAllocate(size, ::SimdAlignment());
#else
            return ::malloc(size);
#endif
        }

        SYNET_INLINE void Free(void * ptr)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            return ::SimdFree(ptr);
#else
            return ::free(ptr);
#endif
        }
    }

    //-------------------------------------------------------------------------------------------------

    template <class T> struct Buffer
    {
        typedef T Type;

        Type * const data;
        size_t const size;

        SYNET_INLINE Buffer(size_t size_ = 0)
            : data(0)
            , size(0)
            , _owner(false)
        {
            Resize(size_);
        }

        SYNET_INLINE Buffer(const Type * data_, size_t size_)
            : data((Type * const)data_)
            , size(size_)
            , _owner(false)
        {
        }

        SYNET_INLINE Buffer(const Buffer & buffer)
            : data(buffer.data)
            , size(buffer.size)
            , _owner(false)
        {
        }

        SYNET_INLINE ~Buffer()
        {
            if (_owner)
            {
                Detail::Free(data);
                _owner = false;
            }
        }

        SYNET_INLINE void Resize(size_t size_)
        {
            if (size_ != size)
            {
                if (_owner)
                {
                    Detail::Free(data);
                    _owner = false;
                }
                *(size_t*)&size = size_;
                if (size_)
                {
                    *(Type**)&data = (Type*)Detail::Allocate(size * sizeof(Type));
                    _owner = true;
                }
            }
        }

        SYNET_INLINE void Share(const Type * data_, size_t size_)
        {
            if (_owner)
            {
                Detail::Free(data);
                _owner = false;
            }
            *(size_t*)&size = size_;
            *(const Type**)&data = data_;
        }

        SYNET_INLINE void Share(const Buffer & other)
        {
            Share(other.data, other.size);
        }

        SYNET_INLINE void Swap(Buffer & other)
        {
            std::swap((size_t&)size, (size_t&)other.size);
            std::swap((Type*&)data, (Type*&)other.data);
            std::swap((bool&)_owner, (bool&)other._owner);
        }

        SYNET_INLINE Buffer * Clone() const 
        {
            Buffer * clone = new Buffer(size);
            memcpy(clone->data, data, size*sizeof(Type));
            return clone;
        }

        SYNET_INLINE Type & operator[] (size_t i)
        {
            return data[i];
        }

        SYNET_INLINE const Type & operator[] (size_t i) const
        {
            return data[i];
        }

        SYNET_INLINE bool Owner() const
        {
            return _owner;
        }

        SYNET_INLINE void Capture()
        {
            if(!_owner)
            {
                Buffer clone(size);
                memcpy(clone.data, data, size * sizeof(Type));
                Swap(clone);
            }
        }

    private:
        bool _owner;
    };
}
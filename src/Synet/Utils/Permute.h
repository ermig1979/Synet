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

#include "Synet/Tensor.h"

namespace Synet
{
    class Permute
    {
    public:
        Permute()
            : _context(NULL)
        {
        }

        virtual ~Permute()
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdRelease(_context);
#endif
        }

        void Init(const Shape& shape, const Shape& order, TensorType type)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_shape != shape || _order != order)
            {
                assert(shape.size() == order.size());
                _shape = shape;
                _order = order;
                if (_context)
                    ::SimdRelease(_context);
                _context = ::SimdSynetPermuteInit(_shape.data(), _order.data(), _shape.size(), Convert(type));
            }
#endif
        }

        bool Enable()
        {
            return _context != NULL;
        }

        size_t InternalBufferSize() const
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            return _context ? ::SimdSynetPermuteInternalBufferSize(_context) : 0;
#else
            return 0;
#endif
        }

        void Forward(const uint8_t* src, uint8_t* dst)
        {
#if defined(SYNET_SIMD_LIBRARY_ENABLE) && !defined(SYNET_SIMD_SYNET_DISABLE)
            if (_context)
                ::SimdSynetPermuteForward(_context, src, dst);
#endif
        }

    private:
        void * _context;
        Shape _shape, _order;

        static SYNET_INLINE::SimdTensorDataType Convert(TensorType type)
        {
            switch (type)
            {
            case TensorType32f: return SimdTensorData32f;
            case TensorType32i: return SimdTensorData32i;
            case TensorType8i: return SimdTensorData8i;
            case TensorType8u: return SimdTensorData8u;
            default:
                assert(0);
                return SimdTensorData32f;
            }
        }
    };
}
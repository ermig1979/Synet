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

#include "Synet/Params.h"

namespace Synet
{
    class Add16b
    {
    public:
        Add16b()
            : _context(NULL)
        {
        }

        virtual ~Add16b()
        {
            Clear();
        }

        SYNET_INLINE void Init(const Shape & aShape, TensorType aType, const Shape& bShape, TensorType bType, TensorType dstType, TensorFormat format)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_aShape != aShape || _bShape != bShape)
            {
                Clear();
                _aShape = aShape, _bShape = aShape;
                _context = ::SimdSynetAdd16bInit(_aShape.data(), _aShape.size(), (SimdTensorDataType)aType, 
                    _bShape.data(), _bShape.size(), (SimdTensorDataType)bType,
                    (SimdTensorDataType)dstType, (SimdTensorFormatType)format);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetAdd16bForward(_context, a, b, dst);
#endif
        }

        SYNET_INLINE void Clear()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
            _aShape.clear();
            _bShape.clear();
#endif
        }

    private:
        void* _context;
        Shape _aShape, _bShape;
    };

    //-------------------------------------------------------------------------------------------------

    class QuantizedAdd
    {
    public:
        QuantizedAdd()
            : _context(NULL)
        {
        }

        virtual ~QuantizedAdd()
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdRelease(_context), _context = NULL;
#endif
        }

        SYNET_INLINE void Init(const Shape& aShape, TensorType aType, float aScale, int32_t aZero, const Shape& bShape, TensorType bType, float bScale, int32_t bZero,
            ActivationFunctionType actType, const float* actParams, TensorType dstType, float dstScale, int32_t dstZero)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_aShape != aShape || _bShape != bShape)
            {
                _aShape = aShape, _bShape = aShape;
                if (_context)
                    ::SimdRelease(_context), _context = NULL;
                _context = ::SimdSynetQuantizedAddInit(
                    _aShape.data(), _aShape.size(), (SimdTensorDataType)aType, &aScale, aZero, 
                    _bShape.data(), _bShape.size(), (SimdTensorDataType)bType, &bScale, bZero,
                    (SimdConvolutionActivationType)actType, actParams, (SimdTensorDataType)dstType, &dstScale, dstZero);
            }
#endif
        }

        SYNET_INLINE bool Enable() const
        {
            return _context != NULL;
        }

        SYNET_INLINE void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
#ifdef SYNET_SIMD_LIBRARY_ENABLE
            if (_context)
                ::SimdSynetQuantizedAddForward(_context, a, b, dst);
#endif
        }

    private:
        void* _context;
        Shape _aShape, _bShape;
    };

}
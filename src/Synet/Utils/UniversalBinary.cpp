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

#include "Synet/Utils/UniversalBinary.h"
#include "Synet/Utils/Math.h"
#include "Synet/Quantization/Convert.h"

namespace Synet
{
    template <BinaryOperationType type, typename T> struct BinaryOperation;

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeAdd, T>
    {
        static T Run(T a, T b)
        {
            return a + b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeAdd, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <typename T> struct BinaryOperation<BinaryOperationTypeAnd, T>
    {
        static T Run(T a, T b)
        {
            return And(a, b);
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeDiv, T>
    {
        static T Run(T a, T b)
        {
            return a / b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeDiv, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeMod, T>
    {
        static T Run(T a, T b)
        {
            return a % b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeMod, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeMod, float>
    {
        static float Run(float a, float b)
        {
            return ::fmodf(a, b);
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <class T> struct BinaryOperation<BinaryOperationTypeSub, T>
    {
        static T Run(T a, T b)
        {
            return a - b;
        }
    };

    template <> struct BinaryOperation<BinaryOperationTypeSub, bool>
    {
        static bool Run(bool a, bool b)
        {
            return false;
        }
    };

    //-------------------------------------------------------------------------------------------------

    template <BinaryOperationType O, class T, size_t N> static void UniversalBinary(const uint8_t* a8, const Shape& aSteps, const uint8_t* b8, const Shape& bSteps, uint8_t* dst8, const Shape& dstShape)
    {
        const T* a = (const T*)a8;
        const T* b = (const T*)b8;
        T* dst = (T*)dst8;
        if (N == 1)
        {
            const T* a0 = a;
            const T* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                *dst++ = BinaryOperation<O, T>::Run(*a0, *b0);
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 2)
        {
            const T* a0 = a;
            const T* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0;
                const T* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    *dst++ = BinaryOperation<O, T>::Run(*a1, *b1);
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 3)
        {
            const T* a0 = a;
            const T* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0;
                const T* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const T* a2 = a1;
                    const T* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        *dst++ = BinaryOperation<O, T>::Run(*a2, *b2);
                        a2 += aSteps[2];
                        b2 += bSteps[2];
                    }
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else if (N == 4)
        {
            const T* a0 = a;
            const T* b0 = b;
            for (size_t i0 = 0; i0 < dstShape[0]; ++i0)
            {
                const T* a1 = a0;
                const T* b1 = b0;
                for (size_t i1 = 0; i1 < dstShape[1]; ++i1)
                {
                    const T* a2 = a1;
                    const T* b2 = b1;
                    for (size_t i2 = 0; i2 < dstShape[2]; ++i2)
                    {
                        const T* a3 = a2;
                        const T* b3 = b2;
                        for (size_t i3 = 0; i3 < dstShape[3]; ++i3)
                        {
                            *dst++ = BinaryOperation<O, T>::Run(*a3, *b3);
                            a3 += aSteps[3];
                            b3 += bSteps[3];
                        }
                        a2 += aSteps[2];
                        b2 += bSteps[2];
                    }
                    a1 += aSteps[1];
                    b1 += bSteps[1];
                }
                a0 += aSteps[0];
                b0 += bSteps[0];
            }
        }
        else
            assert(0);
    }

    //-------------------------------------------------------------------------------------------------

    template<BinaryOperationType O, class T> UniversalBinaryPtr GetUniversalBinary(size_t dim)
    {
        switch (dim)
        {
        case 1: return UniversalBinary<O, T, 1>;
        case 2: return UniversalBinary<O, T, 2>;
        case 3: return UniversalBinary<O, T, 3>;
        case 4: return UniversalBinary<O, T, 4>;
        default: return NULL;
        }
    }

    template<class T> UniversalBinaryPtr GetUniversalBinary(BinaryOperationType op, size_t dim)
    {
        switch (op)
        {
        case BinaryOperationTypeAdd: return std::is_same<T, bool>::value ? NULL : GetUniversalBinary<BinaryOperationTypeAdd, T>(dim);
        case BinaryOperationTypeAnd: return GetUniversalBinary<BinaryOperationTypeAnd, T>(dim);
        case BinaryOperationTypeDiv: return std::is_same<T, bool>::value ? NULL : GetUniversalBinary<BinaryOperationTypeDiv, T>(dim);
        case BinaryOperationTypeMod: return std::is_same<T, bool>::value ? NULL : GetUniversalBinary<BinaryOperationTypeMod, T>(dim);
        case BinaryOperationTypeSub: return std::is_same<T, bool>::value ? NULL : GetUniversalBinary<BinaryOperationTypeSub, T>(dim);
        default: return NULL;
        }
    }

    UniversalBinaryPtr GetUniversalBinary(BinaryOperationType op, TensorType type, size_t dim)
    {
        switch (type)
        {
        case TensorType32f: return GetUniversalBinary<float>(op, dim);
        case TensorType64i: return GetUniversalBinary<int64_t>(op, dim);
        case TensorTypeBool: return GetUniversalBinary<bool>(op, dim);
        default: return NULL;
        }
    }
}
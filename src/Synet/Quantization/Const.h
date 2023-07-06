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
    const int QUANT_IE_COMP_SRC_U8_MIN = 0;
    const int QUANT_IE_COMP_SRC_U8_MAX = 255;
    const int QUANT_IE_COMP_SRC_I8_MAX = 127;
    const int QUANT_IE_COMP_SRC_I8_MIN = -128;
    const int QUANT_IE_COMP_WEIGHT_MIN = -128;
    const int QUANT_IE_COMP_WEIGHT_MAX = 127;

    const int QUANT_SYMM_NARR_SRC_U8_MIN = 0;
    const int QUANT_SYMM_NARR_SRC_U8_MAX = 180;
    const int QUANT_SYMM_NARR_SRC_I8_MIN = -90;
    const int QUANT_SYMM_NARR_SRC_I8_MAX = 90;
    const int QUANT_SYMM_NARR_WEIGHT_MIN = -90;
    const int QUANT_SYMM_NARR_WEIGHT_MAX = 90;
}

/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Test/TestCommon.h"

namespace Test
{
    namespace 
    {
        typedef std::vector<int> Ints;

        struct Test0
        {
            SYNET_PARAM_VALUE(int, t0_p1, 1);
            SYNET_PARAM_VALUE(double, t0_p2, 2.0);
            SYNET_PARAM_VALUE(String, t0_p3, "3.0");
        };

        struct Test1
        {
            SYNET_PARAM_VALUE(int, t1_p1, 4);
            SYNET_PARAM_STRUCT(Test0, t1_p2);
            SYNET_PARAM_VECTOR(Test0, t1_p3);
        };

        struct Test2
        {
            SYNET_PARAM_VALUE(Ints, t2_p0, Ints());
            SYNET_PARAM_VALUE(String, t2_p1, "5.0");
            SYNET_PARAM_STRUCT(Test0, t2_p2);
            SYNET_PARAM_STRUCT(Test1, t2_p3);
            SYNET_PARAM_VECTOR(Test1, t2_p4);
            SYNET_PARAM_VALUE(Ints, t2_p5, Ints());
        };

        SYNET_PARAM_HOLDER(Holder, Test2, root);
    }

    bool TestParam()
    {
        Holder holder1;
        holder1().t2_p2().t0_p1() = 3;
        holder1().t2_p1() = "Value";
        holder1().t2_p3().t1_p2().t0_p2() = 1.2;
        holder1().t2_p4().resize(2);
        holder1().t2_p4()[1].t1_p1() = 10;
        holder1().t2_p0().resize(3, 6);

        std::cout << std::endl << "Saved (only changed):" << std::endl;
        std::stringstream tmp;
        holder1.Save(std::cout, false);
        holder1.Save(tmp, false);

        Holder holder2;
        holder2.Load(tmp);
        std::cout << std::endl << "Saved and Loaded (full):" << std::endl;
        holder2.Save(std::cout, true);

        return true;
    }
}

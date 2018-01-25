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
    SYNET_PARAM_ROOT(Synet::NetworkParam, Config);

    bool TestParams()
    {
        Config config;
        config().name() = "Network_Test_1";
        config().layers().resize(2);

        config().layers()[0].type() = Synet::LayerTypeInput;
        config().layers()[0].name() = "Input";
        config().layers()[0].inputLayer().shape() = Synet::Shape({1, 2, 3});

        config().layers()[1].type() = Synet::LayerTypeInnerProduct;
        config().layers()[1].name() = "Full_1";

        std::cout << std::endl << "Saved (only changed):" << std::endl;
        config.Save(std::cout, false);

        //std::cout << std::endl << "Saved (full):" << std::endl;
        //config.Save(std::cout, true);

        return true;
    }
}
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

#include "Test/TestConfig.h"

namespace Test
{
    struct Test0
    {
        SYNET_ELEM_PARAM(int, test0_param1, 1);
        SYNET_ELEM_PARAM(double, test0_param2, 2.0);
        SYNET_ELEM_PARAM(Synet::String, test0_param3, "3.0");
    };

    struct Test1
    {
        SYNET_ELEM_PARAM(int, test1_param1, 4);
        SYNET_NODE_PARAM(Test0, test1_param2);
        SYNET_NODE_PARAM(Test0, test1_param3);
    };

    struct Test2
    {
        SYNET_ELEM_PARAM(Synet::String, test2_param1, "5.0");
        SYNET_NODE_PARAM(Test0, test2_param2);
        SYNET_NODE_PARAM(Test1, test2_param3);
    };

    SYNET_ROOT_CLASS(Test2, Config);

    bool ParamTest()
    {
        Config config;
        config().test2_param2().test0_param1() = 3;
        config().test2_param1() = "Value";
        config().test2_param3().test1_param2().test0_param2() = 1.2;

        Synet::XmlSaver saver;
        config.Save(saver);

        return true;
    }
}

int main(int argc, char* argv[])
{
    Test::ParamTest();


    Synet::NetworkParam netParam;
    typedef Synet::Network<float> Network;
    Network net(netParam);

    Synet::InputLayerParam inputLayerParam("Input", Synet::Shape({1, 1, 1, 1}));
    Synet::InputLayer<float> inputLayer(inputLayerParam);

    Synet::InnerProductLayer<float> innerProductLayer(Synet::InnerProductLayerParam("InnerLayer"));

    return 0;
}


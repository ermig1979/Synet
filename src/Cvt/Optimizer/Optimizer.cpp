/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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

#include "Cvt/Optimizer/Optimizer.h"

namespace Synet
{
    bool OptimizeSynetModel(const String& srcXml, const String& srcBin, const String& dstXml, const String & dstBin, const OptimizerParam & param)
    {
        NetworkParamHolder network;
        if (!network.Load(srcXml))
        {
            std::cout << "Can't load Synet model '" << srcXml << "' !" << std::endl;
            return false;
        }
        Bytes bin;
        if (!srcBin.empty() && !LoadBinaryData(srcBin, bin))
        {
            std::cout << "Can't load Synet weight '" << srcBin << "' !" << std::endl;
            return false;
        }
        Optimizer optimizer(param);
        if (!optimizer.Run(network(), bin))
        {
            std::cout << "Can't optimize Synet model!" << std::endl;
            return false;
        }
        if (!network.Save(dstXml, false))
        {
            std::cout << "Can't save Synet model '" << dstXml << "' !" << std::endl;
            return false;
        }
        if (!dstBin.empty() && !SaveBinaryData(bin, dstBin))
        {
            std::cout << "Can't save Synet weight '" << dstBin << "' !" << std::endl;
            return false;
        }
        return true;
    }
}
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

#include "Cvt/Deoptimizer/Deoptimizer.h"

namespace Synet
{
	bool DeoptimizeSynetModel(const String& srcModel, const String& dstModel)
	{
		NetworkParamHolder network;
		if (!network.Load(srcModel))
		{
			std::cout << "Can't load '" << srcModel << "' model file!" << std::endl;
			return false;
		}
		Deoptimizer deoptimizer;
		if (!deoptimizer.Run(network()))
		{
			std::cout << "Can't deoptimize Synet model!" << std::endl;
			return false;
		}
		if (!network.Save(dstModel, false))
		{
			std::cout << "Can't save '" << dstModel << "' model file!" << std::endl;
			return false;
		}
		return true;
	}
}
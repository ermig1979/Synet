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

#pragma once

#include "Synet/Common.h"
#include "Synet/Params.h"
#include "Synet/Utils/FileUtils.h"

#include "Cvt/Common/Params.h"
#include "Cvt/Common/SynetUtils.h"
#include "Cvt/Optimizer/Common.h"

namespace Synet
{
    class Optimizer : public SynetUtils
    {
    public:
        Optimizer(const OptimizerParam& param);

        bool Run(Synet::NetworkParam& network, Bytes& bin);

    private:
        const OptimizerParam & _param;

        bool OptimizeLayers(Synet::NetworkParam& network, Bytes& bin, int stage);

        bool Rename(const Change & change, LayerParams & layers);

        bool Rename(const Changes & changes, LayerParams & layers);

        size_t Users(const String& name, const LayerParams& layers, size_t start, const String & parent) const;

        bool CanReuse(const LayerParam & layer);

        bool HasOutput(const Synet::NetworkParam& network, const LayerParam & layer);

        bool ReuseLayers(Synet::NetworkParam& network);

        bool IsStub(const LayerParam& layer, const Synet::NetworkParam& network);

        bool RemoveStub(Synet::NetworkParam& network);

        bool IsNnwc(const NetworkParam& network);
    };

    //--------------------------------------------------------------------------------------------------

    bool OptimizeSynetModel(const String& srcXml, const String& srcBin, const String& dstXml, const String& dstBin, const OptimizerParam& param = OptimizerParam());
}
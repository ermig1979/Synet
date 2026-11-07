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

namespace Synet
{
    class Bf16OptSetter : public SynetUtils
    {
    public:
        Bf16OptSetter(const Bf16OptParam& param);

        bool Run(Synet::NetworkParam & network, Bytes & bin);

    private:
        typedef std::vector<Synet::LayerParam> LayerParams;
        typedef std::pair<String, String> Change;
        typedef std::vector<Change> Changes;
        typedef std::vector<LayerType> LayerTypes;
        typedef std::set<String> StringSet;

        const Bf16OptParam& _param;

        //-------------------------------------------------------------------------------------------------

        bool SetSimpleCase(LayerParams& layers);

        bool SetManualActive(LayerParams& layers);

        bool SetConvAddReluCase(LayerParams& layers);

        bool IsExclude(const LayerParams& layers, const String& name) const;

        size_t EffectiveSrcC(const LayerParam& layer);

        size_t EffectiveDstC(const LayerParam& layer);

        size_t IsDwepthwiseNhwc(const LayerParam& layer);
    };
}

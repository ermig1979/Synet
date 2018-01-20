/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Layer.h"
#include "Synet/InputLayer.h"
#include "Synet/InnerProductLayer.h"
#include "Synet/ReluLayer.h"
#include "Synet/SigmoidLayer.h"
#include "Synet/PoolingLayer.h"
#include "Synet/ConvolutionLayer.h"

namespace Synet
{
    struct LayerTypeName
    {
        LayerParam::Type type;
        String name;
    };

    const LayerTypeName g_layerTypeNames[] =
    {
        { LayerParam::InputLayer, "InputLayer" },
        { LayerParam::InnerProductLayer, "InnerProductLayer" },
        { LayerParam::ReluLayer, "ReluLayer" },
        { LayerParam::SigmoidLayer, "SigmoidLayer" },
        { LayerParam::PoolingLayer, "PoolingLayer" },
        { LayerParam::ConvolutionLayer, "ConvolutionLayer" },
    };

    String LayerParam::ToString(Type type)
    {
        if (type > LayerParam::UnknownLayer && type < LayerParam::LayerTypeSize)
            return g_layerTypeNames[type].name;
        else
            return "";
    }

    LayerParam::Type LayerParam::FromString(const String & name)
    {
        LayerParam::Type type = (LayerParam::Type)(LayerParam::LayerTypeSize - 1);
        for (; type > LayerParam::UnknownLayer; type = (LayerParam::Type)((int)type - 1))
        {
            if (g_layerTypeNames[type].name == name)
                return type;
        }
        return type;
    }

    template <class T, template<class> class A> Layer<T, A> * Layer<T, A>::Create(const LayerParam & param)
    {
        switch (param.type)
        {
        case LayerParam::InputLayer: return new InputLayer<T, A>(*(InputLayerParam*)&param);
        case LayerParam::InnerProductLayer: return new InnerProductLayer<T, A>(*(InnerProductLayerParam*)&param);
        case LayerParam::ReluLayer: return new ReluLayer<T, A>(*(ReluLayerParam*)&param);
        case LayerParam::SigmoidLayer: return new SigmoidLayer<T, A>(*(SigmoidLayerParam*)&param);
        case LayerParam::PoolingLayer: return new PoolingLayer<T, A>(*(PoolingLayerParam*)&param);
        case LayerParam::ConvolutionLayer: return new ConvolutionLayer<T, A>(*(ConvolutionLayerParam*)&param);
        default:
            return NULL;
        }
    }

    SYNET_CLASS_INSTANCE(Layer);
}
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
#include "Synet/LrnLayer.h"
#include "Synet/ConcatLayer.h"

namespace Synet
{
    template <class T, template<class> class A> Layer<T, A>::Layer(const LayerParam & param)
        : _param(param)
    {
        _weight.resize(_param.data().size());
        for (size_t i = 0; i < _weight.size(); ++i)
            _weight[i].Reshape(_param.data()[i].dim());
    }

    template <class T, template<class> class A> bool Layer<T, A>::Load(const void * & data, size_t & size)
    {
        for (size_t i = 0; i < _weight.size(); ++i)
        {
            size_t requred = _weight[i].Size()*sizeof(Type);
            if (requred > size)
                return false;
            ::memcpy(_weight[i].Data(), data, requred);
            (char*&)data += requred;
            size -= requred;
        }
        return true;
    }

    template <class T, template<class> class A> bool Layer<T, A>::Load(std::istream & is)
    {
        for (size_t i = 0; i < _weight.size(); ++i)
            is.read((char*)_weight[i].Data(), _weight[i].Size()*sizeof(T));
        return true;
    }

    template <class T, template<class> class A> Layer<T, A> * Layer<T, A>::Create(const LayerParam & param)
    {
        switch (param.type())
        {
        case LayerTypeInput: return new InputLayer<T, A>(param);
        case LayerTypeInnerProduct: return new InnerProductLayer<T, A>(param);
        case LayerTypeRelu: return new ReluLayer<T, A>(param);
        case LayerTypeSigmoid: return new SigmoidLayer<T, A>(param);
        case LayerTypePooling: return new PoolingLayer<T, A>(param);
        case LayerTypeConvolution: return new ConvolutionLayer<T, A>(param);
        case LayerTypeLrn: return new LrnLayer<T, A>(param);
        case LayerTypeConcat: return new ConcatLayer<T, A>(param);
        default:
            return NULL;
        }
    }

    SYNET_CLASS_INSTANCE(Layer);
}
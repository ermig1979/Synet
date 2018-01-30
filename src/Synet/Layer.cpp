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
    template <class T, template<class> class A> bool Layer<T, A>::Load(const void * & data, size_t & size)
    {
        for (size_t i = 0; i < _tensors.size(); ++i)
        {
            size_t requred = _tensors[i]->Size()*sizeof(Type);
            if (requred > size)
                return false;
            ::memcpy(_tensors[i]->Data(), data, requred);
            (char*&)data += requred;
            size -= requred;
        }
        return true;
    }

    template <class T> inline void LoadValue(std::istream & is, T & value)
    {
        char buffer[64];
        is >> buffer;
        value = (T)::atof(buffer);
    }

    template <class T, template<class> class A> bool Layer<T, A>::Load(std::istream & is)
    {
        for (size_t i = 0; i < _tensors.size(); ++i)
        {
            T * data = _tensors[i]->Data();
            size_t size = _tensors[i]->Size();
            if (is.flags() & std::istream::binary)
            {
                is.read((char*)data, size*sizeof(T));
            }
            else
            {
                for (size_t j = 0; j < size; ++j)
                    LoadValue(is, data[j]);
            }
        }
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
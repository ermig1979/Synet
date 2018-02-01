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

#include "Synet/Network.h"

namespace Synet
{
    template <class T, template<class> class A> Network<T, A>::Network()
        : _empty(true)
    {
    }

    template <class T, template<class> class A> bool Network<T, A>::Load(const String & param, const String & weight)
    {
        if (!_param.Load(param))
            return false;

        _layers.clear();
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            LayerSharedPtr layer(Synet::Layer<T, A>::Create(_param().layers()[i]));
            if(layer)
                _layers.push_back(layer);
        }

        std::ifstream ifs(weight.c_str(), std::ifstream::binary);
        if (!ifs.is_open())
            return false;

        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(ifs))
            {
                ifs.close();
                return false;
            }
        }
        ifs.close();

        return Init();
    }

    template <class T, template<class> class A> bool Network<T, A>::Init()
    {
        _tensors.resize(_layers.size());
        for (size_t i = 0; i < _tensors.size(); ++i)
            _tensors[i].reset(new Tensor());

        _nameIndex.clear();
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            const String & name = _layers[i]->Param().name();
            if (_nameIndex.find(name) == _nameIndex.end())
                _nameIndex[name] = i;
            else
                assert(0);
        }

        for (size_t i = 0; i < _stages.size(); ++i)
            _stages[i].layer->Setup(_stages[i].src, _stages[i].dst);

        _empty = false;
        return true;
    }

    template <class T, template<class> class A> void Network<T, A>::Reshape()
    {
        for (size_t i = 0; i < _stages.size(); ++i)
            _stages[i].layer->Reshape(_stages[i].src, _stages[i].dst);
    }

    template <class T, template<class> class A> void Network<T, A>::Forward()
    {
        for (size_t i = 0; i < _stages.size(); ++i)
            _stages[i].layer->Forward(_stages[i].src, _stages[i].dst);
    }

    SYNET_CLASS_INSTANCE(Network);
}
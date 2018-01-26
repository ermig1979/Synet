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
    template <class T, template<class> class A> Network<T, A>::Network(const NetworkParam & param)
        : _param(param)
    {
        Init();
    }

    template <class T, template<class> class A> void Network<T, A>::Init()
    {
        _layers.clear();
        for (size_t i = 0; i < _param.layers().size(); ++i)
        {
            _layers.push_back(LayerSharedPtr(Synet::Layer<T, A>::Create(_param.layers()[i])));
        }
    }

    template <class T, template<class> class A> bool Network<T, A>::Load(const void * data, size_t size)
    {
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(data, size))
                return false;
        }
        return true;
    }

    template <class T, template<class> class A> bool Network<T, A>::Load(std::istream & is)
    {
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(is))
                return false;
        }
        return true;
    }

    template <class T, template<class> class A> bool Network<T, A>::Load(const String & path)
    {
        std::ifstream ifs(path.c_str());
        if (ifs.is_open())
        {
            bool result = Load(ifs);
            ifs.close();
            return result;
        }
        return false;
    }

    SYNET_CLASS_INSTANCE(Network);
}
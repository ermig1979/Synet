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

#pragma once

#include "Synet/Common.h"
#include "Synet/Tensor.h"
#include "Synet/Layer.h"

namespace Synet
{
    template <class T, template<class> class Allocator = std::allocator> class Network
    {
    public:
        typedef T Type;

        void Forward(const std::vector<Tensor<Type, Allocator>*> & src, const std::vector<Tensor<Type, Allocator>*> & dst)
        {

        }

        const String & Name() const { return _name; }

    protected:


    private:
        typedef Synet::Layer<Type, Allocator> Layer;
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        LayerSharedPtrs _layers;

        String _name;
    };
}
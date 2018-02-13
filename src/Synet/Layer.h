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
#include "Synet/Params.h"

namespace Synet
{
    template <class T, template<class> class A = std::allocator> class Layer
    {
    public:
        typedef T Type;
        typedef Synet::Tensor<T, A> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<Tensor*> TensorPtrs;

        Layer(const LayerParam & param)
            : _param(param)
        {
            _weight.resize(_param.weight().size());
            for (size_t i = 0; i < _weight.size(); ++i)
                _weight[i].Reshape(_param.weight()[i].dim());
        }

        const LayerParam & Param() const 
        { 
            return _param; 
        }

        const Tensors & Weight() const 
        { 
            return _weight; 
        }

        inline void Forward(const TensorPtrs & src, const TensorPtrs & dst)
        {
            ForwardCpu(src, dst);
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) = 0;

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst) = 0;

        bool Load(const void * & data, size_t & size)
        {
            for (size_t i = 0; i < _weight.size(); ++i)
            {
                size_t requred = _weight[i].Size() * sizeof(Type);
                if (requred > size)
                    return false;
                ::memcpy(_weight[i].Data(), data, requred);
                (char*&)data += requred;
                size -= requred;
            }
            return true;
        }

        bool Load(std::istream & is)
        {
            for (size_t i = 0; i < _weight.size(); ++i)
                is.read((char*)_weight[i].Data(), _weight[i].Size() * sizeof(T));
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst) = 0;

    private:
        LayerParam _param;
        Tensors _weight;
    };
}
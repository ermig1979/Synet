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

#include "Synet/LrnLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> void LrnLayer<T, A>::Setup(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        _normRegion = this->Param().lrn().normRegion();
        _size = this->Param().lrn().localSize();
        assert(_size%2 == 1);
        _prePad = (_size - 1) / 2;
        _alpha = this->Param().lrn().alpha();
        _beta = this->Param().lrn().beta();
        _k = this->Param().lrn().k();
        if (_normRegion == NormRegionTypeWithinChannel)
        {
            assert(0);
        }
    }

    template <class T, template<class> class A> void LrnLayer<T, A>::Reshape(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        assert(src[0]->Count() == 4);
        _num = src[0]->Axis(0);
        _channels = src[0]->Axis(1);
        _height = src[0]->Axis(2);
        _width = src[0]->Axis(3);
        switch (_normRegion)
        {
        case NormRegionTypeAcrossChannels:
            dst[0]->Reshape({ _num, _channels, _height, _width });
            _scale.Reshape({ _num, _channels, _height, _width });
            break;
        case NormRegionTypeWithinChannel:
            assert(0);
            break;
        default:
            assert(0);
            break;
        }
    }

    template <class T, template<class> class A> void LrnLayer<T, A>::ForwardCpu(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        switch (_normRegion)
        {
        case NormRegionTypeAcrossChannels:
            ForwardCpuCrossChannels(src, dst);
            break;
        case NormRegionTypeWithinChannel:
            assert(0);
            break;
        default:
            assert(0);
            break;
        }
    }

    template <class T, template<class> class A> void LrnLayer<T, A>::ForwardCpuCrossChannels(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        Tensor paddedSquare({1, _channels + _size - 1, _height, _width});
        CpuSet(paddedSquare.Size(), Type(0), paddedSquare.Data());
        Type alphaOverSize = _alpha / _size;
        for (size_t n = 0; n < _num; ++n)
        {
            CpuSqr<Type>(src[0]->Data({ n, 0, 0, 0 }), _channels * _height * _width, paddedSquare.Data({ 0, _prePad, 0, 0 }));
            for (size_t c = 0; c < _size; ++c) 
                CpuAxpy<Type>(paddedSquare.Data({ 0, c, 0, 0 }), _height * _width, alphaOverSize, _scale.Data({ n, 0, 0, 0 }));
            for (size_t c = 1; c < _channels; ++c)
            {
                CpuCopy(_scale.Data({ n, c - 1, 0, 0 }), _height * _width, _scale.Data({ n, c, 0, 0 }));
                CpuAxpy<Type>(paddedSquare.Data({ 0, c + _size - 1, 0, 0 }), _height * _width, alphaOverSize, _scale.Data({ n, c, 0, 0 }));
                CpuAxpy<Type>(paddedSquare.Data({ 0, c - 1, 0, 0 }), _height * _width, -alphaOverSize, _scale.Data({ n, c, 0, 0 }));
            }
        }
        CpuPow<Type>(_scale.Data(), _scale.Size(), -_beta, dst[0]->Data());
        CpuMul<Type>(src[0]->Data(), dst[0]->Data(), src[0]->Size(), dst[0]->Data());
    }

    SYNET_CLASS_INSTANCE(LrnLayer);
}
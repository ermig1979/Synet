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

#include "Synet/ConcatLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    template <class T, template<class> class A> void ConcatLayer<T, A>::Setup(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {

    }

    template <class T, template<class> class A> void ConcatLayer<T, A>::Reshape(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        _concatAxis = this->Param().concat().axis();
        _concatNum = src[0]->Size(0, _concatAxis);
        _concatInputSize = src[0]->Size(_concatAxis + 1);
        size_t srcSizeSum = src[0]->Size();
        Shape dstShape = src[0]->Shape();
        for (size_t i = 1; i < src.size(); ++i) 
        {
            assert(src[0]->Count() == src[i]->Count());
            for (size_t j = 0; j < src[0]->Count(); ++j)
            {
                if (_concatAxis)
                    continue;
                assert(dstShape[j] == src[i]->Axis(j));
            }
            srcSizeSum += src[i]->Size();
            dstShape[_concatAxis] += src[i]->Axis(_concatAxis);
        }
        dst[0]->Reshape(dstShape);
        assert(srcSizeSum == dst[0]->Size());
        if (src.size() == 1)
            dst[0]->Share(*src[0]);
    }

    template <class T, template<class> class A> void ConcatLayer<T, A>::ForwardCpu(const std::vector<Synet::Tensor<T, A>*> & src, const std::vector<Synet::Tensor<T, A>*> & dst)
    {
        if (src.size() == 1) 
            return; 

        Type * dstData = dst[0]->Data();
        size_t concatAxisOffset = 0;
        size_t dstConcatAxis = dst[0]->Axis(_concatAxis);
        for (size_t i = 0; i < src.size(); ++i) 
        {
            const Type * srcData = src[i]->Data();
            size_t srcConcatAxis = src[i]->Axis(_concatAxis);
            for (size_t n = 0; n < _concatNum; ++n)
                CpuCopy(srcData + n * srcConcatAxis * _concatInputSize, srcConcatAxis * _concatInputSize, dstData + (n * dstConcatAxis + concatAxisOffset) * _concatInputSize);
            concatAxisOffset += srcConcatAxis;
        }
    }

    SYNET_CLASS_INSTANCE(ConcatLayer);
}
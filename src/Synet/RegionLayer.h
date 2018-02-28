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
#include "Synet/Layer.h"
#include "Synet/SoftmaxLayer.h"
#include "Synet/Math.h"

namespace Synet
{
    namespace Detail
    {
        template<class T> void FlattenCpu(const T * src, size_t size, size_t layers, size_t batch, T * dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < layers; ++c)
                {
                    for (size_t i = 0; i < size; ++i)
                    {
                        size_t i1 = b*layers*size + c*size + i;
                        size_t i2 = b*layers*size + i*layers + c;
                        dst[i2] = src[i1];
                    }
                }
            }
        }
    }


    template <class T, template<class> class A> class RegionLayer : public Synet::Layer<T, A>
    {
    public:
        typedef T Type;
        typedef Layer<T, A> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        RegionLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & dst) 
        {
            const RegionParam & param = this->Param().region();
            _coords = param.coords();
            _classes = param.classes();
            _num = param.num();
            _softmax = param.softmax();
            _anchors.resize(param.anchors().size());
            for (size_t i = 0; i < param.anchors().size(); ++i)
                _anchors[i] = param.anchors()[i];
            _classfix = 0;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & dst)
        {
            assert(src[0]->Axis(1) == _num*(_coords + _classes + 1));
            dst[0]->Reshape(src[0]->Shape());
        }

        struct Region
        {
            Type x, y, w, h, v;
            size_t i;
        };
        typedef std::vector<Region> Regions;

        void GetRegions(const TensorPtrs & src, Type threshold, Type overlap, Regions & dst)
        {
            SYNET_PERF_FUNC();
            dst.clear();

            const Type * pPredict = src[0]->Data();
            size_t height = src[0]->Axis(2);
            size_t width = src[0]->Axis(3);
            size_t outputs = src[0]->Size(1);
            for (size_t i = 0; i < width*height; ++i) 
            {
                size_t row = i / height;
                size_t col = i % width;
                for (size_t n = 0; n < _num; ++n) 
                {
                    size_t index = i*_num + n;
                    size_t p_index = index * (_classes + 5) + 4;
                    Type scale = pPredict[p_index];
                    if (_classfix == -1 && scale < Type(0.5)) 
                        scale = Type(0);
                    size_t regionIndex = index * (_classes + 5);
                    Region r;
                    r.x = (col + CpuSigmoid(pPredict[regionIndex + 0])) / width;
                    r.y = (row + CpuSigmoid(pPredict[regionIndex + 1])) / height;
                    r.w = ::exp(pPredict[regionIndex + 2]) * _anchors[2 * n] / width;
                    r.h = ::exp(pPredict[regionIndex + 3]) * _anchors[2 * n + 1] / height;
                    size_t classIndex = index * (_classes + 5) + 5;
                    for (size_t j = 0; j < _classes; ++j) 
                    {
                        Type prob = scale*pPredict[classIndex + j];
                        if (prob > threshold)
                        {
                            r.v = prob;
                            r.i = j;
                            dst.push_back(r);
                        }
                        //probs[index][j] = (prob > threshold) ? prob : 0;
                    }
                }
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            size_t size = _coords + _classes + 1;
            size_t batch = src[0]->Axis(0);
            size_t height = src[0]->Axis(2);
            size_t width = src[0]->Axis(3);
            size_t outputs = src[0]->Size(1);
            T * pDst = dst[0]->Data();
            Detail::FlattenCpu(src[0]->Data(), width*height, size*_num, batch, pDst);


            for (size_t b = 0; b < batch; ++b) 
            {
                for (size_t i = 0; i < height*width*_num; ++i)
                {
                    size_t index = size*i + b*outputs;
                    CpuSigmoid(pDst + index + 4, 1, pDst + index + 4);
                }
            }

            if (_softmax) 
            {
                for (size_t b = 0; b < batch; ++b) 
                {
                    for (size_t i = 0; i < height*width*_num; ++i)
                    {
                        size_t index = size*i + b*outputs;
                        T buffer;
                        Detail::SoftmaxLayerForwardCpu(pDst + index + 5, _classes, 1, &buffer, pDst + index + 5);
                    }
                }
            }
        }

    private:
        typedef typename Base::Tensor Tensor;
        typedef std::vector<Type> Vector;

        size_t _coords, _classes, _num, _classfix;
        bool _softmax;
        Vector _anchors;

        SYNET_INLINE Type Overlap(Type x1, Type w1, Type x2, Type w2)
        {
            Type l1 = x1 - w1 / 2;
            Type l2 = x2 - w2 / 2;
            Type left = l1 > l2 ? l1 : l2;
            Type r1 = x1 + w1 / 2;
            Type r2 = x2 + w2 / 2;
            Type right = r1 < r2 ? r1 : r2;
            return right - left;
        }

        SYNET_INLINE Type RegionIntersection(const Region & a, const Region & b)
        {
            Type w = overlap(a.x, a.w, b.x, b.w);
            Type h = overlap(a.y, a.h, b.y, b.h);
            return (w < 0 || h < 0) ? 0 : w*a;
        }

        SYNET_INLINE Type RegionUnion(const Region & a, const Region & b)
        {
            Type i = RegionIntersection(a, b);
            return a.w*a.h + b.w*b.h - i;
        }

        //float box_iou(box a, box b)
        //{
        //    return box_intersection(a, b) / box_union(a, b);
        //}
    };
}
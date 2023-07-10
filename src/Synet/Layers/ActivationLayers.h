/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
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
#include "Synet/Utils/Activation.h"

namespace Synet
{
    class EluLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        EluLayer(const LayerParam& param, Context* context);

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _alpha;
    };

    //---------------------------------------------------------------------------------------------

    class GeluLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        GeluLayer(const LayerParam& param, Context* context);

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
    };

    //---------------------------------------------------------------------------------------------

    class HardSigmoidLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        HardSigmoidLayer(const LayerParam& param, Context* context);

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _scale, _shift;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class HswishLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        HswishLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            HswishParam hswish = this->Param().hswish();
            _shift = hswish.shift();
            _scale = hswish.scale();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            CpuHswish(src[0]->CpuData(), src[0]->Size(), _shift, _scale, dst[0]->CpuData());
        }

    private:
        Type _shift, _scale;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class MishLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        MishLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            _threshold = this->Param().softplus().threshold();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            CpuMish<Type>(src[0]->CpuData(), src[0]->Size(), _threshold, dst[0]->CpuData());
        }

    private:
        Type _threshold;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class ReluLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReluLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            _negativeSlope = this->Param().relu().negativeSlope();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            CpuRelu<Type>(src[0]->CpuData(), src[0]->Size(), _negativeSlope, dst[0]->CpuData());
        }

    private:
        Type _negativeSlope;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class RestrictRangeLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        RestrictRangeLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            const RestrictRangeParam& param = this->Param().restrictRange();
            _lower = param.lower();
            _upper = param.upper();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            CpuRestrictRange<Type>(src[0]->CpuData(), src[0]->Size(), _lower, _upper, dst[0]->CpuData());
        }

    private:
        Type _lower, _upper;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class SigmoidLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SigmoidLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            CpuSigmoid<Type>(src[0]->CpuData(), src[0]->Size(), dst[0]->CpuData());
        }

    private:
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class SoftplusLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SoftplusLayer(const LayerParam& param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            _beta = this->Param().softplus().beta();
            _threshold = this->Param().softplus().threshold();
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
        {
            CpuSoftplus<Type>(src[0]->CpuData(), src[0]->Size(), _beta, _threshold, dst[0]->CpuData());
        }

    private:
        Type _beta, _threshold;
    };

    //---------------------------------------------------------------------------------------------

    template <class T> class SwishLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SwishLayer(const LayerParam & param, Context* context)
            : Base(param, context)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            dst[0]->Reshape(src[0]->Shape(), src[0]->Format());
            this->UsePerfStat();
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            CpuSwish(src[0]->CpuData(), src[0]->Size(), dst[0]->CpuData());
        }
    };
}
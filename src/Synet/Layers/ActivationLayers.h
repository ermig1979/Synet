/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

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

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

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

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _scale, _shift;
    };

    //---------------------------------------------------------------------------------------------

    class HswishLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        HswishLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _shift, _scale;
    };

    //---------------------------------------------------------------------------------------------

    class MishLayer : public Synet::Layer<float>
    {
    public:

        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        MishLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _threshold;
    };

    //---------------------------------------------------------------------------------------------

    class ReluLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        ReluLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual bool Can16b() const;

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        TensorType _type;
        float _negativeSlope;
    };

    //---------------------------------------------------------------------------------------------

    class RestrictRangeLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        RestrictRangeLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _lower, _upper;
    };

    //---------------------------------------------------------------------------------------------

    class SigmoidLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SigmoidLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
    };

    //---------------------------------------------------------------------------------------------

    class SoftplusLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SoftplusLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
        float _beta, _threshold;
    };

    //---------------------------------------------------------------------------------------------

    class SwishLayer : public Synet::Layer<float>
    {
    public:
        typedef Layer<float> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        SwishLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        size_t _size;
    };
}
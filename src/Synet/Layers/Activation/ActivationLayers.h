/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar.
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
    class EluLayer : public Layer
    {
    public:
        EluLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual int64_t Flop() const;

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _alpha;
    };

    //---------------------------------------------------------------------------------------------

    class GeluLayer : public Layer
    {
    public:
        GeluLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
    };

    //---------------------------------------------------------------------------------------------

    class HardSigmoidLayer : public Layer
    {
    public:
        HardSigmoidLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _scale, _shift;
    };

    //---------------------------------------------------------------------------------------------

    class HswishLayer : public Layer
    {
    public:
        HswishLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _shift, _scale;
    };

    //---------------------------------------------------------------------------------------------

    class MishLayer : public Layer
    {
    public:
        MishLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual int64_t Flop() const;

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _threshold;
    };

    //---------------------------------------------------------------------------------------------

    class ReluLayer : public Layer
    {
    public:
        ReluLayer(const LayerParam& param, Context* context);

        virtual LowPrecisionType LowPrecision(TensorType type) const;

        virtual int64_t Flop() const;        
        
        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        TensorType _type;
        float _negativeSlope;
    };

    //---------------------------------------------------------------------------------------------

    class RestrictRangeLayer : public Layer
    {
    public:
        RestrictRangeLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _lower, _upper;
    };

    //---------------------------------------------------------------------------------------------

    class SigmoidLayer : public Layer
    {
    public:
        SigmoidLayer(const LayerParam& param, Context* context);

        virtual int64_t Flop() const;

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
    };

    //---------------------------------------------------------------------------------------------

    class SoftplusLayer : public Layer
    {
    public:
        SoftplusLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual int64_t Flop() const;

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
        float _beta, _threshold;
    };

    //---------------------------------------------------------------------------------------------

    class SwishLayer : public Layer
    {
    public:
        SwishLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual int64_t Flop() const;

    protected:
        virtual void Forward(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

    private:
        size_t _size;
    };
}
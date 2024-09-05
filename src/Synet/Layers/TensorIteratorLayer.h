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

#include "Synet/Layer.h"

namespace Synet
{
    class TensorIteratorLayer : public Layer
    {
    public:
        typedef std::shared_ptr<Tensor> TensorSharedPtr;
        typedef std::vector<TensorSharedPtr> TensorSharedPtrs;
        typedef Layer* LayerPtr;

        TensorIteratorLayer(const LayerParam& param, Context* context);

        virtual bool Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

        virtual void AddChild(const LayerSharedPtr& child);

        virtual int64_t Flop() const;

        virtual size_t MemoryUsage() const;

    protected:
        virtual void ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst);

    private:
        struct Stage
        {
            LayerPtr layer;
            TensorPtrs src;
            TensorPtrs buf;
            TensorPtrs dst;
        };
        typedef std::vector<Stage> Stages;
        typedef std::set<String> NameSet;
        typedef std::map<String, size_t> NameIdMap;
        typedef std::vector<ConnectionParam> ConnectionParams;
        typedef std::pair<size_t, size_t> Link;
        typedef std::vector<Link> Links;

        bool _empty;
        LayerSharedPtrs _layers;
        TensorSharedPtrs _tensors;
        Stages _input, _stages;
        TensorPtrs _src, _dst;
        NameIdMap _tensorId, _layerId;
        Link _itSrc, _itDst;
        Links _iLink, _oLink, _bLink;
        size_t _itCount, _srcAxis, _srcExt, _srcInt, _dstAxis, _dstExt, _dstInt;

        bool InitGraph(const TensorPtrs& buf);

        bool SetLinks(const TensorPtrs& src, const TensorPtrs& dst);
    };
}
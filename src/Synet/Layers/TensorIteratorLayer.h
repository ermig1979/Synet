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
#include "Synet/Fabric.h"

namespace Synet
{
    template <class T> class TensorIteratorLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::Tensor Tensor;
        typedef typename Base::TensorPtrs TensorPtrs;
        typedef std::shared_ptr<Tensor> TensorSharedPtr;
        typedef std::vector<TensorSharedPtr> TensorSharedPtrs;
        typedef Layer<T>* LayerPtr;
        typedef std::shared_ptr<Layer<T>> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        TensorIteratorLayer(const LayerParam & param, Context* context)
            : Base(param, context)
            , _empty(true)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_empty)
            {
                InitGraph(buf);
                SetLinks(src, dst);
                _empty = false;
            }

            for (size_t i = 0; i < _iLink.size(); ++i)
            {
                const Tensor * iSrc = src[_iLink[i].first];
                _src[_iLink[i].second]->Reshape(iSrc->Shape(), iSrc->Format());
            }
            const Tensor* itSrc = src[_itSrc.first];
            Shape srcShape = itSrc->Shape();
            _srcExt = itSrc->Size(0, _srcAxis);
            _itCount = srcShape[_srcAxis];
            _srcInt = itSrc->Size(_srcAxis + 1);
            srcShape[_srcAxis] = 1;
            _src[_itSrc.second]->Reshape(srcShape, itSrc->Format());

            for (size_t i = 0; i < _stages.size(); ++i)
                _stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst);

            for (size_t b = 0; b < _bLink.size(); ++b)
            {
                assert(_dst[_bLink[b].first]->Shape() == _src[_bLink[b].second]->Shape());
                if(_src[_bLink[b].second]->CpuData() != _dst[_bLink[b].first]->CpuData())
                    _src[_bLink[b].second]->Share(*_dst[_bLink[b].first]);
            }
            for (size_t o = 0; o < _oLink.size(); ++o)
            {
                const Tensor* oDst = src[_oLink[o].first];
                src[_oLink[o].second]->Reshape(oDst->Shape(), oDst->Format());
            }
            const Tensor* itDst = _dst[_itDst.first];
            Shape dstShape = itDst->Shape();
            assert(dstShape[_dstAxis] == 1);
            _dstExt = itDst->Size(0, _dstAxis);
            dstShape[_dstAxis] = _itCount;
            _dstInt = itDst->Size(_dstAxis + 1);
            dst[_itDst.second]->Reshape(dstShape, itDst->Format());

            std::stringstream desc;
            desc << _srcExt << "x" << _itCount << "x" << _srcInt << "-" << _dstInt;
            this->UsePerfStat(desc.str(), Flop());
        }

        virtual void AddChild(const LayerSharedPtr& child)
        {
            _layers.push_back(child);
        }

        virtual int64_t Flop() const
        {
            int64_t flop = 0;
            for (size_t i = 0; i < _layers.size(); ++i)
                flop += _layers[i]->Flop();
            return flop*_itCount;
        }

        virtual size_t MemoryUsage() const
        {
            std::set<const void*> unique;
            size_t memoryUsage = 0;
            for (size_t i = 0; i < _tensors.size(); ++i)
            {
                const void* ptr = _tensors[i]->RawCpuData();
                if (unique.find(ptr) == unique.end())
                {
                    memoryUsage += _tensors[i]->MemoryUsage();
                    unique.insert(ptr);
                }
            }
            return memoryUsage;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            for (size_t i = 0; i < _iLink.size(); ++i)
            {
                size_t size = src[_iLink[i].first]->Size();
                const T* pSrc = src[_iLink[i].first]->CpuData();
                T* pDst = _src[_iLink[i].second]->CpuData();
                memcpy(pDst, pSrc, size * sizeof(T));
            }
            const T* pSrcE = src[_itSrc.first]->CpuData();
            T* pSrcI = _src[_itSrc.second]->CpuData();
            const T* pDstI = _dst[_itDst.first]->CpuData();
            T* pDstE = dst[_itDst.second]->CpuData();
            for (size_t it = 0; it < _itCount; ++it)
            {
                for (size_t i = 0; i < _srcExt; ++i)
                    memcpy(pSrcI, pSrcE + (i * _itCount + it) * _srcInt, _srcInt * sizeof(T));

                for (size_t s = 0; s < _stages.size(); ++s)
                    _stages[s].layer->Forward(_stages[s].src, _stages[s].buf, _stages[s].dst);

                for (size_t b = 0; b < _bLink.size(); ++b)
                {
                    size_t size = _dst[_bLink[b].first]->Size();
                    const T* pSrc = _dst[_bLink[b].first]->CpuData();
                    T* pDst = _src[_bLink[b].second]->CpuData();
                    if(pSrc != pDst)
                        memcpy(pDst, pSrc, size * sizeof(T));
                }
                for (size_t o = 0; o < _dstExt; ++o)
                    memcpy(pDstE + (o * _itCount + it) * _dstInt, pDstI, _dstInt * sizeof(T));
            }
        }

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

        void InitGraph(const TensorPtrs& buf)
        {
            NameSet available;
            for (size_t i = 0; i < _layers.size(); ++i)
            {
                Stage stage;
                stage.layer = _layers[i].get();
                const LayerParam& param = stage.layer->Param();
                _layerId[param.name()] = i;
                for (size_t j = 0; j < param.src().size(); ++j)
                {
                    const String& name = param.src()[j];
                    if (_tensorId.find(name) != _tensorId.end())
                    {
                        stage.src.push_back(_tensors[_tensorId[name]].get());
                        if (available.find(name) != available.end())
                            available.erase(name);
                    }
                    else
                        assert(0);
                }
                for (size_t j = 0; j < param.dst().size(); ++j)
                {
                    const String& name = param.dst()[j];
                    if (j < param.src().size() && name == param.src()[j])
                    {
                        stage.dst.push_back(_tensors[_tensorId[name]].get());
                    }
                    else  if (_tensorId.find(name) != _tensorId.end())
                    {
                        assert(0);
                    }
                    else
                    {
                        TensorSharedPtr tensor(new Tensor());
                        tensor->SetName(name);
                        _tensorId[name] = _tensors.size();
                        _tensors.push_back(tensor);
                        stage.dst.push_back(tensor.get());
                    }
                    available.insert(name);
                    if (param.type() == LayerTypeInput || (param.type() == LayerTypeMeta && (param.meta().type() == MetaTypeInput || param.meta().type() == MetaTypeInputWithDefault)))
                    {
                        _src.push_back(_tensors.back().get());
                    }
                }
                stage.buf = buf;
                if (param.type() == LayerTypeInput || (param.type() == LayerTypeMeta && param.meta().type() == MetaTypeInput))
                    _input.push_back(stage);
                else
                    _stages.push_back(stage);
            }
            for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
                _dst.push_back(_tensors[_tensorId[*it]].get());
        }

        void SetLinks(const TensorPtrs& src, const TensorPtrs& dst)
        {
            const ConnectionParams& input = this->Param().tensorIterator().input();
            assert(src.size() == input.size() && src.size() == _src.size());
            _itSrc = Link(-1, -1);
            for (size_t i = 0, it = 0; i < input.size(); ++i)
            {
                assert(input[i].port() < src.size());
                Link link(input[i].port(), -1);
                for (size_t j = 0; j < _src.size(); ++j)
                {
                    if (_src[j]->Name() == input[i].dst())
                        link.second = j;
                }
                if (input[i].axis() != -1)
                {
                    _itSrc = link;
                    _srcAxis = input[i].axis();
                    assert(++it == 1);
                }
                else
                    _iLink.push_back(link);
            }
            assert(_itSrc.first < src.size() && _itSrc.second < _src.size());

            const ConnectionParams& output = this->Param().tensorIterator().output();
            assert(dst.size() == output.size() && dst.size() <= _dst.size());
            _itDst = Link(-1, -1);
            for (size_t o = 0, it = 0; o < output.size(); ++o)
            {
                assert(output[o].port() < dst.size());
                Link link(-1, output[o].port());
                for (size_t j = 0; j < _dst.size(); ++j)
                {
                    if (_dst[j]->Name() == output[o].src())
                        link.first = j;
                }
                if (output[o].axis() != -1)
                {
                    _itDst = link;
                    _dstAxis = output[o].axis();
                    assert(++it == 1);
                }
                else
                    _oLink.push_back(link);
            }
            assert(_itDst.first < _dst.size() && _itDst.second < src.size());

            const ConnectionParams& back = this->Param().tensorIterator().back();
            assert(back.size() <= _src.size());
            for (size_t b = 0; b < back.size(); ++b)
            {
                Link link(-1, -1);
                for (size_t i = 0; i < _dst.size(); ++i)
                {
                    if (_dst[i]->Name() == back[b].src())
                        link.first = i;
                }
                for (size_t o = 0; o < _src.size(); ++o)
                {
                    if (_src[o]->Name() == back[b].dst())
                        link.second = o;
                }
                assert(link.first < _dst.size() && link.second < _src.size());
                assert(link.first != _itDst.first && link.second != _itSrc.second);
                _bLink.push_back(link);
            }
        }
    };
}
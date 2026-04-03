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

#include "Synet/Layers/Recurrent/TensorIteratorLayer.h"
#include "Synet/Fabric.h"

namespace Synet
{
    TensorIteratorLayer::TensorIteratorLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
        , _empty(true)
    {
    }

    bool TensorIteratorLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (_empty)
        {
            _threads.resize(1);
            if (!InitGraph())
                return false;
            if (!SetLinks(src, dst))
                return true;
            _empty = false;
        }

        for (size_t i = 0; i < _iLink.size(); ++i)
        {
            const Tensor * iSrc = src[_iLink[i].first];
            _threads[0].src[_iLink[i].second]->Reshape(iSrc->GetType(), iSrc->Shape(), iSrc->Format());
        }
        const Tensor* itSrc = src[_itSrc.first];
        Shape srcShape = itSrc->Shape();
        _srcExt = itSrc->Size(0, _srcAxis);
        _itCount = srcShape[_srcAxis];
        _srcInt = itSrc->Size(_srcAxis + 1);
        srcShape[_srcAxis] = 1;
        _threads[0].src[_itSrc.second]->Reshape(itSrc->GetType(), srcShape, itSrc->Format());

        for (size_t s = 0; s < _threads[0].stages.size(); ++s)
        {
            Stage& stage = _threads[0].stages[s];
            stage.layer->Reshape(stage.src, buf, stage.dst);
        }

        for (size_t b = 0; b < _bLink.size(); ++b)
        {
            if(_threads[0].dst[_bLink[b].first]->Shape() != _threads[0].src[_bLink[b].second]->Shape())
                SYNET_ERROR("TensorIteratorLayer: incompatible back link tensors shapes!");
            if (_threads[0].dst[_bLink[b].first]->GetType() != _threads[0].src[_bLink[b].second]->GetType())
                SYNET_ERROR("TensorIteratorLayer: incompatible back link tensors types!");
            if(_threads[0].src[_bLink[b].second]->RawData() != _threads[0].dst[_bLink[b].first]->RawData())
                _threads[0].src[_bLink[b].second]->Share(*_threads[0].dst[_bLink[b].first]);
        }
        for (size_t o = 0; o < _oLink.size(); ++o)
        {
            const Tensor* oDst = src[_oLink[o].first];
            src[_oLink[o].second]->Reshape(oDst->GetType(), oDst->Shape(), oDst->Format());
        }
        const Tensor* itDst = _threads[0].dst[_itDst.first];
        Shape dstShape = itDst->Shape();
        if(dstShape[_dstAxis] != 1)
            SYNET_ERROR("TensorIteratorLayer: dstShape[_dstAxis] != 1 !");
        _dstExt = itDst->Size(0, _dstAxis);
        dstShape[_dstAxis] = _itCount;
        _dstInt = itDst->Size(_dstAxis + 1);
        dst[_itDst.second]->Reshape(itDst->GetType(), dstShape, itDst->Format());

        std::stringstream desc;
        desc << _srcExt << "x" << _itCount << "x" << _srcInt << "-" << _dstInt;
        this->UsePerfStat(desc.str(), Flop());

        return CloneThreadBuffers();
    }

    void TensorIteratorLayer::AddChild(const LayerSharedPtr& child)
    {
        _layers.push_back(child);
    }

    int64_t TensorIteratorLayer::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _layers.size(); ++i)
            flop += _layers[i]->Flop();
        return flop*_itCount;
    }

    size_t TensorIteratorLayer::MemoryUsage() const
    {
        std::set<const void*> unique;
        size_t memoryUsage = 0;
        for (size_t t = 0; t < _threads.size(); ++t)
        {
            const TensorSharedPtrs &tensors = _threads[t].tensors;
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                const void* ptr = tensors[i]->RawData();
                if (unique.find(ptr) == unique.end())
                {
                    memoryUsage += tensors[i]->MemoryUsage();
                    unique.insert(ptr);
                }
            }
        }
        return memoryUsage;
    }

    void TensorIteratorLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        for (size_t i = 0; i < _iLink.size(); ++i)
            memcpy(_threads[thread].src[_iLink[i].second]->RawData(), src[_iLink[i].first]->RawData(), src[_iLink[i].first]->RawSize());
        size_t eSrcE = src[_itSrc.first]->TypeSize();
        const uint8_t* pSrcE = src[_itSrc.first]->RawData();
        uint8_t* pSrcI = _threads[thread].src[_itSrc.second]->RawData();
        const uint8_t* pDstI = _threads[thread].dst[_itDst.first]->RawData();
        size_t eDstI = _threads[thread].dst[_itDst.first]->TypeSize();
        uint8_t* pDstE = dst[_itDst.second]->RawData();
        for (size_t it = 0; it < _itCount; ++it)
        {
            for (size_t i = 0; i < _srcExt; ++i)
                memcpy(pSrcI, pSrcE + (i * _itCount + it) * _srcInt * eSrcE, _srcInt * eSrcE);
            for (size_t s = 0; s < _threads[thread].stages.size(); ++s)
            {
                Stage& stage = _threads[thread].stages[s];
                stage.layer->ForwardPerf(stage.src, buf, stage.dst, thread);
            }
            for (size_t b = 0; b < _bLink.size(); ++b)
            {
                const uint8_t* pSrc = _threads[thread].dst[_bLink[b].first]->RawData();
                uint8_t* pDst = _threads[thread].src[_bLink[b].second]->RawData();
                if(pSrc != pDst)
                    memcpy(pDst, pSrc, _threads[thread].dst[_bLink[b].first]->RawSize());
            }
            for (size_t o = 0; o < _dstExt; ++o)
                memcpy(pDstE + (o * _itCount + it) * _dstInt * eDstI, pDstI, _dstInt * eDstI);
        }
    }

    bool TensorIteratorLayer::InitGraph()
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
                    stage.src.push_back(_threads[0].tensors[_tensorId[name]].get());
                    if (available.find(name) != available.end())
                        available.erase(name);
                }
                else
                    SYNET_ERROR("TensorIteratorLayer: can't find " << name << " layer in subgraph!");
            }
            for (size_t j = 0; j < param.dst().size(); ++j)
            {
                const String& name = param.dst()[j];
                if (j < param.src().size() && name == param.src()[j])
                {
                    stage.dst.push_back(_threads[0].tensors[_tensorId[name]].get());
                }
                else  if (_tensorId.find(name) != _tensorId.end())
                {
                    SYNET_ERROR("TensorIteratorLayer: find " << name << " unnecessary layer in subgraph!");
                }
                else
                {
                    TensorSharedPtr tensor(new Tensor());
                    tensor->SetName(name);
                    _tensorId[name] = _threads[0].tensors.size();
                    _threads[0].tensors.push_back(tensor);
                    stage.dst.push_back(tensor.get());
                }
                available.insert(name);
                if (param.type() == LayerTypeInput)
                    _threads[0].src.push_back(_threads[0].tensors.back().get());
            }
            //stage.buf = buf;
            if (param.type() == LayerTypeInput)
                _threads[0].input.push_back(stage);
            else
                _threads[0].stages.push_back(stage);
        }
        for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
            _threads[0].dst.push_back(_threads[0].tensors[_tensorId[*it]].get());
        return true;
    }

    bool TensorIteratorLayer::SetLinks(const TensorPtrs& src, const TensorPtrs& dst)
    {
        const ConnectionParams& input = this->Param().tensorIterator().input();
        if(src.size() != input.size() || src.size() != _threads[0].src.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().input() parameter!");
        _itSrc = Link(-1, -1);
        for (size_t i = 0, it = 0; i < input.size(); ++i)
        {
            if(input[i].port() >= src.size())
                SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().input()[" << i << "] parameter!");
            Link link(input[i].port(), -1);
            for (size_t j = 0; j < _threads[0].src.size(); ++j)
            {
                if (_threads[0].src[j]->Name() == input[i].dst())
                    link.second = j;
            }
            if (input[i].axis() != -1)
            {
                _itSrc = link;
                _srcAxis = input[i].axis();
                if(++it != 1)
                    SYNET_ERROR("TensorIteratorLayer: multiple input linking!");
            }
            else
                _iLink.push_back(link);
        }
        if (_itSrc.first >= src.size() || _itSrc.second >= _threads[0].src.size())
            SYNET_ERROR("TensorIteratorLayer can't correct link input!");

        const ConnectionParams& output = this->Param().tensorIterator().output();
        if (dst.size() != output.size() || dst.size() > _threads[0].dst.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().output() parameter!");
        _itDst = Link(-1, -1);
        for (size_t o = 0, it = 0; o < output.size(); ++o)
        {
            if (output[o].port() >= dst.size())
                SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().output()[" << o << "] parameter!");
            Link link(-1, output[o].port());
            for (size_t j = 0; j < _threads[0].dst.size(); ++j)
            {
                if (_threads[0].dst[j]->Name() == output[o].src())
                    link.first = j;
            }
            if (output[o].axis() != -1)
            {
                _itDst = link;
                _dstAxis = output[o].axis();
                if (++it != 1)
                    SYNET_ERROR("TensorIteratorLayer: multiple output linking!");
            }
            else
                _oLink.push_back(link);
        }
        if (_itDst.first >= _threads[0].dst.size() || _itDst.second >= src.size())
            SYNET_ERROR("TensorIteratorLayer can't correct link output!");

        const ConnectionParams& back = this->Param().tensorIterator().back();
        if(back.size() > _threads[0].src.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().back() parameter!");
        for (size_t b = 0; b < back.size(); ++b)
        {
            Link link(-1, -1);
            for (size_t i = 0; i < _threads[0].dst.size(); ++i)
            {
                if (_threads[0].dst[i]->Name() == back[b].src())
                    link.first = i;
            }
            for (size_t o = 0; o < _threads[0].src.size(); ++o)
            {
                if (_threads[0].src[o]->Name() == back[b].dst())
                    link.second = o;
            }
            if (link.first >= _threads[0].dst.size() || link.second >= _threads[0].src.size() ||
                link.first == _itDst.first || link.second == _itSrc.second)
                SYNET_ERROR("TensorIteratorLayer can't correct link back!");
            _bLink.push_back(link);
        }
        return true;
    }

    bool TensorIteratorLayer::CloneThreadBuffers()
    {
        size_t threads = this->Layer::Threads();
        if (threads < 1)
            return false;
        _threads.resize(threads);
        if (threads == 1)
            return true;
        typedef std::map<void*, size_t> PtrIdMap;
        PtrIdMap ptrs;
        for (size_t i = 0; i < _threads[0].tensors.size(); ++i)
        {
            void* ptr = _threads[0].tensors[i]->RawData();
            if (ptrs.find(ptr) == ptrs.end())
                ptrs[ptr] = i;
        }
        for (size_t t = 1; t < _threads.size(); ++t)
        {
            _threads[t].tensors.resize(_threads[0].tensors.size());
            for (size_t i = 0; i < _threads[0].tensors.size(); ++i)
            {
                void* ptr = _threads[0].tensors[i]->RawData();
                size_t idx = ptrs[ptr];
                TensorSharedPtr tensor(new Tensor());
                if (ptr)
                {
                    if (_threads[0].tensors[i]->Const())
                        tensor->Share(*_threads[0].tensors[i]);
                    else
                    {
                        if (idx == i)
                            tensor->Clone(*_threads[0].tensors[i]);
                        else
                        {
                            tensor->ShareAs(*_threads[t].tensors[idx], _threads[0].tensors[i]->Shape(), _threads[0].tensors[i]->Format());
                            tensor->SetName(_threads[0].tensors[i]->Name());
                        }
                    }
                }
                _threads[t].tensors[i] = tensor;
            }

            _threads[t].src.resize(_threads[0].src.size());
            for (size_t i = 0; i < _threads[0].src.size(); ++i)
            {
                size_t idx = _tensorId[_threads[0].src[i]->Name()];
                _threads[t].src[i] = _threads[t].tensors[idx].get();
            }

            _threads[t].dst.resize(_threads[0].dst.size());
            for (size_t i = 0; i < _threads[0].dst.size(); ++i)
            {
                size_t idx = _tensorId[_threads[0].dst[i]->Name()];
                _threads[t].dst[i] = _threads[t].tensors[idx].get();
            }

            _threads[t].input.resize(_threads[0].input.size());
            for (size_t i = 0; i < _threads[0].input.size(); ++i)
            {
                _threads[t].input[i].layer = _threads[0].input[i].layer;
                _threads[t].input[i].dst.resize(_threads[0].input[i].dst.size());
                for (size_t j = 0; j < _threads[0].input[i].dst.size(); ++j)
                {
                    if (_threads[0].input[i].dst[j]->Const())
                        _threads[t].input[i].dst[j] = _threads[0].input[i].dst[j];
                    else
                    {
                        size_t idx = _tensorId[_threads[0].input[i].dst[j]->Name()];
                        _threads[t].input[i].dst[j] = _threads[t].tensors[idx].get();
                    }
                }
            }

            _threads[t].stages.resize(_threads[0].stages.size());
            for (size_t i = 0; i < _threads[0].stages.size(); ++i)
            {
                _threads[t].stages[i].layer = _threads[0].stages[i].layer;
                _threads[t].stages[i].src.resize(_threads[0].stages[i].src.size());
                for (size_t j = 0; j < _threads[0].stages[i].src.size(); ++j)
                {
                    if (_threads[0].stages[i].src[j]->Const())
                        _threads[t].stages[i].src[j] = _threads[0].stages[i].src[j];
                    else
                    {
                        size_t idx = _tensorId[_threads[0].stages[i].src[j]->Name()];
                        _threads[t].stages[i].src[j] = _threads[t].tensors[idx].get();
                    }
                }
                _threads[t].stages[i].dst.resize(_threads[0].stages[i].dst.size());
                for (size_t j = 0; j < _threads[0].stages[i].dst.size(); ++j)
                {
                    if (_threads[0].stages[i].dst[j]->Const())
                        _threads[t].stages[i].dst[j] = _threads[0].stages[i].dst[j];
                    else
                    {
                        size_t idx = _tensorId[_threads[0].stages[i].dst[j]->Name()];
                        _threads[t].stages[i].dst[j] = _threads[t].tensors[idx].get();
                    }
                }
            }
        }
        return true;
    }
}
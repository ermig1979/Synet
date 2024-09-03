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

#include "Synet/Layers/TensorIteratorLayer.h"
#include "Synet/Fabric.h"

namespace Synet
{
    TensorIteratorLayer::TensorIteratorLayer(const LayerParam & param, Context* context)
        : Base(param, context)
        , _empty(true)
    {
    }

    bool TensorIteratorLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        if (_empty)
        {
            if (!InitGraph(buf))
                return false;
            if (!SetLinks(src, dst))
                return true;
            _empty = false;
        }

        for (size_t i = 0; i < _iLink.size(); ++i)
        {
            const Tensor * iSrc = src[_iLink[i].first];
            _src[_iLink[i].second]->Reshape(iSrc->GetType(), iSrc->Shape(), iSrc->Format());
        }
        const Tensor* itSrc = src[_itSrc.first];
        Shape srcShape = itSrc->Shape();
        _srcExt = itSrc->Size(0, _srcAxis);
        _itCount = srcShape[_srcAxis];
        _srcInt = itSrc->Size(_srcAxis + 1);
        srcShape[_srcAxis] = 1;
        _src[_itSrc.second]->Reshape(itSrc->GetType(), srcShape, itSrc->Format());

        for (size_t i = 0; i < _stages.size(); ++i)
            _stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst);

        for (size_t b = 0; b < _bLink.size(); ++b)
        {
            if(_dst[_bLink[b].first]->Shape() != _src[_bLink[b].second]->Shape())
                SYNET_ERROR("TensorIteratorLayer: incompatible back link tensors shapes!");
            if (_dst[_bLink[b].first]->GetType() != _src[_bLink[b].second]->GetType())
                SYNET_ERROR("TensorIteratorLayer: incompatible back link tensors types!");
            if(_src[_bLink[b].second]->RawData() != _dst[_bLink[b].first]->RawData())
                _src[_bLink[b].second]->Share(*_dst[_bLink[b].first]);
        }
        for (size_t o = 0; o < _oLink.size(); ++o)
        {
            const Tensor* oDst = src[_oLink[o].first];
            src[_oLink[o].second]->Reshape(oDst->GetType(), oDst->Shape(), oDst->Format());
        }
        const Tensor* itDst = _dst[_itDst.first];
        Shape dstShape = itDst->Shape();
        if(dstShape[_dstAxis] != 1)
            SYNET_ERROR("TensorIteratorLayer: dstShape[_dstAxis] != 1 !");
        _dstExt = itDst->Size(0, _dstAxis);
        dstShape[_dstAxis] = _itCount;
        _dstInt = itDst->Size(_dstAxis + 1);
        dst[_itDst.second]->Reshape(dstShape, itDst->Format());

        std::stringstream desc;
        desc << _srcExt << "x" << _itCount << "x" << _srcInt << "-" << _dstInt;
        this->UsePerfStat(desc.str(), Flop());
        return true;
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
        for (size_t i = 0; i < _tensors.size(); ++i)
        {
            const void* ptr = _tensors[i]->RawData();
            if (unique.find(ptr) == unique.end())
            {
                memoryUsage += _tensors[i]->MemoryUsage();
                unique.insert(ptr);
            }
        }
        return memoryUsage;
    }

    void TensorIteratorLayer::ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
    {
        for (size_t i = 0; i < _iLink.size(); ++i)
            memcpy(_src[_iLink[i].second]->RawData(), src[_iLink[i].first]->RawData(), src[_iLink[i].first]->RawSize());
        size_t eSrcE = src[_itSrc.first]->TypeSize();
        const uint8_t* pSrcE = src[_itSrc.first]->RawData();
        uint8_t* pSrcI = _src[_itSrc.second]->RawData();
        const uint8_t* pDstI = _dst[_itDst.first]->RawData();
        size_t eDstI = _dst[_itDst.first]->TypeSize();
        uint8_t* pDstE = dst[_itDst.second]->RawData();
        for (size_t it = 0; it < _itCount; ++it)
        {
            for (size_t i = 0; i < _srcExt; ++i)
                memcpy(pSrcI, pSrcE + (i * _itCount + it) * _srcInt * eSrcE, _srcInt * eSrcE);
            for (size_t s = 0; s < _stages.size(); ++s)
                _stages[s].layer->Forward(_stages[s].src, _stages[s].buf, _stages[s].dst);
            for (size_t b = 0; b < _bLink.size(); ++b)
            {
                const uint8_t* pSrc = _dst[_bLink[b].first]->RawData();
                uint8_t* pDst = _src[_bLink[b].second]->RawData();
                if(pSrc != pDst)
                    memcpy(pDst, pSrc, _dst[_bLink[b].first]->RawSize());
            }
            for (size_t o = 0; o < _dstExt; ++o)
                memcpy(pDstE + (o * _itCount + it) * _dstInt * eDstI, pDstI, _dstInt * eDstI);
        }
    }

    bool TensorIteratorLayer::InitGraph(const TensorPtrs& buf)
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
                    SYNET_ERROR("TensorIteratorLayer: can't find " << name << " layer in subgraph!");
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
                    SYNET_ERROR("TensorIteratorLayer: find " << name << " unnecessary layer in subgraph!");
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
                if (param.type() == LayerTypeInput)
                    _src.push_back(_tensors.back().get());
            }
            stage.buf = buf;
            if (param.type() == LayerTypeInput)
                _input.push_back(stage);
            else
                _stages.push_back(stage);
        }
        for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
            _dst.push_back(_tensors[_tensorId[*it]].get());
        return true;
    }

    bool TensorIteratorLayer::SetLinks(const TensorPtrs& src, const TensorPtrs& dst)
    {
        const ConnectionParams& input = this->Param().tensorIterator().input();
        if(src.size() != input.size() || src.size() != _src.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().input() parameter!");
        _itSrc = Link(-1, -1);
        for (size_t i = 0, it = 0; i < input.size(); ++i)
        {
            if(input[i].port() >= src.size())
                SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().input()[" << i << "] parameter!");
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
                if(++it != 1)
                    SYNET_ERROR("TensorIteratorLayer: multiple input linking!");
            }
            else
                _iLink.push_back(link);
        }
        if (_itSrc.first >= src.size() || _itSrc.second >= _src.size())
            SYNET_ERROR("TensorIteratorLayer can't correct link input!");

        const ConnectionParams& output = this->Param().tensorIterator().output();
        if (dst.size() != output.size() || dst.size() > _dst.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().output() parameter!");
        _itDst = Link(-1, -1);
        for (size_t o = 0, it = 0; o < output.size(); ++o)
        {
            if (output[o].port() >= dst.size())
                SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().output()[" << o << "] parameter!");
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
                if (++it != 1)
                    SYNET_ERROR("TensorIteratorLayer: multiple output linking!");
            }
            else
                _oLink.push_back(link);
        }
        if (_itDst.first >= _dst.size() || _itDst.second >= src.size())
            SYNET_ERROR("TensorIteratorLayer can't correct link output!");

        const ConnectionParams& back = this->Param().tensorIterator().back();
        if(back.size() > _src.size())
            SYNET_ERROR("TensorIteratorLayer has wrong tensorIterator().back() parameter!");
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
            if (link.first >= _dst.size() || link.second >= _src.size() ||
                link.first == _itDst.first || link.second == _itSrc.second)
                SYNET_ERROR("TensorIteratorLayer can't correct link back!");
            _bLink.push_back(link);
        }
        return true;
    }
}
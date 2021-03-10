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

        TensorIteratorLayer(const LayerParam & param)
            : Base(param)
            , _empty(true)
        {
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            if (_empty)
                Init(buf);
            assert(0);
        }

        virtual void AddChild(const LayerSharedPtr& child)
        {
            _layers.push_back(child);
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
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

        bool _empty;
        LayerSharedPtrs _layers;
        TensorSharedPtrs _tensors;
        Stages _input, _stages;
        TensorPtrs _src, _dst;
        NameIdMap _tensorId, _layerId;

        void Init(const TensorPtrs& buf)
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
            _empty = false;
        }
    };
}
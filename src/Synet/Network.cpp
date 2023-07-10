/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar,
*               2018-2021 Antonenka Mikhail.
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

#include "Synet/Fabric.h"
#include "Synet/Network.h"

#include "Synet/Utils/SetInput.h"
#include "Synet/Utils/Statistics.h"

namespace Synet
{
    Network::Network()
        : _empty(true)
    {
    }

    bool Network::Empty() const
    { 
        return _empty; 
    }

    const NetworkParam & Network::Param() const
    { 
        return _param(); 
    }

    void Network::Clear()
    {
        _param() = NetworkParam();
        _layers.clear();
        _tensors.clear();
        _stats.clear();
        _input.clear();
        _stages.clear();
        _stats.clear();
        _src.clear();
        _dst.clear();
        _back.clear();
        _tensorId.clear();
        _layerId.clear();
        _statId.clear();
        _srcIds.clear();
        _dstIds.clear();
        _empty = true;
    }

    bool Network::Load(const String & model, const String & weight, const Options & options)
    {
        Clear();

        if (!_param.Load(model))
        {
            std::cout << "Can't load model file '" << model << "' !" << std::endl;
            return false;
        }
        _context.options = options;
        CreateLayers();

        std::ifstream ifs(weight.c_str(), std::ifstream::binary);
        if (!ifs.is_open())
        {
            std::cout << "Can't open weight file '" << weight << "' !" << std::endl;
            return false;
        }
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(ifs, _layers))
            {
                std::cout << "Can't load weight from file '" << weight << "' !" << std::endl;
                ifs.close();
                return false;
            }
        }
        ifs.close();

        return Init();
    }

    bool Network::Load(const char * modelData, size_t modelSize, const char * weightData, size_t weightSize, const Options& options)
    {
        Clear();

        if (!_param.Load(modelData, modelSize, Cpl::ParamFormatXml))
            return false;
        _context.options = options;
        CreateLayers();

        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if (!_layers[i]->Load(weightData, weightSize, _layers))
                return false;
        }

        return Init();
    }

    bool Network::Save(const String& model) const
    {
        return _param.Save(model, false);
    }

    Network::TensorPtrs & Network::Src()
    { 
        return _src; 
    }

    const Network::TensorPtrs & Network::Src() const
    {
        return _src;
    }

    const Network::TensorPtrs & Network::Dst() const
    { 
        return _dst; 
    }

    const Network::Tensor * Network::Dst(const String & name) const
    {
        for (size_t i = 0; i < _dst.size(); ++i)
            if (_dst[i]->Name() == name)
                return _dst[i];
        return NULL;
    }

    Network::LayerPtrs Network::Back() const
    {
        return _back;
    }

    bool Network::Reshape(const Strings & srcNames, const Shapes & srcShapes, const Strings & dstNames)
    {
        if (srcNames.size() != srcShapes.size())
        {
            std::cout << "srcNames.size() != srcShapes.size() !" << std::endl;
            return false;
        }

        for (size_t i = 0; i < _tensors.size(); ++i)
            _tensors[i]->Clear(true);

        if (srcNames.size())
        {
            _src.clear();
            for (size_t i = 0; i < srcNames.size(); ++i)
            {
                bool found = false;
                for (size_t j = 0; j < _input.size(); ++j)
                {
                    const LayerParam & param = _input[j].layer->Param();
                    if (param.name() == srcNames[i])
                    {
                        if (param.type() == LayerTypeInput)
                        {
                            _input[j].dst[0]->Reshape(srcShapes[i], Type(0), param.input().shape()[0].format(), param.name());
                            _src.push_back(_input[j].dst[0]);
                        }
                        else if (param.type() == LayerTypeMeta && (param.meta().type() == MetaTypeInput || param.meta().type() == MetaTypeInputWithDefault))
                        {
                            Synet::Tensor<int32_t> & i32 = _input[j].dst[0]->As32i();
                            i32.Reshape({ srcShapes[j].size()});
                            for (size_t l = 0; l < srcShapes[j].size(); ++l)
                                i32.Data<int>()[l] = (int)srcShapes[j][l];
                        }
                        else
                            assert(0);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cout << "Input layer '" << srcNames[i] << "' is not found!" << std::endl;
                    return false;
                }
            }            
        }
        else
        {
            for (size_t i = 0; i < _input.size(); ++i)
                _input[i].layer->Reshape(_input[i].src, _input[i].buf, _input[i].dst);
        }

        if (!ReshapeStages())
            return false;

        if (dstNames.size())
        {
            _dst.clear();
            _back.clear();
            for (size_t i = 0; i < dstNames.size(); ++i)
            {
                bool found = false;
                for (size_t j = 0; j < _stages.size(); ++j)
                {
                    Layer * layer = _stages[j].layer;
                    const LayerParam & param = layer->Param();
                    if (param.name() == dstNames[i])
                    {
                        _dst.push_back(_stages[j].dst[0]);
                        layer->_isBack = true;
                        _back.push_back(layer);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    std::cout << "Output layer '" << dstNames[i] << "' is not found!" << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

    bool Network::Reshape(size_t width, size_t height, size_t batch)
    {
        if (_input.size() != 1)
            return false;
        const LayerParam & param = _input[0].layer->Param();
        if (param.type() != LayerTypeInput || param.input().shape().size() != 1)
            return false;
        const TensorFormat & format = param.input().shape()[0].format();
        Shape shape = param.input().shape()[0].dim();
        if (shape.size() != 4)
            return false;
        shape[0] = batch;
        if (format == TensorFormatNchw)
        {
            if ((shape[2] == -1 && shape[3] == -1) || Resizable())
            {
                shape[2] = height;
                shape[3] = width;
            }
            else if (shape[2] != height || shape[3] != width)
                return false;
        }
        else if (format == TensorFormatNhwc)
        {
            if ((shape[1] == -1 && shape[2] == -1) || Resizable())
            {
                shape[1] = height;
                shape[2] = width;
            }
            else if (shape[1] != height || shape[2] != width)
                return false;
        }
        else
            return false;
        _input[0].dst[0]->Reshape(shape, Type(0), format);
        return ReshapeStages();
    }

    bool Network::SetBatch(size_t batch)
    {
        if (_input.size() != 1)
            return false;
        const LayerParam& param = _input[0].layer->Param();
        if (param.type() != LayerTypeInput || param.input().shape().size() != 1)
            return false;
        const TensorFormat& format = param.input().shape()[0].format();
        Shape shape = param.input().shape()[0].dim();
        if (shape.size() < 2)
            return false;
        shape[0] = batch;
        _input[0].dst[0]->Reshape(shape, Type(0), format);
        ReshapeStages();
        return true;
    }

    bool Network::Dynamic() const
    {
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            const LayerParam& layer = _param().layers()[i];
            if (layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeInput)
                return true;
            if (layer.type() == LayerTypeInput)
            {
                if (layer.input().shape().empty())
                    return true;
                for (size_t j = 0; j < layer.input().shape().size(); ++j)
                    for (size_t k = 0; k < layer.input().shape()[j].dim().size(); ++k)
                        if (layer.input().shape()[j].dim()[k] == (size_t)-1)
                            return true;
            }
        }
        return false;
    }

    bool Network::Resizable() const
    {
        for (size_t i = 0; i < _layers.size(); ++i)
            if (!_layers[i]->Resizable())
                return false;
        return true;
    }

    Shape Network::NchwShape() const
    {
        assert(_src.size() >= 1);
        Shape shape = _src[0]->Shape();
        if (_src[0]->Format() == TensorFormatNhwc && shape.size() == 4)
            shape = Shape({ shape[0], shape[3] , shape[1] , shape[2] });
        return shape;
    }

#ifdef SYNET_SIMD_LIBRARY_ENABLE
    bool Network::SetInput(const View & view, float lower, float upper)
    {
        return Synet::SetInput(*this, Views({ view }), Floats({ lower }), Floats({ upper }));
    }

    bool Network::SetInput(const View & view, const Floats & lower, const Floats & upper)
    {
        return Synet::SetInput(*this, Views({ view }), lower, upper);
    }

    bool Network::SetInput(const Views & views, float lower, float upper)
    {
        return Synet::SetInput(*this, views, Floats({ lower }), Floats({ upper }));
    }

    bool Network::SetInput(const Views & views, const Floats & lower, const Floats & upper)
    {
        return Synet::SetInput(*this, views, lower, upper);
    }
#endif

    bool Network::GetMetaConst(const String & name, Tensor & value) const
    {
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            const LayerParam & layer = _param().layers()[i];
            if (layer.name() == name && layer.type() == LayerTypeMeta && layer.meta().type() == MetaTypeConst)
            {
                value.Import(layer.meta().alpha());
                return true;
            }
        }
        return false;
    }

    TensorFormat Network::Format() const
    {
        for (size_t i = 0; i < _input.size(); ++i)
        {
            const LayerParam & param = _input[i].layer->Param();
            if (param.type() == LayerTypeInput && param.input().shape().size())
                return param.input().shape()[0].format();
        }
        assert(0);
        return TensorFormatUnknown;
    }

    void Network::Forward()
    {
        //SYNET_PERF_FUNC();
        bool mode = GetFastMode();
        SetFastMode(true);
        for (size_t i = 0; i < _stages.size(); ++i)
        {
#if 0
            std::cout << _stages[i].layer->Param().name() << " : { ";
            if (_stages[i].src.size())
            {
                const Shape& shape = _stages[i].src[0]->Shape();
                for (size_t j = 0; j < shape.size(); ++j)
                    std::cout << shape[j] << " ";
            }
            std::cout << "}" << std::endl;
#endif
            _stages[i].layer->Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
        }
        SetFastMode(mode);
    }

    void Network::UpdateStatistics(float quantile, float epsilon)
    {
        SYNET_PERF_FUNC();
        for (size_t i = 0; i < _tensors.size(); ++i)
                UpdateStatistics(*_tensors[i], quantile, epsilon);
    }

    void Network::DebugPrint(std::ostream & os, int flag, int first, int last, int precision)
    {
        bool printOutput = (flag & (1 << DebugPrintOutput)) != 0;
        bool printLayerDst = (flag & (1 << DebugPrintLayerDst)) != 0;
        bool printLayerWeight = (flag & (1 << DebugPrintLayerWeight)) != 0;
        bool printInt8Buffers = (flag & (1 << DebugPrintInt8Buffers)) != 0;
        bool printLayerInternal = (flag & (1 << DebugPrintLayerInternal)) != 0;
        for (size_t i = 0; i < _input.size() && (printLayerDst || printOutput); ++i)
        {
            Layer & layer = *_input[i].layer;
            if ((layer._isBack && printOutput) || printLayerDst)
            {
                os << "Layer: " << layer.Param().name() << " : ";
                os << Cpl::ToStr(layer.Param().type()) << " ( ";
                for (size_t j = 0; j < layer.Param().src().size(); ++j)
                    os << layer.Param().src()[j] << " ";
                os << ")." << std::endl;
                for (size_t j = 0; j < _input[i].dst.size(); ++j)
                    _input[i].dst[j]->DebugPrint(os, String("dst[") + Cpl::ToStr(j) + "]", false, first, last, precision);
            }
        }
        for (size_t i = 0; i < _stages.size(); ++i)
        {
            Layer & layer = *_stages[i].layer;
            if ((layer._isBack && printOutput) || printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
            {
                if(printLayerDst || printLayerWeight || printInt8Buffers || printLayerInternal)
                    layer.Forward(_stages[i].src, _stages[i].buf, _stages[i].dst);
                os << "Layer: " << layer.Param().name() << " : ";
                os << Cpl::ToStr(layer.Param().type()) << " ( ";
                for (size_t j = 0; j < layer.Param().src().size(); ++j)
                    os << layer.Param().src()[j] << " ";
                os << ")." << std::endl;
                if (printLayerWeight)
                {
                    for (size_t j = 0; j < layer.Weight().size(); ++j)
                        layer.Weight()[j].DebugPrint(os, String("weight[") + Cpl::ToStr(j) + "]", true, first, last, precision);
                }
                if (printInt8Buffers && layer.Is8i())
                {
                    const Tensor& src = *_stages[i].src[0];
                    if (src.GetType() == TensorType32f)
                        _stages[i].buf[TensorType8u * BUFFER_COUNT + 1]->As8u().DebugPrint(os, src.Shape(), src.Format(), String("src"), false, first, last, precision);
                    const Tensor& dst = *_stages[i].dst[0];
                    _stages[i].buf[TensorType32i * BUFFER_COUNT + 0]->As32i().DebugPrint(os, dst.Shape(), dst.Format(), String("sum"), false, first, last, precision);
                }
                if (printLayerInternal)
                {
                    layer.DebugPrint(os, flag, first, last, precision);
                }
                if ((layer._isBack && printOutput) || printLayerDst)
                {
                    for (size_t j = 0; j < _stages[i].dst.size(); ++j)
                        _stages[i].dst[j]->DebugPrint(os, String("dst[") + Cpl::ToStr(j) + "]", false, first, last, precision);
                }
            }
        }
    }

    Network::Regions Network::GetRegions(size_t imageW, size_t imageH, Type threshold, Type overlap) const
    {
        const Shape & netNCHW = NchwShape();
        size_t netH = netNCHW[2];
        size_t netW = netNCHW[3];
        Regions regions;
        for (size_t i = 0; i < _dst.size(); ++i)
        {
            TensorPtrs dst(1, _dst[i]);
            const Layer * layer = _back[i];
            if (layer->Param().type() == Synet::LayerTypeStub && layer->Param().src().size() == 1)
            {
                for (size_t j = 0; j < _layers.size(); ++j)
                {
                    if (_layers[j]->Param().name() == layer->Param().src()[0])
                    {
                        layer = _layers[j].get();
                        break;
                    }
                }
            }
            Regions candidats;
            if (layer->Param().type() == Synet::LayerTypeYolo)
                ((YoloLayer<float>*)layer)->GetRegions(dst, netW, netH, threshold, candidats);
            if (layer->Param().type() == Synet::LayerTypeRegion)
                ((RegionLayer<float>*)layer)->GetRegions(dst, threshold, candidats);
            if (layer->Param().type() == Synet::LayerTypeDetectionOutput)
                ((DetectionOutputLayer<float>*)layer)->GetRegions(dst, threshold, candidats);
            for (size_t j = 0; j < candidats.size(); ++j)
            {
                Region & c = candidats[j];
                c.x *= imageW;
                c.w *= imageW;
                c.y *= imageH;
                c.h *= imageH;
                bool insert = true;
                for (size_t k = 0; k < regions.size(); ++k)
                {
                    Region & r = regions[k];
                    if (c.id == r.id && Overlap(c, r) >= overlap)
                    {
                        if (c.prob > r.prob)
                            r = c;
                        insert = false;
                        break;
                    }
                }
                if (insert)
                    regions.push_back(c);
            }
        }
        std::sort(regions.begin(), regions.end(), [](const Region & a, const Region & b) {return a.prob > b.prob; });
        return regions;
    }

    size_t Network::MemoryUsage() const
    {
        std::set<const void*> unique;
        size_t memoryUsage = 0;
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            for (size_t j = 0; j < _layers[i]->Weight().size(); ++j)
            {
                if (_layers[i]->Weight()[j].Size() == 0)
                    continue;
                const void * ptr = _layers[i]->Weight()[j].RawData();
                if (unique.find(ptr) == unique.end())
                {
                    memoryUsage += _layers[i]->Weight()[j].MemoryUsage();
                    unique.insert(ptr);
                }
            }
            memoryUsage += _layers[i]->MemoryUsage();
        }
        for (size_t i = 0; i < _tensors.size(); ++i)
        {
            const void * ptr = _tensors[i]->RawData();
            if (unique.find(ptr) == unique.end())
            {
                memoryUsage += _tensors[i]->MemoryUsage();
                unique.insert(ptr);
            }
        }
        for (size_t i = 0; i < _stats.size(); ++i)
            memoryUsage += _stats[i]->MemoryUsage();
        return memoryUsage;
    }

    void Network::CompactWeight()
    {
        for (size_t i = 0; i < _layers.size(); ++i)
            _layers[i]->CompactWeight();
    }

    int64_t Network::Flop() const
    {
        int64_t flop = 0;
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            if(_layers[i]->Param().parent().empty())
                flop += _layers[i]->Flop();
        }
        return flop;
    }

    bool Network::Is8i() const
    {
        return _param().quantization().method() != QuantizationMethodUnknown;
    }  

    const Network::Tensor* Network::GetInternalTensor(const String& name) const
    {
        NameIdMap::const_iterator it = _tensorId.find(name);
        if (it != _tensorId.end())
            return _tensors[it->second].get();
        return NULL;
    }

    void Network::CreateLayers()
    {
        NameIdMap layerId;
        for (size_t i = 0; i < _param().layers().size(); ++i)
        {
            const LayerParam& param = _param().layers()[i];
            LayerSharedPtr layer(Fabric<Type>::Create(param, &_context, _param().quantization().method()));
            if (layer)
            {
                layerId[param.name()] = _layers.size();
                _layers.push_back(layer);
                if (!param.parent().empty())
                {
                    assert(layerId.find(param.parent()) != layerId.end());
                    _layers[layerId[param.parent()]]->AddChild(layer);
                }
            }
        }
    }

    bool Network::Init()
    {
        TensorPtrs buf;
        SetBuffers(buf);
        SetStats();

        NameSet available;
        for (size_t i = 0; i < _layers.size(); ++i)
        {
            Stage stage;
            stage.layer = _layers[i].get();
            const LayerParam& param = stage.layer->Param();
            if (!param.parent().empty())
                continue;
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
            if (Is8i())
                stage.layer->SetStats(_stats);
            if (param.type() == LayerTypeInput || (param.type() == LayerTypeMeta && param.meta().type() == MetaTypeInput))
                _input.push_back(stage);
            else
            {
                for (size_t j = 0; j < param.src().size(); ++j)
                    _srcIds[param.src()[j]].insert(_stages.size());
                for (size_t j = 0; j < param.dst().size(); ++j)
                    _dstIds[param.dst()[j]].insert(_stages.size());
                _stages.push_back(stage);
            }
        }
        IdNameMap sorted;
        for (NameSet::const_iterator it = available.begin(); it != available.end(); ++it)
            sorted[_layerId[*it]] = *it;
        for (IdNameMap::const_iterator it = sorted.begin(); it != sorted.end(); ++it)
        {
            if (InsertDst(it->second))
            {
                LayerPtr layer = _layers[_layerId[it->second]].get();
                if (layer->Param().type() != LayerTypeMeta)
                {
                    _dst.push_back(_tensors[_tensorId[it->second]].get());
                    layer->_isBack = true;
                    _back.push_back(layer);
                }
            }
        }
        if (Is8i())
        {
            SetTensorTypes();
            UnifyStats();
        }
        if (!Dynamic())
            Reshape();
        _empty = false;
        return true;
    }

    bool Network::Is8iInSubGraph(const Stage & stage)
    {
        const Layer & layer = *stage.layer;
        if (layer._isBack)
            return false;
        const LayerParam & param = layer.Param();
        for (size_t d = 0; d < param.dst().size(); ++d)
        {
            const String & name = param.dst()[d];
            const IdSet & ids = _srcIds[name];
            for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
            {
                const Stage & dst = _stages[*id];
                if (&dst == &stage)
                    continue;
                if (dst.layer->Is8i())
                    continue;
                if (dst.layer->Param().type() == LayerTypePriorBox)
                    continue;
                if (dst.layer->Can8i() && Is8iInSubGraph(dst))
                    continue;
                return false;
            }
        }
        return true;
    }

    void Network::Set8iInSubGraph(const Stage & stage)
    {
        const LayerParam & param = stage.layer->Param();
        for (size_t d = 0; d < param.dst().size(); ++d)
        {
            const String & name = param.dst()[d];
            _tensors[_tensorId[name]]->SetType(TensorType8u);
            const IdSet & ids = _srcIds[name];
            for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
            {
                const Stage & dst = _stages[*id];
                if (&dst == &stage)
                    continue;
                if (dst.layer->Is8i())
                    continue;
                Set8iInSubGraph(dst);
            }
        }
    }

    void Network::SetTensorTypes()
    {
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            const LayerParam & param = _stages[s].layer->Param();
            if (param.type() == LayerTypeMeta)
                continue;
            for (size_t d = 0; d < param.dst().size(); ++d)
            {
                Tensor & tensor = *_tensors[_tensorId[param.dst()[d]]];
                if (tensor.GetType() == TensorTypeUnknown)
                    tensor.SetType(TensorType32f);
            }
        }
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            const Layer & layer = *_stages[s].layer;
            if (!layer.Is8i())
                continue;
            if (Is8iInSubGraph(_stages[s]))
                Set8iInSubGraph(_stages[s]);
        }
    }

    bool Network::IsSubGraphEndConv(size_t s)
    {
        const Stage& stage = _stages[s];
        const Layer & layer = *stage.layer;
        if (layer._isBack)
            return false;
        const LayerParam & param = layer.Param();
        for (size_t d = 0; d < param.dst().size(); ++d)
        {
            const String & name = param.dst()[d];
            const IdSet & ids = _srcIds[name];
            for (IdSet::const_iterator id = ids.begin(); id != ids.end(); ++id)
            {
                if (*id <= s)
                    continue;
                const Stage & dst = _stages[*id];
                const LayerParam & param = dst.layer->Param();
                if (param.type() == LayerTypeConvolution && param.convolution().group() != param.convolution().outputNum())
                    continue;
                if (param.type() == LayerTypeMergedConvolution)
                    continue;
                if (param.type() == LayerTypePriorBox)
                    continue;
                if (IsSubGraphEndConv(*id))
                    continue;
                return false;
            }
        }
        return true;
    }

    void Network::UnifyStats()
    {
        if (_param().quantization().method() == QuantizationMethodSymmetricNarrowed)
            return;
        for (size_t i = 0; i < _input.size(); ++i)
            _stats[_statId[_input[i].layer->Param().name()]]->Unify();
        for (size_t s = 0; s < _stages.size(); ++s)
        {
            if (IsSubGraphEndConv(s))
            {
                const LayerParam & param = _stages[s].layer->Param();
                if (param.type() == LayerTypeConvolution || param.type() == LayerTypeMergedConvolution || param.type() == LayerTypeScale)
                    _stats[_statId[param.dst()[0]]]->Unify();

                if (param.type() == LayerTypePooling && param.pooling().method() == PoolingMethodTypeMax && !_stats[_statId[param.src()[0]]]->channels)
                    _stats[_statId[param.dst()[0]]]->UnifyAs(*_stats[_statId[param.src()[0]]]);
                if (param.type() == LayerTypeRelu && param.relu().negativeSlope() == 0.0f)
                    _stats[_statId[param.dst()[0]]]->UnifyAs(*_stats[_statId[param.src()[0]]]);
                if (param.type() == LayerTypeConcat)
                {
                    StatPtrs stats;
                    for (size_t c = 0; c < param.src().size(); ++c)
                        stats.push_back(_stats[_statId[param.src()[c]]].get());
                    _stats[_statId[param.dst()[0]]]->UnifyAs(stats.data(), stats.size());
                }
            }
        }
    }

    bool Network::ReshapeStages()
    {
        for (size_t i = 0; i < _stages.size(); ++i)
        {
            const LayerParam& param = _stages[i].layer->Param();
            if (!_stages[i].layer->Reshape(_stages[i].src, _stages[i].buf, _stages[i].dst))
            {
                const LayerParam& param = _stages[i].layer->Param();
                SYNET_ERROR("Can't reshape " << i << " layer (name : " << Cpl::ToStr(param.name()) << ", type : "  << Cpl::ToStr(param.type()) << ") !");
            }
            if (_stages[i].layer->_isBack && param.type() != LayerTypeStub)
                _stages[i].dst[0]->SetName(param.name());
        }
        return true;
    }

    void Network::SetBuffers(TensorPtrs & buf)
    {
        for (TensorType type = TensorType32f; type <= TensorType8u; type = TensorType((int)type + 1))
        {
            for (int i = 0; i < BUFFER_COUNT; ++i)
            {
                TensorSharedPtr tensor(new Tensor());
                tensor->SetType(type);
                _tensors.push_back(tensor);
                buf.push_back(tensor.get());
            }
        }
    }

    void Network::SetStats()
    {
        for (size_t i = 0; i < _param().quantization().statistics().size(); ++i)
        {
            const StatisticParam & src = _param().quantization().statistics()[i];
            StatSharedPtr stat(new Stat(src));
            _statId[src.name()] = _stats.size();
            _stats.push_back(stat);
        }
    }

    void Network::UpdateStatistics(const Tensor & tensor, float quantile, float epsilon)
    {
        if (tensor.Name().empty() || tensor.GetType() != TensorType32f)
            return;
        size_t channels = GetChannels(tensor);
        if (channels == 0)
            return;
        size_t index = 0;
        for (; index < _param().quantization().statistics().size(); ++index)
            if (_param().quantization().statistics()[index].name() == tensor.Name())
                break;
        if (index == _param().quantization().statistics().size())
            _param().quantization().statistics().push_back(StatisticParam());
        StatisticParam & stat = _param().quantization().statistics()[index];
        if(stat.name().empty())
            stat.name() = tensor.Name();
        if (stat.min().empty())
            stat.min().resize(channels, FLT_MAX);
        if (stat.max().empty())
            stat.max().resize(channels, -FLT_MAX);
        UpdateChannelsQuantile(tensor, quantile, epsilon, stat.min().data(), stat.max().data());
    }

    bool Network::InsertDst(const String & name)
    {
        if (_param().dst().empty())
            return true;
        for (size_t i = 0; i < _param().dst().size(); ++i)
        {
            if (_param().dst()[i] == name)
                return true;
        }
        return false;
    }
}

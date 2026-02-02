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

#include "Synet/Layer.h"

namespace Synet
{
    bool Layer::SetStats(const StatSharedPtrs& stats)
    {
        bool result = true;
        if (LowPrecision(TensorType8u) == LowPrecisionTypeActive)
        {
            result = result && SetStats(stats, _param.src(), _stats[0]);
            result = result && SetStats(stats, _param.origin(), _stats[1]);
            result = result && SetStats(stats, _param.dst(), _stats[2]);
        }
        return result;
    }

    void Layer::ForwardPerf(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread)
    {
        if (_const)
            return;
        InitPerfStat(thread);
        SYNET_PERF_TEST(_perfComm[thread]);
        SYNET_PERF_TEST(_perfSpec[thread]);
        Forward(src, buf, dst, thread);
    }

    bool Layer::Load(std::istream& is, const LayerSharedPtrs& layers)
    {
        _weight.resize(_param.weight().size());
        for (size_t i = 0; i < _weight.size(); ++i)
        {
            const WeightParam& param = _param.weight()[i];
            Tensor& tensor = _weight[i];
            ptrdiff_t offset = param.offset();
            ptrdiff_t size = param.size();
            if (offset < 0 && size < 0)
            {
                Reshape(param, tensor);
                if (!is.read((char*)tensor.RawData(), tensor.RawSize()))
                    SYNET_ERROR("Can't load weight[" << i << "]!");
            }
            else
            {
                if (!ShareExisted(offset, layers, tensor))
                {
                    Reshape(param, tensor);
                    if (!(is.seekg(offset, std::ios::beg) && is.read((char*)tensor.RawData(), size)))
                        SYNET_ERROR("Can't load weight[" << i << "] of type " << Cpl::ToStr(param.type()) << " for offset " << offset << " and size " << size << " !");
                    if (param.scalar())
                        FillByFirstValue(tensor);
                }
            }
            tensor.SetConst(true);
        }
        return true;
    }

    bool Layer::Load(const char*& data, size_t& size, const LayerSharedPtrs& layers)
    {
        _weight.resize(_param.weight().size());
        for (size_t i = 0; i < _weight.size(); ++i)
        {
            const WeightParam& param = _param.weight()[i];
            Tensor& tensor = _weight[i];
            ptrdiff_t offset = param.offset();
            ptrdiff_t length = param.size();
            if (offset < 0 && length < 0)
            {
                Reshape(param, tensor);
                length = tensor.RawSize();
                if (length > (ptrdiff_t)size)
                    return false;
                memcpy(tensor.RawData(), data, length);
                data += length;
                size -= length;
            }
            else
            {
                if (!ShareExisted(offset, layers, tensor))
                {
                    if (offset + length > (ptrdiff_t)size)
                        return false;
                    Reshape(param, tensor);
                    memcpy(tensor.RawData(), data + offset, length);
                    if (param.scalar())
                        FillByFirstValue(tensor);
                }
            }
            tensor.SetConst(true);
        }
        return true;
    }

    void Layer::UsePerfStat(const String& desc, int64_t flop)
    {
        _perfInited.resize(_context->threads, 0);
        SYNET_PERF_VEC_RESZ(_perfComm, _context->threads, NULL);
        SYNET_PERF_VEC_RESZ(_perfSpec, _context->threads, NULL);
        if (_context->options.performanceLog < Options::PerfomanceLogSubnet && !_param.parent().empty())
            return;
        _perfEnable = true;
        _perfDesc = desc;
        _perfFlop = flop;
    }

    bool Layer::ShareExisted(size_t offset, const LayerSharedPtrs& layers, Tensor& tensor)
    {
        for (size_t j = 0; j < layers.size(); ++j)
        {
            if (layers[j].get() == this)
                break;
            for (size_t k = 0; k < layers[j]->Param().weight().size(); ++k)
            {
                if (layers[j]->Param().weight()[k].offset() == offset)
                {
                    tensor.Share(layers[j]->Weight()[k]);
                    return true;
                }
            }
        }
        return false;
    }

    void Layer::InitPerfStat(size_t thread)
    {
        if (_perfEnable && _perfInited[thread] == 0)
        {
            if (_context->options.performanceLog >= Options::PerfomanceLogLayer)
            {
                String type = Cpl::ToStr(_param.type());
                SYNET_PERF_INIT(_perfComm[thread], "void Synet::" + type + "Layer::Forward() {  " +
                    (LowPrecision(TensorType8u) == LowPrecisionTypeActive ? "int8" : LowPrecision(TensorType16b) == LowPrecisionTypeActive ? "bf16" : "fp32") + " } ", 0);
                if (_context->options.performanceLog >= Options::PerfomanceLogSize && _perfDesc.size())
                {
                    SYNET_PERF_INIT(_perfSpec[thread], "void Synet::" + type + "Layer::Forward() { " + _perfDesc + " } ", _perfFlop);
                }
            }
            _perfInited[thread] = 1;
        }
    }

    bool Layer::SetStats(const StatSharedPtrs& src, const Strings& names, StatPtrs& dst)
    {
        dst.clear();
        for (size_t i = 0; i < names.size(); ++i)
        {
            const String& name = names[i];
            size_t j = 0;
            for (; j < src.size(); ++j)
            {
                if (name == src[j]->name)
                {
                    dst.push_back(src[j].get());
                    break;
                }
            }
            if (j == src.size())
            {
                assert(0);
                return false;
            }
        }
        return true;
    }

    void Layer::Reshape(const WeightParam& param, Tensor& tensor) const
    {
        switch (param.type())
        {
        case TensorType32f: tensor.Reshape(TensorType32f, param.dim(), param.format(), 0.0f); break;
        case TensorType32i: tensor.Reshape(TensorType32i, param.dim(), param.format(), int32_t(0)); break;
        case TensorType8i: tensor.Reshape(TensorType8i, param.dim(), param.format(), int8_t(0)); break;
        case TensorType8u: tensor.Reshape(TensorType8u, param.dim(), param.format(), uint8_t(0)); break;
        case TensorType64i: tensor.Reshape(TensorType64i, param.dim(), param.format(), int64_t(0)); break;
        case TensorTypeBool: tensor.Reshape(TensorTypeBool, param.dim(), param.format(), false); break;
        default:
            assert(0);
        }
    }

    void Layer::FillByFirstValue(Tensor& tensor)
    {
        size_t unit = tensor.TypeSize(), size = tensor.RawSize();
        uint8_t* data = tensor.RawData();
        for (size_t i = unit; i < size; i += unit)
        {
            for (size_t j = 0; j < unit; ++j)
                data[i + j] = data[j];
        }
    }
}
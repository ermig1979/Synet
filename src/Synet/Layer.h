/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2021 Yermalayeu Ihar.
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
#include "Synet/Tensor.h"
#include "Synet/Params.h"
#include "Synet/Region.h"
#include "Synet/Context.h"
#include "Synet/Quantization/Stat.h"

namespace Synet
{
    template <class T> class Layer
    {
    public:
        typedef T Type;
        typedef Synet::Tensor<Type> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Tensor* TensorPtr;
        typedef std::vector<TensorPtr> TensorPtrs;
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        Layer(const LayerParam & param, Context * context)
            : _param(param)
            , _context(context)
            , _isBack(false)
            , _perfEnable(false)
            , _perfInited(false)
            , _perfFlop(0)
        {
            SYNET_PERF_SET(_perfComm, NULL);
            SYNET_PERF_SET(_perfSpec, NULL);
        }

        virtual ~Layer()
        {
        }

        const LayerParam & Param() const 
        { 
            return _param; 
        }

        const Tensors & Weight() const 
        { 
            return _weight; 
        }

        bool IsBack() const
        {
            return _isBack;
        }

        virtual size_t MemoryUsage() const
        {
            return 0;
        }

        virtual int64_t Flop() const
        {
            return 0;
        }

        virtual void CompactWeight()
        {
        }

        virtual bool Can8i() const
        {
            return false;
        }

        virtual bool Is8i() const
        {
            return false;
        }

        virtual bool HasZero() const
        {
            return false;
        }

        virtual bool Resizable() const
        {
            return true;
        }

        virtual void DebugPrint(std::ostream & os, int flag, int first, int last, int precision)
        {
        }

        virtual void AddChild(const LayerSharedPtr& child)
        {
            assert(0);
        }

        const StatPtrs & Stats(size_t index) const
        {
            assert(index < 3);
            return _stats[index];
        }

        bool SetStats(const StatSharedPtrs & stats)
        {
            bool result = true;
            if (Is8i())
            {
                result = result && SetStats(stats, _param.src(), _stats[0]);
                result = result && SetStats(stats, _param.origin(), _stats[1]);
                result = result && SetStats(stats, _param.dst(), _stats[2]);
            }
            return result;
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst) = 0;

        inline void Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            InitPerfStat();
            SYNET_PERF_TEST(_perfComm);
            SYNET_PERF_TEST(_perfSpec);
            ForwardCpu(src, buf, dst);
        }

        bool Load(std::istream & is, const LayerSharedPtrs & layers)
        {
            _weight.resize(_param.weight().size());
            for (size_t i = 0; i < _weight.size(); ++i)
            {
                const WeightParam & param = _param.weight()[i];
                Tensor & tensor = _weight[i];
                ptrdiff_t offset = param.offset();
                ptrdiff_t size = param.size();
                if (offset < 0 && size < 0)
                {
                    tensor.Reshape(param.dim(), Type(), param.format());
                    if (!is.read((char*)tensor.CpuData(), tensor.Size() * sizeof(T)))
                        return false;
                }
                else
                {
                    if (!ShareExisted(offset, layers, tensor))
                    {
                        tensor.Reshape(param.dim(), Type(), param.format());
                        if (!is.seekg(offset, std::ios::beg))
                            return false;
                        if (!is.read((char*)tensor.CpuData(), size))
                            return false;
                    }
                }
            }
            return true;
        }

        bool Load(const char * & data, size_t & size, const LayerSharedPtrs & layers)
        {
            _weight.resize(_param.weight().size());
            for (size_t i = 0; i < _weight.size(); ++i)
            {
                const WeightParam & param = _param.weight()[i];
                Tensor & tensor = _weight[i];
                ptrdiff_t offset = param.offset();
                ptrdiff_t length = param.size();
                if (offset < 0 && length < 0)
                {
                    tensor.Reshape(param.dim(), Type(), param.format());
                    length = tensor.Size() * sizeof(T);
                    if (length > size)
                        return false;
                    memcpy((char*)tensor.CpuData(), data, length);
                    data += length;
                    size -= length;
                }
                else
                {
                    if (!ShareExisted(offset, layers, tensor))
                    {
                        if (offset + length > size)
                            return false;
                        tensor.Reshape(param.dim(), Type(), param.format());
                        memcpy((char*)tensor.CpuData(), data + offset, length);
                    }
                }
            }
            return true;
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst) = 0;

        void UsePerfStat(const String & desc = "", int64_t flop = 0)
        {
            if (_context->options.performanceLog < Options::PerfomanceLogSubnet && !_param.parent().empty())
                return;
            _perfEnable = true;
            _perfDesc = desc;
            _perfFlop = flop;
        }

        static float * Buf32f(const TensorPtrs& buf, size_t idx)
        {
            return buf[TensorType32f * BUFFER_COUNT + idx]->As32f().CpuData();
        }

        static int32_t * Buf32i(const TensorPtrs& buf, size_t idx)
        {
            return buf[TensorType32i * BUFFER_COUNT + idx]->As32i().CpuData();
        }

        static uint8_t* Buf8u(const TensorPtrs& buf, size_t idx)
        {
            return buf[TensorType8u * BUFFER_COUNT + idx]->As8u().CpuData();
        }

        static void Extend32f(const TensorPtrs& buf, size_t idx, const Shape & shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType32f * BUFFER_COUNT + idx]->As32f().Extend(shape, format);
        }

        static void Extend32i(const TensorPtrs& buf, size_t idx, const Shape& shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType32i * BUFFER_COUNT + idx]->As32i().Extend(shape, format);
        }

        static void Extend8u(const TensorPtrs& buf, size_t idx, const Shape& shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType8u * BUFFER_COUNT + idx]->As8u().Extend(shape, format);
        }

    private:
        template<class U> friend class Network;

        const LayerParam & _param;
        Context* _context;
        Tensors _weight;
        StatPtrs _stats[3];
        bool _isBack;

        bool _perfEnable, _perfInited;
        String _perfDesc;
        int64_t _perfFlop;
        SYNET_PERF_DECL(_perfComm);
        SYNET_PERF_DECL(_perfSpec);

        bool ShareExisted(size_t offset, const LayerSharedPtrs& layers, Tensor& tensor)
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

        void InitPerfStat()
        {
            if (_perfEnable && !_perfInited)
            {
                if (_context->options.performanceLog >= Options::PerfomanceLogLayer)
                {
                    String type = Cpl::ToStr(_param.type());
                    SYNET_PERF_INIT(_perfComm, "void Synet::" + type + "Layer::Forward() {  " + (Is8i() ? "int8" : "fp32") + " } ", 0);
                    if (_context->options.performanceLog >= Options::PerfomanceLogSize && _perfDesc.size())
                    {
                        SYNET_PERF_INIT(_perfSpec, "void Synet::" + type + "Layer::Forward() { " + _perfDesc + " } ", _perfFlop);
                    }
                }
                _perfInited = true;
            }
        }

        bool SetStats(const StatSharedPtrs & src, const Strings & names, StatPtrs & dst)
        {
            dst.clear();
            for (size_t i = 0; i < names.size(); ++i)
            {
                const String & name = names[i];
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
    };
}
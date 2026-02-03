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
#include "Synet/Tensor.h"
#include "Synet/Params.h"
#include "Synet/Region.h"
#include "Synet/Context.h"
#include "Synet/Quantization/Stat.h"

namespace Synet
{
    class Layer
    {
    public:
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Tensor* TensorPtr;
        typedef std::vector<TensorPtr> TensorPtrs;
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        Layer(const LayerParam & param, Synet::Context * context)
            : _param(param)
            , _context(context)
            , _isBack(false)
            , _const(false)
            , _perfEnable(false)
            , _perfFlop(0)
        {
        }

        virtual ~Layer()
        {
            //CPL_LOG_SS(Info, "Delete layer of type " << Cpl::ToStr(_param.type()) << " and name " << _param.name() << " :");
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

        virtual LowPrecisionType LowPrecision(TensorType type) const
        {
            return LowPrecisionTypeNone;
        }

        virtual bool Resizable() const
        {
            return true;
        }

        bool Const() const
        {
            return _const;
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

        bool SetStats(const StatSharedPtrs& stats);

        virtual bool Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst) = 0;

        void ForwardPerf(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, size_t thread);

        bool Load(std::istream& is, const LayerSharedPtrs& layers);

        bool Load(const char*& data, size_t& size, const LayerSharedPtrs& layers);

    protected:
        virtual void Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread) = 0;

        void UsePerfStat(const String& desc = "", int64_t flop = 0);

        static float * Buf32f(const TensorPtrs& buf, size_t idx)
        {
            Synet::Tensor<float>* b = buf[TensorType32f * BUFFER_COUNT + idx];
            return b->Data<float>();
        }

        static int32_t * Buf32i(const TensorPtrs& buf, size_t idx)
        {
            Synet::Tensor<float>* b = buf[TensorType32i * BUFFER_COUNT + idx];
            return b->Data<int32_t>();
        }

        static uint8_t* Buf8u(const TensorPtrs& buf, size_t idx)
        {
            Synet::Tensor<float>* b = buf[TensorType8u * BUFFER_COUNT + idx];
            return b->Data<uint8_t>();
        }

        static void Extend32f(const TensorPtrs& buf, size_t idx, const Shape & shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType32f * BUFFER_COUNT + idx]->Extend(TensorType32f, shape, format);
        }

        static void Extend32i(const TensorPtrs& buf, size_t idx, const Shape& shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType32i * BUFFER_COUNT + idx]->Extend(TensorType32i, shape, format);
        }

        static void Extend8u(const TensorPtrs& buf, size_t idx, const Shape& shape, TensorFormat format = TensorFormatUnknown)
        {
            buf[TensorType8u * BUFFER_COUNT + idx]->Extend(TensorType8u, shape, format);
        }

        inline const Synet::Options& Options() const
        {
            return _context->options;
        }

        inline size_t TensorUsers(const String & name) const
        {
            return _context->tensorUsers[name];
        }

        inline size_t Threads() const
        {
            return _context->threads;
        }

    protected:
        bool _const;

    private:
        friend class Network;

        const LayerParam & _param;
        Synet::Context* _context;
        Tensors _weight;
        StatPtrs _stats[3];
        bool _isBack;

        bool _perfEnable;
        Ints _perfInited;
        String _perfDesc;
        int64_t _perfFlop;
        SYNET_PERF_VEC_DECL(_perfComm);
        SYNET_PERF_VEC_DECL(_perfSpec);

        bool ShareExisted(size_t offset, const LayerSharedPtrs& layers, Tensor& tensor);

        void InitPerfStat(size_t thread);

        bool SetStats(const StatSharedPtrs& src, const Strings& names, StatPtrs& dst);

        void Reshape(const WeightParam& param, Tensor& tensor) const;

        void FillByFirstValue(Tensor& tensor);
    };
}
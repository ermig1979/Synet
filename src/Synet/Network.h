/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2025 Yermalayeu Ihar,
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

#pragma once

#include "Synet/Layer.h"

namespace Synet
{
    class Network : public Deletable
    {
    public:
        typedef float Type;
        typedef Synet::Tensor<Type> Tensor;
        typedef std::vector<Tensor*> TensorPtrs;
        typedef Synet::Layer Layer;
        typedef Layer * LayerPtr;
        typedef std::vector<LayerPtr> LayerPtrs;
        typedef Synet::Region<Type> Region;
        typedef std::vector<Region> Regions;

        Network();

        virtual ~Network();

        bool Empty() const;

        const NetworkParam& Param() const;

        void Clear();

        bool Load(const String& model, const String& weight, const Options& options = Options(), size_t threads = 1);

        bool Load(const char* modelData, size_t modelSize, const char* weightData, size_t weightSize, const Options& options = Options(), size_t threads = 1);

        bool Save(const String& model) const;

        TensorPtrs& Src(size_t thread = 0);

        const TensorPtrs& Src(size_t thread = 0) const;

        const TensorPtrs& Dst(size_t thread = 0) const;

        const Tensor* Dst(const String& name, size_t thread = 0) const;

        LayerPtrs Back() const;

        bool Reshape(const Strings& srcNames = Strings(), const Shapes& srcShapes = Shapes(), const Strings& dstNames = Strings(), size_t threads = 1);

        bool Reshape(size_t width, size_t height, size_t batch = 1, size_t threads = 1);

        bool SetBatch(size_t batch, size_t threads = 1);

        bool SetThreads(size_t threads);

        bool Dynamic() const;

        bool Resizable() const;

        Shape NchwShape() const;

        size_t GetThreads() const;

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        bool SetInput(const View& view, float lower, float upper, bool rgb = false, size_t thread = 0);

        bool SetInput(const View& view, const Floats& lower, const Floats& upper, bool rgb = false, size_t thread = 0);

        bool SetInput(const Views& views, float lower, float upper, bool rgb = false, size_t thread = 0);

        bool SetInput(const Views& views, const Floats& lower, const Floats& upper, bool rgb = false, size_t thread = 0);
#endif

        bool GetMetaConst(const String& name, Tensor& value) const;

        TensorFormat Format() const;

        void Forward(size_t thread = 0);

        void UpdateStatistics(float quantile, float epsilon);

        void DebugPrint(std::ostream& os, int flag, int first, int last, int precision, size_t thread = 0);

        Regions GetRegions(size_t imageW, size_t imageH, Type threshold, Type overlap, size_t thread = 0) const;

        size_t MemoryUsage() const;

        void CompactWeight(bool unusedConst = true);

        int64_t Flop() const;

        bool Is8i() const;

        bool Is16b() const;

        const Tensor* GetInternalTensor(const String& name, size_t thread = 0) const;

    private:
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        typedef std::vector<Tensor> Tensors;
        typedef std::shared_ptr<Tensor> TensorSharedPtr;
        typedef std::vector<TensorSharedPtr> TensorSharedPtrs;

        typedef std::map<String, size_t> NameIdMap;
        typedef std::map<size_t, String> IdNameMap;
        typedef std::set<String> NameSet;
        typedef std::set<size_t> IdSet;
        typedef std::map<String, IdSet> NameIdSetMap;

        struct Stage
        {
            Layer * layer;
            TensorPtrs src;
            TensorPtrs buf;
            TensorPtrs dst;
        };
        typedef std::vector<Stage> Stages;

        struct Thread
        {
            TensorSharedPtrs tensors;
            Stages input, stages;
            TensorPtrs src, buf, dst;
        };
        typedef std::vector<Thread> Threads;

        bool _empty;
        NetworkParamHolder _param;
        Context _context;
        LayerSharedPtrs _layers;
        StatSharedPtrs _stats;

        LayerPtrs _back;
        NameIdMap _tensorId, _layerId, _statId;
        NameIdSetMap _srcIds, _dstIds;
        Threads _threads;

        bool CreateLayers();

        bool Init(size_t threads);

        void SetTensorType32f();

        bool ParseSubGraph(TensorType type, const Layer* layer) const;

        bool CanIgnoreInSubGraph(TensorType type, const Layer * layer, bool fromDst) const;

        bool IsLowPrecisionInSubGraph(TensorType type, size_t current, IdSet& visited, bool back);

        void SetLowPrecisionInSubGraph(TensorType type, size_t current, IdSet& visited, bool back);

        void SetLowPrecisionTensorType(TensorType type);

        bool IsSubGraphEndConv(size_t s);

        void UnifyStats();

        bool ReshapeStages();

        void SetBuffers();

        void SetStats();

        void UpdateStatistics(const Tensor& tensor, float quantile, float epsilon);

        bool InsertDst(const String& name);

        bool CloneThreadBuffers(size_t threads);
    };
}

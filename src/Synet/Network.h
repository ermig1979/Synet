/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar,
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
        typedef Synet::Layer<Type> Layer;
        typedef Layer * LayerPtr;
        typedef std::vector<LayerPtr> LayerPtrs;
        typedef Synet::Region<Type> Region;
        typedef std::vector<Region> Regions;

        Network();

        virtual ~Network();

        bool Empty() const;

        const NetworkParam& Param() const;

        void Clear();

        bool Load(const String& model, const String& weight, const Options& options = Options());

        bool Load(const char* modelData, size_t modelSize, const char* weightData, size_t weightSize, const Options& options = Options());

        bool Save(const String& model) const;

        TensorPtrs& Src();

        const TensorPtrs& Src() const;

        const TensorPtrs& Dst() const;

        const Tensor* Dst(const String& name) const;

        LayerPtrs Back() const;

        bool Reshape(const Strings& srcNames = Strings(), const Shapes& srcShapes = Shapes(), const Strings& dstNames = Strings());

        bool Reshape(size_t width, size_t height, size_t batch = 1);

        bool SetBatch(size_t batch);

        bool Dynamic() const;

        bool Resizable() const;

        Shape NchwShape() const;

#ifdef SYNET_SIMD_LIBRARY_ENABLE
        bool SetInput(const View& view, float lower, float upper);

        bool SetInput(const View& view, const Floats& lower, const Floats& upper);

        bool SetInput(const Views& views, float lower, float upper);

        bool SetInput(const Views& views, const Floats& lower, const Floats& upper);
#endif

        bool GetMetaConst(const String& name, Tensor& value) const;

        TensorFormat Format() const;

        void Forward();

        void UpdateStatistics(float quantile, float epsilon);

        void DebugPrint(std::ostream& os, int flag, int first, int last, int precision);

        Regions GetRegions(size_t imageW, size_t imageH, Type threshold, Type overlap) const;

        size_t MemoryUsage() const;

        void CompactWeight();

        int64_t Flop() const;

        bool Is8i() const;

        bool Is16b() const;

        const Tensor* GetInternalTensor(const String& name) const;

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

        bool _empty;
        NetworkParamHolder _param;
        Context _context;
        LayerSharedPtrs _layers;
        TensorSharedPtrs _tensors;
        StatSharedPtrs _stats;

        Stages _input, _stages;
        TensorPtrs _src, _dst;
        LayerPtrs _back;
        NameIdMap _tensorId, _layerId, _statId;
        NameIdSetMap _srcIds, _dstIds;

        bool CreateLayers();

        bool Init();

        void SetTensorType32f();

        bool Is8iInSubGraph(const Stage& stage);

        void Set8iInSubGraph(const Stage& stage);

        void SetTensorType8i();

        bool Is16bInSubGraph(const Stage& stage);

        void Set16bInSubGraph(const Stage& stage);

        void SetTensorType16b();

        bool IsSubGraphEndConv(size_t s);

        void UnifyStats();

        bool ReshapeStages();

        void SetBuffers(TensorPtrs& buf);

        void SetStats();

        void UpdateStatistics(const Tensor& tensor, float quantile, float epsilon);

        bool InsertDst(const String& name);
    };
}

/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2019 Yermalayeu Ihar.
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
#include "Synet/Utils/Convert.h"

namespace Synet
{
    template <class T> class Layer
    {
    public:
        typedef T Type;
        typedef Synet::Tensor<Type> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef std::vector<Tensor*> TensorPtrs;
        typedef std::shared_ptr<Layer> LayerSharedPtr;
        typedef std::vector<LayerSharedPtr> LayerSharedPtrs;

        Layer(const LayerParam & param)
            : _param(param)
            , _isBack(false)
            , _cvtSrc(false)
            , _cvtDst(false)
            , _src8u(false)
        {
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

        virtual size_t MemoryUsage() const
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

        virtual bool HasPad() const
        {
            return false;
        }

        const StatPtrs & Stats(size_t index) const
        {
            assert(index < 3);
            return _stats[index];
        }

        bool SetStats(const StatSharedPtrs & stats)
        {
            bool result = true;
            result = result && SetStats(stats, _param.src(), _stats[0], true);
            result = result && SetStats(stats, _param.origin(), _stats[1], false);
            result = result && SetStats(stats, _param.dst(), _stats[2], false);
            return result;
        }

        inline void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, const TensorPtrs & f2i, const TensorPtrs & i2f)
        {
            _src = src, _buf = buf, _dst = dst, _f2i = f2i, _i2f = i2f;
            _src8u = src.size() && src[0]->GetType() == TensorType8u;
            if (_isBack)
            {
                if (Is8i())
                {
                    _cvtSrc = !_src8u;
                    _cvtDst = true;
                }
                else
                {
                    _cvtSrc = _src8u;
                    _cvtDst = false;
                }
            }
            else
            {
                _cvtSrc = _src8u ? !Can8i() : Is8i();
                _cvtDst = false;
            }
            if (_cvtDst)
            {
                if (_cvtSrc)
                {
                    _f2i.resize(_src.size());
                    _i2f.resize(_dst.size());
                    Prepare32fTo8u(_src, _stats[0], _f2i, _f2iCvt);
                    Reshape(_f2i, _buf, _i2f);
                    Prepare8uTo32f(_i2f, _stats[2], _dst, _i2fCvt);
                }
                else
                {
                    _i2f.resize(_dst.size());
                    Reshape(_src, _buf, _i2f);
                    Prepare8uTo32f(_i2f, _stats[2], _dst, _i2fCvt);
                }
            }
            else
            {
                if (_cvtSrc)
                {
                    if (_src8u)
                    {
                        _i2f.resize(_src.size());
                        Prepare8uTo32f(_src, _stats[0], _i2f, _i2fCvt);
                        Reshape(_i2f, _buf, _dst);
                    }
                    else
                    {
                        _f2i.resize(_src.size());
                        Prepare32fTo8u(_src, _stats[0], _f2i, _f2iCvt);
                        Reshape(_f2i, _buf, _dst);
                    }
                }
                else
                    Reshape(_src, _buf, _dst);
            }
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst) = 0;

        inline void Forward()
        {
            if (_cvtDst)
            {
                if (_cvtSrc)
                {
                    Convert32fTo8u(_src, _f2iCvt, _f2i);
                    ForwardCpu(_f2i, _buf, _i2f);
                    Convert8uTo32f(_i2f, _i2fCvt, _dst);
                }
                else
                {
                    ForwardCpu(_src, _buf, _i2f);
                    Convert8uTo32f(_i2f, _i2fCvt, _dst);
                }
            }
            else
            {
                if (_cvtSrc)
                {
                    if (_src8u)
                    {
                        Convert8uTo32f(_src, _i2fCvt, _i2f);
                        ForwardCpu(_i2f, _buf, _dst);
                    }
                    else
                    {
                        Convert32fTo8u(_src, _f2iCvt, _f2i);
                        ForwardCpu(_f2i, _buf, _dst);
                    }
                }
                else
                    ForwardCpu(_src, _buf, _dst);
            }
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
                    bool unique = true;
                    for (size_t j = 0; j < layers.size() && unique; ++j)
                    {
                        if (layers[j].get() == this)
                            break;
                        for (size_t k = 0; k < layers[j]->Param().weight().size() && unique; ++k)
                        {
                            if (layers[j]->Param().weight()[k].offset() == offset)
                            {
                                tensor.Share(layers[j]->Weight()[k]);
                                unique = false;
                            }
                        }
                    }
                    if (unique)
                    {
                        tensor.Reshape(param.dim(), Type(), param.format());
                        is.seekg(offset, std::ios::beg);
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
                    bool unique = true;
                    for (size_t j = 0; j < layers.size() && unique; ++j)
                    {
                        if (layers[j].get() == this)
                            break;
                        for (size_t k = 0; k < layers[j]->Param().weight().size() && unique; ++k)
                        {
                            if (layers[j]->Param().weight()[k].offset() == offset)
                            {
                                tensor.Share(layers[j]->Weight()[k]);
                                unique = false;
                            }
                        }
                    }
                    if (unique)
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

    private:
        template<class U> friend class Network;

        struct CvtParam
        {
            size_t batch, channels, spatial;
            TensorFormat format;
            Floats scale, shift;
        };
        typedef std::vector<CvtParam> CvtParams;

        const LayerParam & _param;
        Tensors _weight;
        StatPtrs _stats[3];
        bool _isBack, _cvtSrc, _cvtDst, _src8u;
        TensorPtrs _src, _buf, _dst, _f2i, _i2f;
        CvtParams _f2iCvt, _i2fCvt;

        bool SetStats(const StatSharedPtrs & src, const Strings & names, StatPtrs & dst, bool isSrc)
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
                        if (isSrc && HasPad())
                        {
                            Stat & stat = *dst.back();
                            for (size_t i = 0; i < stat.min.size(); ++i)
                            {
                                stat.min[i] = std::min(0.0f, stat.min[i]);
                                stat.max[i] = std::max(0.0f, stat.max[i]);
                            }
                        }
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

        static void InitCvtSize(const Tensor & src, const Stat & stat, CvtParam & param)
        {
            assert(stat.min.size() == stat.max.size());
            param.batch = src.Count() > 1 ? src.Axis(0) : 1;
            param.channels = stat.min.size();
            param.spatial = src.Size() / param.batch / param.channels;
            param.scale.resize(param.channels);
            param.shift.resize(param.channels);
            param.format = src.Format();
        }

        static void InitCvt32fTo8u(const Stat & stat, CvtParam & param)
        {
            for (int i = 0; i < param.channels; ++i)
            {
                param.scale[i] = 255.0f / (stat.max[i] - stat.min[i]);
                param.shift[i] = - stat.min[i] * param.scale[i];
            }
        }

        static void Prepare32fTo8u(const TensorPtrs & src, const StatPtrs & stats, const TensorPtrs & dst, CvtParams & params)
        {
            assert(src.size() == stats.size() && src.size() == dst.size());
            params.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                dst[i]->As8u().Reshape(src[i]->Shape(), src[i]->Format());
                InitCvtSize(*src[i], *stats[i], params[i]);
                InitCvt32fTo8u(*stats[i], params[i]);
            }
        }

        static void Convert32fTo8u(const TensorPtrs & src, const CvtParams & params, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            assert(src.size() == params.size() && src.size() == dst.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                const float * pSrc = src[i]->As32f().CpuData();
                uint8_t * pDst = dst[i]->As8u().CpuData();
                const CvtParam & p = params[i];
                for (size_t b = 0; b < p.batch; ++b)
                {
                    Synet::Convert32fTo8u(pSrc, p.channels, p.spatial, p.format, p.scale.data(), p.shift.data(), pDst);
                    pSrc += p.channels*p.spatial;
                    pDst += p.channels*p.spatial;
                }
            }
        }

        static void InitCvt8uTo32f(const Stat & stat, CvtParam & param)
        {
            for (int i = 0; i < param.channels; ++i)
            {
                param.scale[i] = (stat.max[i] - stat.min[i]) / 255.0f;
                param.shift[i] = stat.min[i];
            }
        }

        static void Prepare8uTo32f(const TensorPtrs & src, const StatPtrs & stats, const TensorPtrs & dst, CvtParams & params)
        {
            assert(src.size() == stats.size() && src.size() == dst.size());
            params.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                dst[i]->As32f().Reshape(src[i]->Shape(), src[i]->Format());
                InitCvtSize(*src[i], *stats[i], params[i]);
                InitCvt8uTo32f(*stats[i], params[i]);
            }
        }

        static void Convert8uTo32f(const TensorPtrs & src, const CvtParams & params, const TensorPtrs & dst)
        {
            SYNET_PERF_FUNC();

            assert(src.size() == params.size() && src.size() == dst.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                const uint8_t * pSrc = src[i]->As8u().CpuData();
                float * pDst = dst[i]->As32f().CpuData();
                const CvtParam & p = params[i];
                for (size_t b = 0; b < p.batch; ++b)
                {
                    Synet::Convert8uTo32f(pSrc, p.channels, p.spatial, p.format, p.scale.data(), p.shift.data(), pDst);
                    pSrc += p.channels*p.spatial;
                    pDst += p.channels*p.spatial;
                }
            }
        }
    };
}
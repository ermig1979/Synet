/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2026 Yermalayeu Ihar.
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

#include "Synet/Layers/Statistics/PoolingStatisticsLayer.h"

namespace Synet
{
    void PoolingStatistics(const float * src, size_t channels, size_t spatial, TensorFormat format, const float & alpha, float * dst)
    {
        memset(dst, 0, channels * sizeof(float) * 2);
        float* sum = dst, *sqs = sum + channels;
        if (format == TensorFormatNhwc)
        {
            for (size_t s = 0; s < spatial; ++s)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float val = src[c];
                    sum[c] += val;
                    sqs[c] += val * val;
                }
                src += channels;
            }
        }
        else if (format == TensorFormatNchw)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    float val = src[s];
                    sum[c] += val;
                    sqs[c] += val * val;
                }
                src += spatial;
            }
        }
        else
            assert(0);
        float k = 1.0f / float(spatial);
        for (size_t c = 0; c < channels; ++c)
        {
            sum[c] = sum[c] * k;
            sqs[c] = ::sqrt(sqs[c] * k - sum[c] * sum[c] + alpha);
        }
    }

    //-------------------------------------------------------------------------------------------------

    PoolingStatisticsLayer::PoolingStatisticsLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    int64_t PoolingStatisticsLayer::Flop() const
    {
        return _batch * _channels * _spatial * 3;
    }
        
    bool PoolingStatisticsLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst, bool init)
    {
        if (src.size() != 1 || dst.size() != 1)
            SYNET_ERROR("PoolingStatisticsLayer supports only 1 input and 1 output!");
        _type = src[0]->GetType();
        _format = src[0]->Format();
        if (_type != TensorType32f)
            SYNET_ERROR("PoolingStatisticsLayer supports only FP32 input tensor!");
        if (src[0]->Count() != 4)
            SYNET_ERROR("PoolingStatisticsLayer supports only 4D input tensor!");

        const SharedParam & param = this->Param().shared();
        if(param.floats().empty())
            SYNET_ERROR("PoolingStatisticsLayer alpha parameter is not set!");
        _alpha = param.floats()[0];
        _batch = src[0]->Axis(0);
        if (_format == TensorFormatNchw)
        {
            _channels = src[0]->Axis(1);
            _spatial = src[0]->Axis(2) * src[0]->Axis(3);
        }
        else if (_format == TensorFormatNhwc)
        {
            _channels = src[0]->Axis(3);
            _spatial = src[0]->Axis(1) * src[0]->Axis(2);
        }
        else
            SYNET_ERROR("PoolingStatisticsLayer unknown input tensor format!");

        dst[0]->Reshape(_type, Shp(_batch, 2 * _channels), _format);

        if (src[0]->Const())
        {
            Forward(src, buf, dst, 0);
            dst[0]->SetConst(true);
            _const = true;
        }
        else
        {
            this->UsePerfStat();
            _const = false;
        }

        return true;
    }

    void PoolingStatisticsLayer::Forward(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst, size_t thread)
    {
        for (size_t b = 0; b < _batch; ++b)
            PoolingStatistics(src[0]->Data<float>(Shp(b, 0, 0, 0)), _channels, _spatial, _format, _alpha, dst[0]->Data<float>(Shp(b, 0)));
    }
}
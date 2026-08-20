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

#pragma once

#include "Synet/Network.h"
#include "Synet/Decoders/Anchor.h"
#include "Synet/Decoders/Ultraface.h"
#include "Synet/Decoders/YoloV5.h"
#include "Synet/Decoders/YoloV7.h"
#include "Synet/Decoders/YoloV8.h"
#include "Synet/Decoders/Iim.h"
#include "Synet/Decoders/Region.h"
#include "Synet/Decoders/Rtdetr.h"
#include "Synet/Decoders/RtdetrV2.h"
#include "Synet/Decoders/DetOut.h"
#include "Synet/Decoders/Yolo.h"
#include "Synet/Decoders/Scrfd.h"
#include "Synet/Decoders/ScrfdV2.h"
#include "Synet/Decoders/Alpha.h"
#include "Synet/Decoders/Nanodet.h"
#include "Synet/Decoders/YoloV11.h"
#include "Synet/Decoders/Ssd.h"

namespace Synet
{
    struct DetectionParam
    {
        CPL_PARAM_VALUE(float, confidence, 0.5f);
        CPL_PARAM_VALUE(float, overlap, 0.5f);
        CPL_PARAM_VALUE(String, decoder, String());
        CPL_PARAM_STRUCT_MOD(AnchorParam, epsilon, GetEpsilonParam());
        CPL_PARAM_STRUCT_MOD(AnchorParam, retina, GetRetinaParam());
        CPL_PARAM_STRUCT(UltrafaceParam, ultraface);
        CPL_PARAM_STRUCT(YoloV5Param, yoloV5);
        CPL_PARAM_STRUCT(IimParam, iim);
        CPL_PARAM_VECTOR(YoloParam, yolo);
        CPL_PARAM_STRUCT(ScrfdParam, scrfd);
        CPL_PARAM_STRUCT(ScrfdV2Param, scrfdV2);
        CPL_PARAM_STRUCT(RtdetrV2Param, rtdetrV2);
        CPL_PARAM_STRUCT(AlphaParam, alpha);
        CPL_PARAM_STRUCT(RegionParam, region);
        CPL_PARAM_STRUCT(NanodetParam, nanodet);
        CPL_PARAM_STRUCT(YoloV11Param, yoloV11);
        CPL_PARAM_STRUCT(SsdParam, ssd);
    };

    class RegionDetection
    {
        Shape _shape;
        Strings _names;
        bool _enable;

        AnchorDecoder _anchor;
        UltrafaceDecoder _ultraface;
        YoloV5Decoder _yoloV5;
        YoloV7Decoder _yoloV7;
        YoloV8Decoder _yoloV8;
        IimDecoder _iim;
        RtdetrDecoder _rtdetr;
        DetOutDecoder _detOut;
        YoloDecoder _yolo;
        ScrfdDecoder _scrfd;
        ScrfdV2Decoder _scrfdV2;
        RtdetrV2Decoder _rtdetrV2;
        AlphaDecoder _alpha;
        RegionDecoder _region;
        NanodetDecoder _nanodet;
        YoloV11Decoder _yoloV11;
        SsdDecoder _ssd;

    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        RegionDetection()
            : _enable(false)
        {
        }

        bool Init(const Net & net, const DetectionParam& param)
        {
            Strings names;
            for(size_t i = 0; i < net.Dst().size(); ++i)
                names.push_back(net.Dst()[i]->Name());
            return Init(net.NchwShape(), names, param);
        }

        bool Init(const Shape & shape, const Strings & names, const DetectionParam& param)
        {
            if (shape.size() != 4)
                return false;
            _shape = shape;
            _names = names;
            const String& decoder = param.decoder();
            if (decoder == "epsilon")
                _enable = _anchor.Init(_shape[3], _shape[2], param.epsilon());
            else if (decoder == "retina")
                _enable = _anchor.Init(_shape[3], _shape[2], param.retina());
            else if (decoder == "ultraface")
                _enable = _ultraface.Init(param.ultraface());
            else if (decoder == "yoloV5")
                _enable = _yoloV5.Init(_shape[3], _shape[2], param.yoloV5());
            else if (decoder == "yoloV7")
                _enable = _yoloV7.Init(_shape[3], _shape[2]);
            else if (decoder == "yoloV8")
                _enable = _yoloV8.Init(_shape[3], _shape[2]);
            else if (decoder == "iim")
                _enable = _iim.Init(_shape[3], _shape[2], param.iim());
            else if (decoder == "rtdetr")
                _enable = _rtdetr.Init();
            else if (decoder == "detOut")
                _enable = _detOut.Init();
            else if (decoder == "yolo")
                _enable = _yolo.Init(_shape[3], _shape[2], param.yolo());
            else if (decoder == "scrfd")
                _enable = _scrfd.Init(_shape[3], _shape[2], param.scrfd());
            else if (decoder == "scrfdV2")
                _enable = _scrfdV2.Init(_shape[3], _shape[2], param.scrfdV2());
            else if (decoder == "rtdetrV2")
                _enable = _rtdetrV2.Init(_shape[3], _shape[2], param.rtdetrV2());
            else if (decoder == "alpha")
                _enable = _alpha.Init(_shape[3], _shape[2], param.alpha());
            else if (decoder == "region")
                _enable = _region.Init(param.region());
            else if (decoder == "nanodet")
                _enable = _nanodet.Init(_shape[3], _shape[2], param.nanodet());
            else if (decoder == "yoloV11")
                _enable = _yoloV11.Init(_shape[3], _shape[2], param.yoloV11());
            else if (decoder == "ssd")
                _enable = _ssd.Init(param.ssd());
            return _enable;
        }

        bool Enable() const
        {
            return _enable;
        }

        Regions GetRegions(const Net& net, size_t imgW, size_t imgH, float threshold, float overlap, size_t thread = 0) const
        {
            if (_anchor.Enable())
                return _anchor.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_ultraface.Enable())
                return _ultraface.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_iim.Enable())
                return _iim.GetRegions(net, imgW, imgH, thread)[0];
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_detOut.Enable())
                return _detOut.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_yolo.Enable())
                return _yolo.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_scrfd.Enable())
                return _scrfd.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_scrfdV2.Enable())
                return _scrfdV2.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_rtdetrV2.Enable())
                return _rtdetrV2.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_alpha.Enable())
                return _alpha.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_region.Enable())
                return _region.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_nanodet.Enable())
                return _nanodet.GetRegions(net, imgW, imgH, threshold, overlap, Index(), thread)[0];
            else if (_yoloV11.Enable())
                return _yoloV11.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else if (_ssd.Enable())
                return _ssd.GetRegions(net, imgW, imgH, threshold, overlap, thread)[0];
            else
                return net.GetRegions(imgW, imgH, threshold, overlap, thread);
        }

        Regions GetRegions(const Tensors & dst, size_t imgW, size_t imgH, float threshold, float overlap) const
        {
            if (_anchor.Enable())
                return _anchor.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_ultraface.Enable())
                return _ultraface.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_iim.Enable())
                return _iim.GetRegions(dst, imgW, imgH)[0];
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_detOut.Enable())
                return _detOut.GetRegions(dst[0], imgW, imgH, threshold, overlap)[0];
            else if (_yolo.Enable())
                return _yolo.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_scrfd.Enable())
                return _scrfd.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_scrfdV2.Enable())
                return _scrfdV2.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_rtdetrV2.Enable())
                return _rtdetrV2.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_alpha.Enable())
                return _alpha.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_region.Enable())
                return _region.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_nanodet.Enable())
                return _nanodet.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_yoloV11.Enable())
                return _yoloV11.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else if (_ssd.Enable())
                return _ssd.GetRegions(dst, imgW, imgH, threshold, overlap)[0];
            else
                return Regions();
        }
    };
}

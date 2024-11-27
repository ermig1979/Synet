/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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

#include "TestParams.h"

#include "Synet/Network.h"

namespace Test
{
    class RegionDecoder
    {
        Shape _shape;
        Strings _names;
        bool _enable;

        Synet::AnchorDecoder _anchor;
        Synet::UltrafaceDecoder _ultraface;
        Synet::YoloV5Decoder _yoloV5;
        Synet::YoloV7Decoder _yoloV7;
        Synet::YoloV8Decoder _yoloV8;
        Synet::IimDecoder _iim;
        Synet::RtdetrDecoder _rtdetr;
        Synet::DetOutDecoder _detOut;
        Synet::YoloDecoder _yolo;
        Synet::ScrfdDecoder _scrfd;
        Synet::RtdetrV2Decoder _rtdetrV2;
        Synet::AlphaDecoder _alpha;
        Synet::RegionDecoder _region;

    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Tensor<float> Tensor;
        typedef std::vector<Tensor> Tensors;
        typedef Synet::Network Net;

        RegionDecoder()
            : _enable(false)
        {
        }

        bool Init(const Net & net, const TestParam& param)
        {
            Strings names;
            for(size_t i = 0; i < net.Dst().size(); ++i)
                names.push_back(net.Dst()[i]->Name());
            return Init(net.NchwShape(), names, param);
        }

        bool Init(const Shape & shape, const Strings & names, const TestParam& param)
        {
            if (shape.size() != 4)
                return false;
            _shape = shape;
            _names = names;
            const String& decoder = param.detection().decoder();
            if (decoder == "epsilon")
                _enable = _anchor.Init(_shape[3], _shape[2], param.detection().epsilon());
            else if (decoder == "retina")
                _enable = _anchor.Init(_shape[3], _shape[2], param.detection().retina());
            else if (decoder == "ultraface")
                _enable = _ultraface.Init(param.detection().ultraface());
            else if (decoder == "yoloV5")
                _enable = _yoloV5.Init(_shape[3], _shape[2], param.detection().yoloV5());
            else if (decoder == "yoloV7")
                _enable = _yoloV7.Init(_shape[3], _shape[2]);
            else if (decoder == "yoloV8")
                _enable = _yoloV8.Init(_shape[3], _shape[2]);
            else if (decoder == "iim")
                _enable = _iim.Init(_shape[3], _shape[2], param.detection().iim());
            else if (decoder == "rtdetr")
                _enable = _rtdetr.Init();
            else if (decoder == "detOut")
                _enable = _detOut.Init();
            else if (decoder == "yolo")
                _enable = _yolo.Init(_shape[3], _shape[2], param.detection().yolo());
            else if (decoder == "scrfd")
                _enable = _scrfd.Init(_shape[3], _shape[2], param.detection().scrfd());
            else if (decoder == "rtdetrV2")
                _enable = _rtdetrV2.Init(_shape[3], _shape[2], param.detection().rtdetrV2());
            else if (decoder == "alpha")
                _enable = _alpha.Init(_shape[3], _shape[2], param.detection().alpha());
            else if (decoder == "region")
                _enable = _region.Init(param.detection().region());
            return _enable;
        }

        bool Enable() const
        {
            return _enable;
        }

        Regions GetRegions(const Net& net, const Size& size, float threshold, float overlap) const
        {
            if (_anchor.Enable())
                return _anchor.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_ultraface.Enable())
                return _ultraface.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_iim.Enable())
                return _iim.GetRegions(net, size.x, size.y)[0];
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_detOut.Enable())
                return _detOut.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_yolo.Enable())
                return _yolo.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_scrfd.Enable())
                return _scrfd.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_rtdetrV2.Enable())
                return _rtdetrV2.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_alpha.Enable())
                return _alpha.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else if (_region.Enable())
                return _region.GetRegions(net, size.x, size.y, threshold, overlap)[0];
            else
                return net.GetRegions(size.x, size.y, threshold, overlap);
        }

        Regions GetRegions(const Tensors & dst, const Size& size, float threshold, float overlap) const
        {
            if (_anchor.Enable())
                return _anchor.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_ultraface.Enable())
                return _ultraface.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV5.Enable())
                return _yoloV5.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV7.Enable())
                return _yoloV7.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_yoloV8.Enable())
                return _yoloV8.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_iim.Enable())
                return _iim.GetRegions(dst, size.x, size.y)[0];
            else if (_rtdetr.Enable())
                return _rtdetr.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_detOut.Enable())
                return _detOut.GetRegions(dst[0], size.x, size.y, threshold, overlap)[0];
            else if (_yolo.Enable())
                return _yolo.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_scrfd.Enable())
                return _scrfd.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_rtdetrV2.Enable())
                return _rtdetrV2.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_alpha.Enable())
                return _alpha.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else if (_region.Enable())
                return _region.GetRegions(dst, size.x, size.y, threshold, overlap)[0];
            else
                return Regions();
        }
    };
}


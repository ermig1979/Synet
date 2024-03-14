/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2022 Yermalayeu Ihar.
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
        Strings _names;
        size_t _srcW, _srcH;
        bool _enable;

        Synet::AnchorDecoder _anchor;
        Synet::UltrafaceDecoder _ultraface;
        Synet::YoloV5Decoder _yoloV5;
        Synet::YoloV7Decoder _yoloV7;
        Synet::YoloV8Decoder _yoloV8;
        Synet::IimDecoder _iim;
        Synet::RtdetrDecoder _rtdetr;
        Synet::DetOutDecoder _detOut;

    public:
        typedef Synet::Region<float> Region;
        typedef std::vector<Region> Regions;
        typedef Synet::Network Net;

        RegionDecoder()
            : _enable(false)
            , _srcW(0)
            , _srcH(0)
        {
        }

        bool Init(const Net & net, const TestParam& param)
        {
            const Shape& shape = net.NchwShape();
            Strings names;
            for(size_t i = 0; i < net.Dst().size(); ++i)
                names.push_back(net.Dst()[i]->Name());
            return Init(shape[3], shape[2], names, param);
        }

        bool Init(size_t srcW, size_t srcH, const Strings & names, const TestParam& param)
        {
            _srcW = srcW;
            _srcH = srcH;
            _names = names;
            const String& decoder = param.detection().decoder();
            if (decoder == "epsilon")
                _enable = _anchor.Init(_srcW, _srcH, param.detection().epsilon());
            else if (decoder == "retina")
                _enable = _anchor.Init(_srcW, _srcH, param.detection().retina());
            else if (decoder == "ultraface")
                _enable = _ultraface.Init(param.detection().ultraface());
            else if (decoder == "yoloV5")
                _enable = _yoloV5.Init(param.detection().yoloV5());
            else if (decoder == "yoloV7")
                _enable = _yoloV7.Init();
            else if (decoder == "yoloV8")
                _enable = _yoloV8.Init();
            else if (decoder == "iim")
                _enable = _iim.Init(param.detection().iim());
            else if (decoder == "rtdetr")
                _enable = _rtdetr.Init();
            else if (decoder == "detOut")
                _enable = _detOut.Init();
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
            else
                return net.GetRegions(size.x, size.y, threshold, overlap);
        }
    };
}


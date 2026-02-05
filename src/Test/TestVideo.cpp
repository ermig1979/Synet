/*
* Tests for Synet Framework (http://github.com/ermig1979/Synet).
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

#ifdef SYNET_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include "opencv2/core/utils/logger.hpp"
#define SIMD_OPENCV_ENABLE
#endif

#include "Simd/SimdFrame.hpp"

#include "Cpl/Args.h"
#include "Cpl/Log.h"

#include "Synet/Network.h"

#include "TestCommon.h"

namespace Test
{
    struct Options : public Cpl::ArgsParser
    {
        String source;
        String output;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
        {
            source = GetArg2("-s", "-source", "", true);
            output = GetArg2("-o", "-output", "");
        }
    };

    //-------------------------------------------------------------------------------------------------

    struct Video
    {
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef Simd::View<Simd::Allocator> Image;

        struct Filter
        {
            virtual ~Filter() = default;

            virtual bool Process(const Frame& input, Frame& output) = 0;
        };

#ifdef SYNET_OPENCV_ENABLE
    protected:
        const String SYNET_DEBUG_WINDOW_NAME = "Synet::Output";

        bool _window;
        cv::VideoCapture _capture;
        cv::VideoWriter _writer;
        Filter* _filter;

        Frame Convert(const cv::Mat& frame)
        {
            if (frame.channels() == 3)
                return Frame(Image(frame), false, _capture.get(cv::CAP_PROP_POS_MSEC) * 0.001);
            return Frame();
        }
    public:

        Video(bool window = true)
            : _window(window)
            , _filter(NULL)
        {
        }

        virtual ~Video()
        {
            if (_window)
                cv::destroyWindow(SYNET_DEBUG_WINDOW_NAME.c_str());
        }

        bool SetSource(const String& source)
        {
            if (source == "0")
                _capture.open(0);
            else
                _capture.open(source);
            return _capture.isOpened();
        }

        bool SetOutput(const String& output)
        {
            if (!_capture.isOpened())
                return false;
            _writer.open(output, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), _capture.get(cv::CAP_PROP_FPS),
                cv::Size((int)_capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)_capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
            return _writer.isOpened();
        }

        bool SetFilter(Filter* filter)
        {
            _filter = filter;
            return true;
        }

        bool Start()
        {
            if (!_capture.isOpened())
                return false;
            while (1)
            {
                cv::Mat frame;
                if (!_capture.read(frame))
                    break;
                if (_filter)
                    _filter->Process(Convert(frame), Convert(frame).Ref());
                if (_writer.isOpened())
                    _writer.write(frame);
                if (_window)
                    cv::imshow(SYNET_DEBUG_WINDOW_NAME, frame);
                if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
                    break;
            }
            return true;
        }
#else
    public:
        Video(bool window = true)
        {
        }

        virtual ~Video()
        {
        }

        bool SetSource(const String& source)
        {
            SYNET_ERROR("OpenCV is not enabled!");
        }

        bool SetOutput(const String& output)
        {
            SYNET_ERROR("OpenCV is not enabled!");
        }

        bool SetFilter(Filter* filter)
        {
            SYNET_ERROR("OpenCV is not enabled!");
        }

        bool Start()
        {
            SYNET_ERROR("OpenCV is not enabled!");
        }
#endif
    };
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    return 0;
}



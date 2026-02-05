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

    typedef Simd::View<Simd::Allocator> Image;

    struct Frame
    {
        cv::Mat original;
        double videoTime;
        double startTime;

        Frame()
            : videoTime(0)
            , startTime(0)
        {
        }
    };

    typedef std::shared_ptr<Frame> FramePtr;
    typedef std::vector<FramePtr> FramePtrs;
    typedef std::list<FramePtr> FramePtrList;

    //-------------------------------------------------------------------------------------------------

    struct FrameQueue
    {
        FrameQueue()
        {
        }

        void Push(FramePtr frame)
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _frames.push_back(frame);
        }

        FramePtr Pop()
        {
            std::lock_guard<std::mutex> lock(_mutex);
            FramePtr frame;
            if (_frames.size())
            {
                frame = _frames.front();
                _frames.pop_front();
            }
            return frame;
        }

        size_t Size() const
        {
            std::lock_guard<std::mutex> lock(_mutex);
            return _frames.size();
        }

        private:
            FramePtrList _frames;
            mutable std::mutex _mutex;
    };

    //-------------------------------------------------------------------------------------------------

    struct Filter
    {
        virtual ~Filter() = default;

        virtual bool Process(FramePtr frame) = 0;
    };

    //-------------------------------------------------------------------------------------------------

    struct VideoManager
    {
    protected:
        const String SYNET_DEBUG_WINDOW_NAME = "Synet::Output";

        bool _window, _started, _finished;
        cv::VideoCapture _capture;
        cv::VideoWriter _writer;
        Filter* _filter;
        FrameQueue _input;
        Simd::Font _font;

    public:

        VideoManager(bool window = true)
            : _window(window)
            , _started(false)
            , _finished(false)
            , _filter(NULL)
        {
            _font.Resize(50);
        }

        virtual ~VideoManager()
        {
            if (_window && _started)
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
            _started = true;

            std::thread readThread = std::thread(ReadThread, this);

            std::thread writeThread = std::thread(WriteThread, this);

            if (readThread.joinable())
                readThread.join();

            if (writeThread.joinable())
                writeThread.join();
            //while (1)
            //{
            //    {
            //        FramePtr frame = std::make_shared<Frame>();
            //        frame->startTime = Cpl::Time();
            //        if (!_capture.read(frame->original))
            //            break;
            //        frame->videoTime = _capture.get(cv::CAP_PROP_POS_MSEC) * 0.001;
            //        _input.Push(frame);
            //    }

            //    //if (_filter)
            //    //    _filter->Process(frame);

            //    {
            //        FramePtr frame = _input.Pop();
            //        if (_writer.isOpened())
            //            _writer.write(frame->original);
            //        if (_window)
            //            cv::imshow(SYNET_DEBUG_WINDOW_NAME, frame->original);
            //    }

            //    if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
            //        break;
            //}
            return true;
        }

        void ReadFrames()
        {
            double startTime = Cpl::Time();
            while (!_finished)
            {
                FramePtr frame = std::make_shared<Frame>();
                frame->startTime = Cpl::Time();
                if (!_capture.read(frame->original))
                    break;
                frame->videoTime = _capture.get(cv::CAP_PROP_POS_MSEC) * 0.001;
                _input.Push(frame);

                double delay = (frame->videoTime - (frame->startTime - startTime)) * 0.5;
                if (delay > 0)
                    Sleep(int(delay * 1000));

                std::cout << Cpl::ToStr(frame->videoTime, 3) << " sec. \r" << std::flush;

                //if (_filter)
                //    _filter->Process(frame);

                if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
                    _finished = true;
            }
            _finished = true;
        }

        static void ReadThread(VideoManager * videoManager)
        {
            videoManager->ReadFrames();
        }

        void WriteFrames()
        {
            while (!(_finished && _input.Size() == 0))
            {
                FramePtr frame = _input.Pop();
                if (frame)
                {
                    double currentTime = Cpl::Time();
                    double delay = currentTime - frame->startTime;
                    Image output = frame->original;
                    Simd::Pixel::Bgr24 red(0, 0, 255), green(0, 255, 0);
                    std::stringstream ss;
                    ss << " Delay " << Cpl::ToStr(delay, 3) << " ; i-queue: " << _input.Size();
                    _font.Draw(output, ss.str(), Image::BottomLeft, delay < 1.0 ? green : red);


                    if (_writer.isOpened())
                        _writer.write(frame->original);


                    if (_window)
                        cv::imshow(SYNET_DEBUG_WINDOW_NAME, frame->original);
                }
            }
        }

        static void WriteThread(VideoManager* videoManager)
        {
            videoManager->WriteFrames();
        }

        inline void Sleep(unsigned int miliseconds)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
        }
    };
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);

    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

#ifdef __linux__
    Test::VideoManager video(false);
#else
    Test::VideoManager video(true);
#endif

    if (options.source.length() == 0)
        SYNET_ERROR("Video source is undefined (-s parameter)!");
    if (!video.SetSource(options.source))
        SYNET_ERROR("Can't open source video file '" << options.source << "'!");
    if (options.output.length() != 0 && !video.SetOutput(options.output))
        SYNET_ERROR("Can't open output video file '" << options.output << "'!");

    video.Start();

    return 0;
}

#else

#include "Cpl/Log.h"

int main(int argc, char* argv[])
{
    Cpl::Log::Global().AddStdWriter(Cpl::Log::Info);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    CPL_LOG_SS(Error, "OpenCV is not enable!"); 

    return 1;
}

#endif



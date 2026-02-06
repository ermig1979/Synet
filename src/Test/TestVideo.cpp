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
#include "Synet/Decoders/YoloV11.h"

#include "TestCommon.h"
#include "TestParams.h"

namespace Test
{
    typedef Synet::Region<float> Region;
    typedef std::vector<Region> Regions;

    //-------------------------------------------------------------------------------------------------

    struct Options : public Cpl::ArgsParser
    {
        String source;
        String output;
        String model;
        String weight;
        String param;
        double realtime;
        double endTime;

        Options(int argc, char* argv[])
            : ArgsParser(argc, argv, true)
        {
            source = GetArg("-s", "");
            output = GetArg("-o", "");
            model = GetArg("-m", "");
            weight = GetArg("-w", "");
            param = GetArg("-p", "");
            realtime = Cpl::ToVal<double>(GetArg("-rt", "1.0"));
            endTime = Cpl::ToVal<double>(GetArg("-et", "0.0"));
        }
    };

    //-------------------------------------------------------------------------------------------------

    typedef Simd::View<Simd::Allocator> Image;

    struct Frame
    {
        cv::Mat original;
        double videoTime;
        double startTime;

        Regions regions;

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

    struct Analyser
    {
    private:
        const Options& _options;
        Synet::Network _detector;
        Test::TestParamHolder _param;
    public:
        Analyser(const Options& options)
            : _options(options)
        {
        }

        bool Init()
        {
            if (!_detector.Load(_options.model, _options.weight))
                SYNET_ERROR("Can't load model " << _options.model << " with weight " << _options.weight << " !");
            if (!_param.Load(_options.param))
                SYNET_ERROR("Can't load param " << _options.param << " !");
            return true;
        }

        virtual ~Analyser()
        {
        };

        virtual bool Process(FramePtr frame)
        {
            Image input = Image(frame->original);
            Image resized(_detector.NchwShape()[3], _detector.NchwShape()[2], input.format);
            Simd::Resize(input, resized, SimdResizeMethodArea);
            _detector.SetInput(resized, _param().lower(), _param().upper());
            _detector.Forward();
            Synet::YoloV11Decoder decoder;
            decoder.Init(_detector.NchwShape()[3], _detector.NchwShape()[2], _param().detection().yoloV11());
            frame->regions = decoder.GetRegions(_detector, input.width, input.height, _param().detection().confidence(), _param().detection().overlap())[0];
            return true;
        }

    private:

    };

    //-------------------------------------------------------------------------------------------------

    struct VideoManager
    {
    protected:

        Options _options;
        bool _readFinished, _analyseFinished;
        cv::VideoCapture _capture;
        cv::VideoWriter _writer;
        Analyser* _analyser;
        FrameQueue _input, _output;
        Simd::Font _font;

    public:

        VideoManager(const Options& options)
            : _options(options)
            , _readFinished(false)
            , _analyseFinished(false)
            , _analyser(NULL)
        {
            _font.Resize(20);
        }

        virtual ~VideoManager()
        {
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

        bool SetAnalyser(Analyser* analyser)
        {
            _analyser = analyser;
            return true;
        }

        bool Start()
        {
            if (!_capture.isOpened())
                return false;

            std::thread readThread = std::thread(ReadThread, this);

            std::thread analyseThread = std::thread(AnalyseThread, this);

            std::thread writeThread = std::thread(WriteThread, this);

            if (readThread.joinable())
                readThread.join();

            if (analyseThread.joinable())
                analyseThread.join();

            if (writeThread.joinable())
                writeThread.join();

            return true;
        }

    protected:

        void ReadFrames()
        {
            double startTime = Cpl::Time();
            while (1)
            {
                FramePtr frame = std::make_shared<Frame>();
                {
                    CPL_PERF_BEG("read");
                    frame->startTime = Cpl::Time();
                    if (!_capture.read(frame->original))
                        break;
                    frame->videoTime = _capture.get(cv::CAP_PROP_POS_MSEC) * 0.001;
                    _input.Push(frame);
                }

                double delay = (frame->videoTime*_options.realtime - (frame->startTime - startTime)) * 0.5;
                if (delay > 0)
                    Sleep(int(delay * 1000));

                std::cout << "Process: " << Cpl::ToStr(frame->videoTime, 3) << " sec. \r" << std::flush;

                if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
                    break;

                if (_options.endTime > 0 && frame->videoTime > _options.endTime)
                    break;
            }
            _readFinished = true;
        }

        static void ReadThread(VideoManager * videoManager)
        {
            videoManager->ReadFrames();
        }

        void AnalyseFrames()
        {
            while (!(_readFinished && _input.Size() == 0))
            {
                FramePtr frame = _input.Pop();
                if (frame)
                {
                    CPL_PERF_BEG("analyse");
                    if (_analyser)
                        _analyser->Process(frame);
                    _output.Push(frame);
                }
                else
                    Sleep(1);
            }
            _analyseFinished = true;
        }

        static void AnalyseThread(VideoManager* videoManager)
        {
            videoManager->AnalyseFrames();
        }

        void WriteFrames()
        {
            while (!(_analyseFinished && _output.Size() == 0))
            {
                FramePtr frame = _output.Pop();
                if (frame)
                {
                    CPL_PERF_BEG("write");
                    double currentTime = Cpl::Time();
                    double delay = currentTime - frame->startTime;
                    Image output = frame->original;
                    Simd::Pixel::Bgr24 red(0, 0, 255), green(0, 255, 0);
                    std::stringstream ss;
                    ss << "Delay " << Cpl::ToStr(delay, 3) << " ; i-queue: " << _input.Size() << " ; o-queue: " << _output.Size();

                    _font.Draw(output, ss.str(), Image::BottomLeft, delay < 1.0 ? green : red);

                    for (size_t i = 0; i < frame->regions.size(); ++i)
                    {
                        const Region& region = frame->regions[i];
                        ptrdiff_t l = ptrdiff_t(region.x - region.w / 2);
                        ptrdiff_t t = ptrdiff_t(region.y - region.h / 2);
                        ptrdiff_t r = ptrdiff_t(region.x + region.w / 2);
                        ptrdiff_t b = ptrdiff_t(region.y + region.h / 2);
                        Simd::DrawRectangle(output, l, t, r, b, red, 1);
                    }

                    if (_writer.isOpened())
                        _writer.write(frame->original);
                }
                else
                    Sleep(1);
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

    Test::VideoManager video(options);

    if (options.source.length() == 0)
        SYNET_ERROR("Video source is undefined (-s parameter)!");
    if (!video.SetSource(options.source))
        SYNET_ERROR("Can't open source video file '" << options.source << "'!");
    if (options.output.length() != 0 && !video.SetOutput(options.output))
        SYNET_ERROR("Can't open output video file '" << options.output << "'!");

    Test::Analyser analyser(options);
    if(!analyser.Init())
        SYNET_ERROR("Can't init video analyser!");

    video.SetAnalyser(&analyser);

    video.Start();

    std::cout << std::endl << Cpl::PerformanceStorage::Global().Report() << std::endl;

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



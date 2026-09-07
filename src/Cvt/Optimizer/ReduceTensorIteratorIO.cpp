/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Cvt/Optimizer/Common.h"
#include "Cvt/Optimizer/Optimizer.h"

namespace Synet
{
    bool ReduceTensorIteratorIO(const LayerParams& src, size_t& index, const Bytes& bin, Bytes& buf, LayerParams& dst)
    {
        const LayerParam & stt = src[index];
        if (stt.type() != LayerTypeTensorIterator || stt.src().size() < 3 || stt.tensorIterator().back().size() < 1)
            return false;
        size_t srcDupls = 0;
        for (size_t i = 2; i < stt.src().size(); ++i)
        {
            if (stt.src()[1] == stt.src()[i])
                srcDupls++;
        }
        size_t backDupls = 0;
        for (size_t i = 1; i < stt.tensorIterator().back().size(); ++i)
        {
            if (stt.tensorIterator().back()[0].src() == stt.tensorIterator().back()[i].src())
                backDupls++;
        }
        if (srcDupls == 0 || srcDupls != backDupls || srcDupls < stt.src().size() - 2)
            return false;
        dst.push_back(stt);
        LayerParam& dtt = dst.back();
        dtt.src().resize(2);
        String rem, iter;
        for (size_t i = 0; i < dtt.tensorIterator().input().size() && iter.empty(); ++i)
            if (dtt.tensorIterator().input()[i].axis() != -1)
                iter = dtt.tensorIterator().input()[i].dst();
        for (size_t i = index + 1; i < src.size() && rem.empty(); ++i)
        {
            if (src[i].parent() != stt.name())
                break;
            if (src[i].type() == LayerTypeInput && src[i].name() != iter)
                rem = src[i].name();
        }
        StringSet del;
        std::vector<ConnectionParam> back, input;
        for (size_t i = 0; i < dtt.tensorIterator().input().size(); ++i)
        {
            ConnectionParam & p = dtt.tensorIterator().input()[i];
            if (p.dst() == rem || p.dst() == iter)
            {
                p.port() = Synet::Min<int>(1, p.port());
                input.push_back(p);
            }
            else
                del.insert(p.dst());
        }
        dtt.tensorIterator().input().swap(input);
        for (size_t i = 0; i < stt.tensorIterator().back().size(); ++i)
        {
            if (del.find(dtt.tensorIterator().back()[i].dst()) == del.end())
                back.push_back(dtt.tensorIterator().back()[i]);
        }
        dtt.tensorIterator().back().swap(back);
        for (size_t i = index + 1; i < src.size(); ++i)
        {
            if (src[i].parent() != stt.name())
                break;
            if (src[i].type() != LayerTypeInput || del.find(src[i].name()) == del.end())
                dst.push_back(src[i]);
            for (size_t j = 0; j < dst.back().src().size(); ++j)
            {
                if (del.find(dst.back().src()[j]) != del.end())
                    dst.back().src()[j] = rem;
            }
            index++;
        }
        return true;
    }
}

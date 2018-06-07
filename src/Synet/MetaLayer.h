/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2018 Yermalayeu Ihar.
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
#include "Synet/Layer.h"

namespace Synet
{
    template <class T> class MetaLayer : public Synet::Layer<T>
    {
    public:
        typedef T Type;
        typedef Layer<T> Base;
        typedef typename Base::TensorPtrs TensorPtrs;

        MetaLayer(const LayerParam & param)
            : Base(param)
        {
        }

        virtual void Setup(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            const MetaParam & param = this->Param().meta();
            _type = param.type();
            _alpha = param.alpha();
        }

        virtual void Reshape(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
            switch (_type)
            {
            case MetaTypeAdd:
            {
                assert(src.size() == 2 && src[0]->Count() == src[1]->Count());
                Shape dstShape = src[0]->Shape();
                for (size_t i = 0; i < dstShape.size(); ++i)
                    dstShape[i] += src[1]->Shape()[i];
                dst[0]->SetShape(dstShape);
                break;
            }
            case MetaTypeConst:
            {
                dst[0]->SetShape(_alpha);
                break;
            }            
            case MetaTypeInput:
            {
                dst[0]->SetShape(src[0]->Shape());
                break;
            }
            case MetaTypePack:
            {
                Shape dstShape;
                for (size_t i = 0; i < src.size(); ++i)
                    for (size_t j = 0; j < src[i]->Count(); ++j)
                        dstShape.push_back(src[i]->Axis(j));
                dst[0]->SetShape(dstShape);
                break;
            }
            case MetaTypeRange:
            {
                assert(src.size() == 3);
                Shape dstShape;
                if(src[1]->Axis(2) > 0)
                    for (size_t i = src[0]->Axis(0); i < src[1]->Axis(0); i += src[2]->Axis(0))
                        dstShape.push_back(i);
                else
                    for (size_t i = src[0]->Axis(0); i > src[1]->Axis(0); i += src[2]->Axis(0))
                        dstShape.push_back(i);
                dst[0]->SetShape(dstShape);
                break;
            }
            case MetaTypeShape:
            {
                Shape shape = src[0]->Shape();
                if (shape.size() == 4)
                    shape = Shape({ shape[0], shape[2], shape[3], shape[1] });
                if (shape.size() == 2)
                    shape = Shape({ shape[1], shape[0] });
                dst[0]->SetShape(shape);
                break;
            }
            case MetaTypeSlice:
            {
                assert(src.size() == 3);
                Shape dstShape(src[2]->Axis(0));
                for (size_t i = 0; i < src[2]->Axis(0); ++i)
                    dstShape[i] = src[0]->Shape()[i + src[1]->Axis(0)];
                dst[0]->SetShape(dstShape);
                break;
            }
            case MetaTypeStridedSlice:
            {
                assert(src.size() == 4);
                Shape dstShape;
                for (size_t i = src[1]->Axis(0); i < src[2]->Axis(0); i += src[3]->Axis(0))
                    dstShape.push_back(src[0]->Shape()[i]);
                dst[0]->SetShape(dstShape);
                break;
            }
            case MetaTypeStub:
            {
                dst[0]->SetShape({});
                break;
            }            
            case MetaTypeSub:
            {
                assert(src.size() == 2 && src[0]->Count() == src[1]->Count());
                Shape dstShape = src[0]->Shape();
                for (size_t i = 0; i < dstShape.size(); ++i)
                    dstShape[i] -= src[1]->Shape()[i];
                dst[0]->SetShape(dstShape);
                break;
            }
            default:
                assert(0);
            }
        }

    protected:
        virtual void ForwardCpu(const TensorPtrs & src, const TensorPtrs & buf, const TensorPtrs & dst)
        {
        }

    private:
        MetaType _type;
        Shape _alpha;
    };
}
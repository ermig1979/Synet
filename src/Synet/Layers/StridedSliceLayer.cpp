/*
* Synet Framework (http://github.com/ermig1979/Synet).
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

#include "Synet/Layers/StridedSliceLayer.h"

namespace Synet
{
	template<class T> void StridedSlice1(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
			pDst0[d0] = pSrc0[s0];
	}

	template<class T> void StridedSlice2(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
		{
			const T* pSrc1 = pSrc0 + s0 * srcStrides[1];
			T* pDst1 = pDst0 + d0 * dstStrides[1];
			for (int64_t s1 = beginDims[1], d1 = 0; d1 < dstDims[1]; d1 += 1, s1 += strideDims[1])
				pDst1[d1] = pSrc1[s1];
		}
	}

	template<class T> void StridedSlice3(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
		{
			const T* pSrc1 = pSrc0 + s0 * srcStrides[1];
			T* pDst1 = pDst0 + d0 * dstStrides[1];
			for (int64_t s1 = beginDims[1], d1 = 0; d1 < dstDims[1]; d1 += 1, s1 += strideDims[1])
			{
				const T* pSrc2 = pSrc1 + s1 * srcStrides[2];
				T* pDst2 = pDst1 + d1 * dstStrides[2];
				for (int64_t s2 = beginDims[2], d2 = 0; d2 < dstDims[2]; d2 += 1, s2 += strideDims[2])
					pDst2[d2] = pSrc2[s2];
			}
		}
	}

	template<class T> void StridedSlice4(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
		{
			const T* pSrc1 = pSrc0 + s0 * srcStrides[1];
			T* pDst1 = pDst0 + d0 * dstStrides[1];
			for (int64_t s1 = beginDims[1], d1 = 0; d1 < dstDims[1]; d1 += 1, s1 += strideDims[1])
			{
				const T* pSrc2 = pSrc1 + s1 * srcStrides[2];
				T* pDst2 = pDst1 + d1 * dstStrides[2];
				for (int64_t s2 = beginDims[2], d2 = 0; d2 < dstDims[2]; d2 += 1, s2 += strideDims[2])
				{
					const T* pSrc3 = pSrc2 + s2 * srcStrides[3];
					T* pDst3 = pDst2 + d2 * dstStrides[3];
					for (int64_t s3 = beginDims[3], d3 = 0; d3 < dstDims[3]; d3 += 1, s3 += strideDims[3])
						pDst3[d3] = pSrc3[s3];
				}
			}
		}
	}

	template<class T> void StridedSlice5(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
		{
			const T* pSrc1 = pSrc0 + s0 * srcStrides[1];
			T* pDst1 = pDst0 + d0 * dstStrides[1];
			for (int64_t s1 = beginDims[1], d1 = 0; d1 < dstDims[1]; d1 += 1, s1 += strideDims[1])
			{
				const T* pSrc2 = pSrc1 + s1 * srcStrides[2];
				T* pDst2 = pDst1 + d1 * dstStrides[2];
				for (int64_t s2 = beginDims[2], d2 = 0; d2 < dstDims[2]; d2 += 1, s2 += strideDims[2])
				{
					const T* pSrc3 = pSrc2 + s2 * srcStrides[3];
					T* pDst3 = pDst2 + d2 * dstStrides[3];
					for (int64_t s3 = beginDims[3], d3 = 0; d3 < dstDims[3]; d3 += 1, s3 += strideDims[3])
					{
						const T* pSrc4 = pSrc3 + s3 * srcStrides[4];
						T* pDst4 = pDst3 + d3 * dstStrides[4];
						for (int64_t s4 = beginDims[4], d4 = 0; d4 < dstDims[4]; d4 += 1, s4 += strideDims[4])
							pDst4[d4] = pSrc4[s4];
					}
				}
			}
		}
	}

	template<class T> void StridedSlice6(const uint8_t* src, const int64_t* srcStrides, const int64_t* beginDims, const int64_t* dstDims, const int64_t* strideDims, uint8_t* dst, const int64_t* dstStrides)
	{
		const T* pSrc0 = (const T*)src;
		T* pDst0 = (T*)dst;
		for (int64_t s0 = beginDims[0], d0 = 0; d0 < dstDims[0]; d0 += 1, s0 += strideDims[0])
		{
			const T* pSrc1 = pSrc0 + s0 * srcStrides[1];
			T* pDst1 = pDst0 + d0 * dstStrides[1];
			for (int64_t s1 = beginDims[1], d1 = 0; d1 < dstDims[1]; d1 += 1, s1 += strideDims[1])
			{
				const T* pSrc2 = pSrc1 + s1 * srcStrides[2];
				T* pDst2 = pDst1 + d1 * dstStrides[2];
				for (int64_t s2 = beginDims[2], d2 = 0; d2 < dstDims[2]; d2 += 1, s2 += strideDims[2])
				{
					const T* pSrc3 = pSrc2 + s2 * srcStrides[3];
					T* pDst3 = pDst2 + d2 * dstStrides[3];
					for (int64_t s3 = beginDims[3], d3 = 0; d3 < dstDims[3]; d3 += 1, s3 += strideDims[3])
					{
						const T* pSrc4 = pSrc3 + s3 * srcStrides[4];
						T* pDst4 = pDst3 + d3 * dstStrides[4];
						for (int64_t s4 = beginDims[4], d4 = 0; d4 < dstDims[4]; d4 += 1, s4 += strideDims[4])
						{
							const T* pSrc5 = pSrc4 + s4 * srcStrides[5];
							T* pDst5 = pDst4 + d4 * dstStrides[5];
							for (int64_t s5 = beginDims[5], d5 = 0; d5 < dstDims[5]; d5 += 1, s5 += strideDims[5])
								pDst5[d5] = pSrc5[s5];
						}
					}
				}
			}
		}
	}

	//----------------------------------------------------------------------------------------------------

	template<class T> StridedSliceLayer::StridedSlicePtr GetStridedSlice(size_t count)
	{
		switch (count)
		{
		case 1: return StridedSlice1<T>;
		case 2: return StridedSlice2<T>;
		case 3: return StridedSlice3<T>;
		case 4: return StridedSlice4<T>;
		case 5: return StridedSlice5<T>;
		case 6: return StridedSlice6<T>;
		default:
			return NULL;
		}
	}

	StridedSliceLayer::StridedSlicePtr GetStridedSlice(TensorType type, size_t count)
	{
		switch (type)
		{
		case TensorType32f: return GetStridedSlice<float>(count);
		default:
			return NULL;
		}
	}

	//----------------------------------------------------------------------------------------------------

    StridedSliceLayer::StridedSliceLayer(const LayerParam & param, Context* context)
        : Layer(param, context)
    {
    }

    bool StridedSliceLayer::Reshape(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
    {
        StridedSliceParam param = this->Param().stridedSlice();
		if ((src.size() != 1 && src.size() != 4 && src.size() != 5) || dst.size() != 1)
			SYNET_ERROR("StridedSliceLayer supports only 1, 4, 5 inputs and 1 output!");
        _srcDims = Lng(src[0]->Shape());
		_type = src[0]->GetType();
		for(size_t i = 1; i < src.size(); ++i)
			if(src[i]->GetType() != TensorType64i)
				SYNET_ERROR("StridedSliceLayer has wrong type of src[" << i << "] type !");
        if (src.size() == 4 && param.beginDims().empty() && param.endDims().empty() && param.axes().empty())
        {
            param.beginDims() = Lng(src[1]->Data<int64_t>(), src[1]->Size());
            param.endDims() = Lng(src[2]->Data<int64_t>(), src[2]->Size());
            param.axes() = Shp(src[3]->Data<int64_t>(), src[3]->Size());
        }
		if (src.size() == 5)
		{
			if (param.beginDims().size() || param.endDims().size() || param.axes().size() || param.strideDims().size())
				SYNET_ERROR("Check StridedSliceLayer parameters!");
			param.beginDims() = Lng(src[1]->Data<int64_t>(), src[1]->Size());
			param.endDims() = Lng(src[2]->Data<int64_t>(), src[2]->Size());
			param.axes() = Shp(src[3]->Data<int64_t>(), src[3]->Size());
			param.strideDims() = Lng(src[4]->Data<int64_t>(), src[4]->Size());
		}
        _axes = param.axes();
        if (param.strideDims().size())
        {
            if (_axes.size())
            {
				if (param.strideDims().size() != _axes.size())
					SYNET_ERROR("Check strideDims and axes StridedSliceLayer parameters!");
                _strideDims.clear();
                for (size_t s = 0, sd = 0; s < _srcDims.size(); ++s)
                {
                    bool found = false;
                    for (size_t a = 0; a < _axes.size(); ++a)
                        if (s == _axes[a])
                            found = true;
                    _strideDims.push_back(found ? param.strideDims()[sd++] : 1);
                }
            }
            else
                _strideDims = param.strideDims();
			if (_strideDims.size() != _srcDims.size())
				SYNET_ERROR("Check strideDims StridedSliceLayer parameters!");
		}
        else
            _strideDims.resize(_srcDims.size(), 1);
		if (param.beginDims().size())
		{
			if (_axes.size())
			{
				if (param.beginDims().size() != _axes.size())
					SYNET_ERROR("Check beginDims and axes StridedSliceLayer parameters!");
				_beginDims.clear();
				for (size_t s = 0, b = 0; s < _srcDims.size(); ++s)
				{
					bool found = false;
					for (size_t a = 0; a < _axes.size(); ++a)
						if (s == _axes[a])
							found = true;
					if (found)
					{
						int64_t beg = param.beginDims()[b++];
						_beginDims.push_back(beg < 0 ? _srcDims[s] + beg : beg);
					}
					else
						_beginDims.push_back(0);
				}
			}
			else
				_beginDims = param.beginDims();
			if (_beginDims.size() != _srcDims.size())
				SYNET_ERROR("Check beginDims StridedSliceLayer parameters!");
		}
		else
			_beginDims.resize(_srcDims.size(), 0);
		if (param.endDims().size())
		{
			if (_axes.size())
			{
				if (param.endDims().size() != _axes.size())
					SYNET_ERROR("Check endDims and axes StridedSliceLayer parameters!");
				_endDims.clear();
				for (size_t s = 0, e = 0; s < _srcDims.size(); ++s)
				{
					bool found = false;
					for (size_t a = 0; a < _axes.size(); ++a)
						if (s == _axes[a])
							found = true;
					if (found)
					{
						int64_t end = param.endDims()[e++];
						if(_strideDims[s] > 0)
							_endDims.push_back(Min(end, _srcDims[s]));
						else
							_endDims.push_back(Max<int64_t>(end, 0));
					}
					else
						_endDims.push_back(_srcDims[s]);
				}
			}
			else
				_endDims = param.endDims();
			if (_endDims.size() != _srcDims.size())
				SYNET_ERROR("Check endDims StridedSliceLayer parameters!");
		}
		else
			_endDims = Lng(_srcDims);

        _dstDims.resize(_srcDims.size());
        for (size_t i = 0; i < _srcDims.size(); ++i)
        {
            size_t count = 0;
            if (_strideDims[i] > 0)
            {
                size_t begin = _beginDims[i];
                size_t end = _endDims[i] == 0 ? _srcDims[i] : Min<size_t>(_endDims[i], _srcDims[i]);
                for (; begin < end; begin += _strideDims[i])
                    count++;
            }
			else
			{
				int64_t begin = _beginDims[i];
				int64_t end = _endDims[i] == _srcDims[i] ? 0 : Max<int64_t>(_endDims[i], 0);
				for (; begin >= end; begin += _strideDims[i])
					count++;
			}
            _dstDims[i] = count;
        }
        dst[0]->Reshape(_type, Shp(_dstDims), src[0]->Format());
        _srcStrides.resize(_srcDims.size(), 1);
        for (size_t i = 0; i < _srcDims.size(); ++i)
            _srcStrides[i] = src[0]->Size(i);
        _dstStrides.resize(_dstDims.size(), 1);
        for (size_t i = 0; i < _dstDims.size(); ++i)
            _dstStrides[i] = dst[0]->Size(i);
		_stridedSlice = GetStridedSlice(_type, _dstDims.size());
		if(_stridedSlice == NULL)
			SYNET_ERROR("StridedSliceLayer: Can't get handler for current src[0]!");
		if (src[0]->Const())
		{
			ForwardCpu(src, buf, dst);
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

	void StridedSliceLayer::ForwardCpu(const TensorPtrs& src, const TensorPtrs& buf, const TensorPtrs& dst)
	{
		_stridedSlice(src[0]->RawData(), _srcStrides.data(), _beginDims.data(), _dstDims.data(), _strideDims.data(), dst[0]->RawData(), _dstStrides.data());
	}
}
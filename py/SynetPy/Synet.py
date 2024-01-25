###################################################################################################
# Synet Framework (http://github.com/ermig1979/Synet).
#
# Copyright (c) 2018-2024 Yermalayeu Ihar.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###################################################################################################

import argparse
import os
import ctypes
import pathlib
import sys
import array
import enum

import Simd

Synet = sys.modules[__name__]

###################################################################################################

## @ingroup python
# A wrapper around %Synet Library API.
class Lib():
	lib : ctypes.CDLL
	
	## Initializes Synet.Lib class (loads %Synet Library binaries).
	# @note This method must be called before any using of this class.
	# @param dir - a directory with %Synet Library binaries (Synet.dll or libSynet.so). By default it is empty string. That means that binaries will be searched in the directory with current Python file.
	def Init(dir = ""):
		if dir == "" :
			dir = pathlib.Path(__file__).parent.resolve()
		if not os.path.isdir(dir):
			raise Exception("Directory '{0}' with binaries is not exist!".format(dir))
		name : str
		if sys.platform == 'win32':
			name = "Synet.dll"
		else :
			name = "libSynet.so"
		path = str(pathlib.Path(dir).absolute() / name)
		if not os.path.isfile(path):
			raise Exception("Binary file '{0}' is not exist!".format(path))

		Lib.lib = ctypes.CDLL(path)
		
		Lib.lib.SynetVersion.argtypes = []
		Lib.lib.SynetVersion.restype = ctypes.c_char_p 
		
		Lib.lib.SynetRelease.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetRelease.restype = None 
	
		Lib.lib.SynetNetworkInit.argtypes = []
		Lib.lib.SynetNetworkInit.restype = ctypes.c_void_p 
		
		Lib.lib.SynetNetworkLoad.argtypes = [ ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p ]
		Lib.lib.SynetNetworkLoad.restype = ctypes.c_bool 
		
		Lib.lib.SynetNetworkEmpty.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetNetworkEmpty.restype = ctypes.c_bool 
		
		Lib.lib.SynetNetworkReshape.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t ]
		Lib.lib.SynetNetworkReshape.restype = ctypes.c_bool 
		
		Lib.lib.SynetNetworkSetBatch.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		Lib.lib.SynetNetworkSetBatch.restype = ctypes.c_bool 
		
		Lib.lib.SynetNetworkSrcSize.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetNetworkSrcSize.restype = ctypes.c_size_t 

		Lib.lib.SynetNetworkSrc.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		Lib.lib.SynetNetworkSrc.restype = ctypes.c_void_p 
		
		Lib.lib.SynetNetworkDstSize.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetNetworkDstSize.restype = ctypes.c_size_t 

		Lib.lib.SynetNetworkDst.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		Lib.lib.SynetNetworkDst.restype = ctypes.c_void_p 
		
		Lib.lib.SynetNetworkDstByName.argtypes = [ ctypes.c_void_p, ctypes.c_char_p ]
		Lib.lib.SynetNetworkDstByName.restype = ctypes.c_void_p 

		Lib.lib.SynetNetworkCompactWeight.argtypes = [ ctypes.c_void_p]
		Lib.lib.SynetNetworkCompactWeight.restype = None 
		
		Lib.lib.SynetNetworkForward.argtypes = [ ctypes.c_void_p]
		Lib.lib.SynetNetworkForward.restype = None 
		
		Lib.lib.SynetTensorCount.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetTensorCount.restype = ctypes.c_size_t 
		
		Lib.lib.SynetTensorAxis.argtypes = [ ctypes.c_void_p, ctypes.c_ssize_t ]
		Lib.lib.SynetTensorAxis.restype = ctypes.c_size_t
		
		Lib.lib.SynetTensorFormatGet.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetTensorFormatGet.restype = ctypes.c_int
		
		Lib.lib.SynetTensorTypeGet.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetTensorTypeGet.restype = ctypes.c_int
		
		Lib.lib.SynetTensorName.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetTensorName.restype = ctypes.c_char_p
		
		Lib.lib.SynetTensorData.argtypes = [ ctypes.c_void_p ]
		Lib.lib.SynetTensorData.restype = ctypes.c_void_p
		
	## Gets version of %Synet Framework.
	# @return A string with version.
	def Version() -> str: 
		ptr = Lib.lib.SynetVersion()
		return str(ptr, encoding='utf-8')
	
###################################################################################################

class Tensor():
	def __init__(self, tensor: ctypes.c_void_p):
		self.__tensor = tensor
	
	def Shape(self):
		shape = []
		for axis in range(Lib.lib.SynetTensorCount(self.__tensor)) :
			shape.append(Lib.lib.SynetTensorAxis(self.__tensor, axis))
		return shape
	
	def Format(self) -> Simd.TensorFormat:
		return Simd.TensorFormat(Lib.lib.SynetTensorFormatGet(self.__tensor))
	
	def Type(self) -> Simd.TensorData:
		return Simd.TensorData(Lib.lib.SynetTensorTypeGet(self.__tensor))
	
	def Name(self) -> str:
		ptr = Lib.lib.SynetTensorName(self.__tensor)
		return str(ptr, encoding='utf-8')
	
	def Data(self) -> ctypes.c_void_p:
		return Lib.lib.SynetTensorData(self.__tensor)
	
	def As32f(self) -> ctypes.POINTER(ctypes.c_float):
		if self.Type() != Simd.TensorData.FP32 :
			raise Exception("Tensor data type is not FP32!")
		return ctypes.cast(self.Data(), ctypes.POINTER(ctypes.c_float))


###################################################################################################

class Network():
	def __init__(self):
		self.__net = Lib.lib.SynetNetworkInit()
	
	def __del__(self) :
		if self.__net != 0 :
			Lib.lib.SynetRelease(self.__net)
			self.__net = 0
			
	def Load(self, model: str, weight: str) -> bool:
		if not os.path.isfile(model):
			print("Model file '{0}' is not exist!".format(model))
			return False
		if not os.path.isfile(weight):
			print("Weight file '{0}' is not exist!".format(weight))
			return False
		return Lib.lib.SynetNetworkLoad(self.__net, model.encode('utf-8'), weight.encode('utf-8'))
	
	def Empty(self) -> bool:
		return Lib.lib.SynetNetworkEmpty(self.__net)
	
	def Reshape(self, width: int, height: int, batch = 1) -> bool:
		return Lib.lib.SynetNetworkReshape(self.__net, width, height, batch)
	
	def SetBatch(self, batch: int) -> bool:
		return Lib.lib.SynetNetworkSetBatch(self.__net, batch)
	
	def SrcSize(self) -> int:
		return Lib.lib.SynetNetworkSrcSize(self.__net)
	
	def Src(self, index: int) -> Tensor:
		return Tensor(Lib.lib.SynetNetworkSrc(self.__net, index))
	
	def NchwShape(self) :
		if self.SrcSize() != 1:
			raise Exception("NchwShape works only for model with 1 input!")
		shape = self.Src(0).Shape()
		if len(shape) == 4 and self.Src(0).Format() == Simd.TensorFormat.Nhwc:
			shape = [shape[0], shape[3] , shape[1] , shape[2]]
		return shape
	
	def DstSize(self) -> int:
		return Lib.lib.SynetNetworkDstSize(self.__net)
	
	def Dst(self, index: int):
		return Tensor(Lib.lib.SynetNetworkDst(self.__net, index))
	
	def DstByName(self, name: str):
		return Tensor(Lib.lib.SynetNetworkDstByName(self.__net, name.encode('utf-8')))
	
	def CompactWeight(self):
		Lib.lib.SynetNetworkCompactWeight(self.__net)
	
	def Forward(self):
		Lib.lib.SynetNetworkForward(self.__net)
			
###################################################################################################
import argparse
import os
import ctypes
import pathlib
import sys
import array
import enum

###################################################################################################

class Synet():
	lib : ctypes.CDLL
	
	def __init__(self, dir: str):
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

		self.lib = ctypes.CDLL(path)
		
		self.lib.SynetVersion.argtypes = []
		self.lib.SynetVersion.restype = ctypes.c_char_p 
		
		self.lib.SynetRelease.argtypes = [ ctypes.c_void_p ]
	
		self.lib.SynetNetworkInit.argtypes = []
		self.lib.SynetNetworkInit.restype = ctypes.c_void_p 
		
		self.lib.SynetNetworkLoad.argtypes = [ ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p ]
		self.lib.SynetNetworkLoad.restype = ctypes.c_bool 
		
		self.lib.SynetNetworkSrcSize.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetNetworkSrcSize.restype = ctypes.c_size_t 

		self.lib.SynetNetworkSrc.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		self.lib.SynetNetworkSrc.restype = ctypes.c_void_p 
		
		self.lib.SynetNetworkDstSize.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetNetworkDstSize.restype = ctypes.c_size_t 

		self.lib.SynetNetworkDst.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		self.lib.SynetNetworkDst.restype = ctypes.c_void_p 
		
		self.lib.SynetTensorCount.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetTensorCount.restype = ctypes.c_size_t 
		
		self.lib.SynetTensorAxis.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		self.lib.SynetTensorAxis.restype = ctypes.c_size_t
		
		self.lib.SynetTensorFormat.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetTensorFormat.restype = ctypes.c_int
		
		self.lib.SynetTensorName.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetTensorName.restype = ctypes.c_char_p
		
	def Version(self) -> str: 
		ptr = self.lib.SynetVersion()
		return str(ptr, encoding='utf-8')
	
###################################################################################################

class TensorFormat(enum.Enum) :
	Unknown = -1
	NCHW = 0
	NHWC = 1
	
	
###################################################################################################

class Tensor():
	lib : ctypes.CDLL
	tensor : ctypes.c_void_p
	
	def __init__(self, tensor: ctypes.c_void_p, lib: ctypes.CDLL):
		self.lib = lib
		self.tensor = tensor
	
	def Format(self) -> int:
		return self.lib.SynetTensorFormat(self.tensor)
	
	def Shape(self):
		shape = []
		for axis in range(self.lib.SynetTensorCount(self.tensor)) :
			shape.append(self.lib.SynetTensorAxis(self.tensor, axis))
		return shape
	
	def Format(self) -> TensorFormat:
		return TensorFormat(self.lib.SynetTensorFormat(self.tensor))
	
	def Name(self) -> str:
		ptr = self.lib.SynetTensorName(self.tensor)
		return str(ptr, encoding='utf-8')
	

###################################################################################################

class Network():
	lib : ctypes.CDLL
	net : ctypes.c_void_p
	
	def __init__(self, synet: Synet):
		self.lib = synet.lib
		self.net = self.lib.SynetNetworkInit()
	
	def __del__(self) :
		if self.net != 0 :
			self.lib.SynetRelease(self.net)
			self.net = 0
			
	def Load(self, model: str, weight: str) -> bool:
		if not os.path.isfile(model):
			print("Model file '{0}' is not exist!".format(model))
			return False
		if not os.path.isfile(weight):
			print("Weight file '{0}' is not exist!".format(weight))
			return False
		return self.lib.SynetNetworkLoad(self.net, model.encode('utf-8'), weight.encode('utf-8'))
	
	def SrcSize(self) -> int:
		return self.lib.SynetNetworkSrcSize(self.net)
	
	def Src(self, index: int) -> Tensor:
		return Tensor(self.lib.SynetNetworkSrc(self.net, index), self.lib)
	
	def DstSize(self) -> int:
		return self.lib.SynetNetworkDstSize(self.net)
	
	def Dst(self, index: int):
		return Tensor(self.lib.SynetNetworkDst(self.net, index), self.lib)

			
###################################################################################################
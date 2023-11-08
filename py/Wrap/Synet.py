import argparse
import os
import ctypes
import pathlib
import sys
import array

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
		
		self.lib.SynetNetworkSrcCount.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetNetworkSrcCount.restype = ctypes.c_size_t 

		self.lib.SynetNetworkSrcDimCount.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		self.lib.SynetNetworkSrcDimCount.restype = ctypes.c_size_t 
		
		self.lib.SynetNetworkSrcDimValue.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t ]
		self.lib.SynetNetworkSrcDimValue.restype = ctypes.c_size_t 

		self.lib.SynetNetworkDstCount.argtypes = [ ctypes.c_void_p ]
		self.lib.SynetNetworkDstCount.restype = ctypes.c_size_t 
		
		self.lib.SynetNetworkDstDimCount.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		self.lib.SynetNetworkDstDimCount.restype = ctypes.c_size_t 
		
		self.lib.SynetNetworkDstDimValue.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t ]
		self.lib.SynetNetworkDstDimValue.restype = ctypes.c_size_t 
		
	def Version(self) -> str: 
		ptr = self.lib.SynetVersion()
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
	
	def SrcCount(self) -> int:
		return self.lib.SynetNetworkSrcCount(self.net)
	
	def SrcShape(self, index: int):
		shape = []
		for dim in range(self.lib.SynetNetworkSrcDimCount(self.net, index)) :
			shape.append(self.lib.SynetNetworkSrcDimValue(self.net, index, dim))
		return shape
	
	def DstCount(self) -> int:
		return self.lib.SynetNetworkDstCount(self.net)
	
	def DstShape(self, index: int):
		shape = []
		for dim in range(self.lib.SynetNetworkDstDimCount(self.net, index)) :
			shape.append(self.lib.SynetNetworkDstDimValue(self.net, index, dim))
		return shape

			
###################################################################################################
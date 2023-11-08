import argparse
import os
import ctypes
import pathlib
import sys

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

		
	def Version(self) -> str: 
		ptr = self.lib.SynetVersion()
		return str(ptr, encoding='utf-8')
	
	def Release(self, ptr: ctypes.c_void_p) : 
		self.lib.SynetRelease(ptr)
		
	def NetworkInit(self) -> ctypes.c_void_p : 
		return self.lib.SynetNetworkInit()
	
	def NetworkLoad(self, network: ctypes.c_void_p, model: ctypes.c_char_p, weight: ctypes.c_char_p) -> bool: 
		return self.lib.SynetNetworkLoad(network, model, weight)
		

###################################################################################################

class Network():
	synet: Synet
	net : ctypes.c_void_p
	
	def __init__(self, synet: Synet):
		self.synet = synet
		self.net = self.synet.NetworkInit()
	
	def __del__(self) :
		if self.net != 0 :
			self.synet.Release(self.net)
			self.net = 0
			
	def Load(self, model: str, weight: str) -> bool:
		if not os.path.isfile(model):
			print("Model file '{0}' is not exist!".format(model))
			return False
		if not os.path.isfile(weight):
			print("Weight file '{0}' is not exist!".format(weight))
			return False
		return self.synet.NetworkLoad(self.net, model.encode('utf-8'), weight.encode('utf-8'))
			
###################################################################################################
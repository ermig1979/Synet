import argparse
import os
import ctypes
import pathlib

###################################################################################################

def Error(text = "") :
	if text != "" :
		print(text)
	return 1

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Synet", description="Synet Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Path to binary file.", required=False, type=str, default="Synet.dll")
	args = parser.parse_args()
	
	if not os.path.isfile(args.bin):
		return Error("Binary file '{0}' is not exist!".format(args.bin))
	
	libPath = str(pathlib.Path().absolute() / str(args.bin))
	lib = ctypes.CDLL(str(libPath))
	lib.SynetVersion.restype = ctypes.c_char_p 
	
	ver = lib.SynetVersion()
	
	print("Synet version: {0} \n".format(ver))
	
	#c_lib = ctypes.CDLL(str(args.bin))
	

	print("Synet Python Wrapper ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

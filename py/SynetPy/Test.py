import argparse
import os
import ctypes
import pathlib
import sys

import Synet
import Simd

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Synet", description="Synet Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	parser.add_argument("-m", "--model", help="Path to model file.", required=False, type=str, default="synet.xml")
	parser.add_argument("-w", "--weight", help="Path to weight file.", required=False, type=str, default="synet.bin")
	parser.add_argument("-i", "--image", help="Path to image file.", required=False, type=str, default="face_000.jpg")
	args = parser.parse_args()
	
	Simd.Lib.Init(args.bin)
	Synet.Lib.Init(args.bin)
	
	print("Synet version: {0}. \n".format(Synet.Lib.Version()))
	
	network = Synet.Network()
	
	if not network.Load(args.model, args.weight) :
		return 1
	print("Load network from {0} and {1}.".format(args.model, args.weight))

	print("Set batch = 1.")	
	network.SetBatch(1)
	
	print("Network inputs:")
	for src in range(network.SrcSize()) :
		tensor = network.Src(src)
		print(" {0}: {1} {2} {3} {4}".format(src, tensor.Name(), tensor.Shape(), tensor.Type().name, tensor.Format().name))
		
	print("Network outputs:")
	for dst in range(network.DstSize()) :
		tensor = network.Dst(dst)
		print(" {0}: {1} {2} {3} {4}".format(dst, tensor.Name(), tensor.Shape(), tensor.Type().name, tensor.Format().name))

	image = Simd.Image()
	if not image.Load(args.image) :
		return 1
	print("Load test image {0}.".format(args.image))	
	
	shape = network.NchwShape()
	resized = Simd.Resized(image, shape[3], shape[2], Simd.ResizeMethod.Area)
	lower = [0.0, 0.0, 0.0]
	upper = [1.0, 1.0, 1.0]
	input = network.Src(0)
	Simd.SynetSetInput(resized, lower, upper, input.Data(), shape[1], input.Format())
	print("Set network input.")	
	
	network.Forward()
	print("Network inference.")		
	
	print("Network output:")	
	dst = network.DstByName("output").As32f()
	for offs in range(5) :
		print(" {0} ".format(dst[offs]))
	print(" ... ")
	
	print("\nSynet Python Wrapper ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

import argparse
import os
import ctypes
import pathlib
import sys

import Synet

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Synet", description="Synet Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	parser.add_argument("-m", "--model", help="Path to model file.", required=False, type=str, default="synet.xml")
	parser.add_argument("-w", "--weight", help="Path to weight file.", required=False, type=str, default="synet.bin")
	args = parser.parse_args()
	
	synet = Synet.Synet(args.bin)
	
	print("Synet version: {0}. \n".format(synet.Version()))
	
	network = Synet.Network(synet)
	
	if not network.Load(args.model, args.weight) :
		return 1
	print("Load network from {0} and {1}.".format(args.model, args.weight))
	
	print("Network inputs:")
	for src in range(network.SrcSize()) :
		tensor = network.Src(src)
		print(" {0}: {1} {2} {3}".format(src, tensor.Name(), tensor.Shape(), tensor.Format().name))
	
	print("Network outputs:")
	for dst in range(network.DstSize()) :
		tensor = network.Dst(dst)
		print(" {0}: {1} {2} {3}".format(dst, tensor.Name(), tensor.Shape(), tensor.Format().name))
	
	print("\nSynet Python Wrapper ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

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
	
	print("Synet version: {0} \n".format(synet.Version()))
	
	network = Synet.Network(synet)
	
	if not network.Load(args.model, args.weight) :
		return 1
	
	#network = synet.NetworkInit()
	
	#synet.Release(network)
	
	print("Synet Python Wrapper ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

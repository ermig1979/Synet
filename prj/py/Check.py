import argparse
import os
from datetime import datetime

###################################################################################################

class Test():
	def __init__(self):
		self.framework = ""
		self.group = ""
		self.name = ""
		self.image = ""
		self.batch = False
		self.path = ""
		
###################################################################################################

def GetTestList(args):
	tests = []
	listPath = args.src + os.path.sep + "tests.txt"
	print(listPath)
	if not os.path.isfile(listPath):
		print("Test list '{}' is not exist!".format(listPath))
		return False, tests
	listFile = open(listPath, "r")
	while True:
		line = listFile.readline()
		if not line:
			break
		vals = line.split()
		if len(vals) < 5 :
			continue
		test = Test()
		test.framework = vals[0]
		test.group = vals[1]
		test.name = vals[2]
		test.image = vals[3]
		test.batch = vals[4]
		
		if not (test.framework == "onnx" or test.framework == "inference_engine") :
			return False, tests
		
		test.path = test.framework
		if test.group != "root" :
			test.path += os.path.sep + test.group
		test.path += os.path.sep + test.name
		if args.include != "" and test.path.find(args.include) == -1:
			continue
		if args.exclude != "" and test.path.find(args.exclude) != -1:
			continue
		
		tests.append(test)
	listFile.close()
	return True, tests

###################################################################################################

def RunTest(args, dstPath, test):
	binPath = args.bin + os.path.sep + "test_" + test.framework
	if not os.path.isfile(binPath):
		print("Binary '{}' is not exist!".format(binPath))
		return False
	testPath = args.src + os.path.sep + test.path
	if not os.path.isdir(testPath):
		print("Test directory '{}' is not exist!".format(testPath))
		return False
	return True

###################################################################################################

def main():

	parser = argparse.ArgumentParser(prog="Check", description="Synet tests check script.")
	parser.add_argument("-s", "--src", help="Tests data path.", required=False, type=str, default="./data")
	parser.add_argument("-b", "--bin", help="Tests binary path.", required=False, type=str, default="./build")
	parser.add_argument("-d", "--dst", help="Output tests path.", required=False, type=str, default="../test/check")
	parser.add_argument("-t", "--threads", help="Tests threads number.", required=False, type=int, default=1)
	parser.add_argument("-f", "--fast", help="Fast check flag (no conversion, small image count).", required=False, type=bool, default=False)
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, type=str, default="")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, type=str, default="")
	args = parser.parse_args()
	
	if not os.path.isdir(args.src):
		print("Test data directory '", args.src, "' is not exist!")
		return 1

	if not os.path.isdir(args.bin):
		print("Binary directory '", args.bin, "' is not exist!")
		return 1
	
	now = datetime.now();
	dstPath = args.dst + "_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "__" + str(now.hour) + "_" + str(now.minute)
	print("Test output directory: ", dstPath)
	if not os.path.isdir(dstPath):
		try:
			os.makedirs(dstPath)
		except OSError as error: 
			print("Can't create output directory '" , dstPath, "' !")
			return 1
	
	print("Synet Check:", args)	
	
	result : bool
	result, tests = GetTestList(args)
	if not result:
		return 1
	
	count = 0
	for test in tests:
		print("Test {}: {} {} {} {} {} ".format(count, test.framework, test.group, test.name, test.image, test.batch))
		if not RunTest(args, dstPath, test):
			return 1
		count += 1

	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

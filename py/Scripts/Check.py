import argparse
import os
import datetime
import threading
import multiprocessing
import subprocess
import random

###################################################################################################

class Test():
	def __init__(self, vals, args):
		self.framework = vals[0]
		self.group = vals[1]
		self.name = vals[2]
		self.image = vals[3]
		self.batch = vals[4]
		if len(vals) > 5 and args.bf16 :
			self.bf16 = vals[5]
		else :
			self.bf16 = "0"
		self.path = ""
		
	def TestNum(self) -> int :
		if self.framework == "synet" :
			if self.bf16 == "1" and self.batch == "0" :
				return 1
			elif self.bf16 == "1" and self.batch == "1" :
				return 2
			else :
				return 0
		else:
			if self.batch == "1" and self.bf16 == "1":
				return 5
			elif self.batch == "1" and self.bf16 == "0" :
				return 3
			elif self.batch == "0" and self.bf16 == "1" :
				return 3
			else :
				return 2
	def Bin(self) -> str :
		if self.framework == "synet" :
			return "test_bf16"
		else :
			return "test_" + self.framework
		
###################################################################################################

class Context():
	def __init__(self, args):
		self.args = args
		self.tests = []
		self.total = 0
		self.curr = 0
		self.block = 0
		self.lock = multiprocessing.Lock()
		self.dst = ""
		self.error = False

	def UpdateProgress(self, log, out) :
		if self.error :
			return
		with self.lock:
			self.curr += 1;
			print("Test {0} of {1} log to : {2} \n{3}".format(self.curr, self.total, log, out))
			
	def Error(self, text="") -> bool :
		if text != "" :
			print(text)
		self.error = True
		return False
	
###################################################################################################

def CheckDirs(context : Context) :
	args = context.args
	
	if not os.path.isdir(args.src):
		return context.Error("Test data directory '{0}' is not exist!".format(args.src))
	
	if not os.path.isdir(args.bin):
		return context.Error("Binary directory '{0}' is not exist!".format(args.bin))
	
	context.dst = args.dst + os.path.sep
	now = datetime.datetime.now()
	context.dst += "{0:04d}_{1:02d}_{2:02d}__{3:02d}_{4:02d}_{5:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
	if args.fast :
		context.dst += "f"
	if len(args.include) > 0 :
		context.dst += "_dbg"
	if not os.path.isdir(context.dst):
		try:
			os.makedirs(context.dst)
		except OSError as error: 
			return context.Error("Can't create output directory '{0}' !".format(context.dst))
	
	return True
		
###################################################################################################

def SetTestList(context : Context):
	args = context.args
	if args.list == "" :
		listPath = args.src + os.path.sep + "tests.txt"
	else :
		listPath = args.list
	if not os.path.isfile(listPath):
		return context.Error("Test list '{}' is not exist!".format(listPath))
	listFile = open(listPath, "r")
	while True:
		line = listFile.readline()
		if not line:
			break
		vals = line.split()
		if len(vals) < 5 :
			continue
		if vals[0][0] == '#' :
			continue
		test = Test(vals, args)
		
		if not (test.framework == "onnx" or test.framework == "inference_engine" or test.framework == "synet") :
			return context.Error("Unknown framework: {0} !".format(test.framework))
		
		test.path = test.framework
		if test.group != "root" :
			test.path += os.path.sep + test.group
		test.path += os.path.sep + test.name
		
		if len(args.include) > 0 :
			skip = True
			for include in args.include :
				if test.path.find(include) != -1 :
					skip = False
			if skip :
				continue
			
		if len(args.exclude) > 0 :
			skip = False
			for exclude in args.exclude :
				if test.path.find(exclude) != -1 :
					skip = True
			if skip :
				continue
		
		context.total += test.TestNum()
		context.tests.append(test)
	listFile.close()
	if len(context.tests) == 0 :
		return context.Error("There are no tests found for given include({0}) and exclude({1}) filters in file '{2}'!".format(args.include, args.exclude, listPath))
	return True

###################################################################################################

def ValidateThreadNumber(context : Context):
	args = context.args
	count = len(context.tests)
	if args.threads < 1 or args.threads > multiprocessing.cpu_count() :
		args.threads = multiprocessing.cpu_count()
	args.threads = min(args.threads, count)
	context.block = (count + args.threads - 1) // args.threads
	args.threads = (count + context.block - 1) // context.block

###################################################################################################

def RunTest(context, test, format, batch, bf16):
	if context.error :
		return False
	args = context.args
	binPath = args.bin + os.path.sep + test.Bin()
	if not os.path.isfile(binPath):
		return context.Error("Binary '{0}' is not exist!".format(binPath))
	
	testPath = args.src + os.path.sep + test.path
	if not os.path.isdir(testPath):
		return context.Error("Test directory '{0}' is not exist!".format(testPath))
	
	imagePath = ""
	if test.image == "local" :
		imagePath = testPath + os.path.sep + "image"
	else :
		imagePath = args.src + os.path.sep + "images" + os.path.sep + test.image
	if not os.path.isdir(imagePath):
		return context.Error("Image directory '{0}' is not exist!".format(imagePath))
	
	log = context.dst + os.path.sep
	if test.group == "root" :
		log = log + "c{0}_{1}_f{2}_b{3}".format(test.framework[0], test.name, format, batch)
	else :
		log = log + "c{0}_{1}__{2}_f{3}_b{4}".format(test.framework[0], test.group, test.name, format, batch)
	if bf16 == "1" :
		log = log + "_bf16.txt"
	else :
		log = log + "_fp32.txt"

	if bf16 == "1" :
		threshold = args.bf16Threshold
	elif test.framework == "inference_engine" :
		threshold = args.inferenceEngineThreshold
	else :	
		threshold = args.onnxThreshold
	pathArgs = ""
	if test.framework == "inference_engine" :
		pathArgs += "-fm={0}/other.xml -fw={0}/other.bin".format(testPath)
	elif test.framework == "onnx" :
		pathArgs += "-fw={0}/other.onnx".format(testPath)
	elif test.framework == "synet" :
		pathArgs += "-fm={0}/synet1.xml -fw={0}/synet1.bin".format(testPath)
	if bf16 == "1" :
		pathArgs += " -sm={0}/synet2.xml -sw={0}/synet2.bin".format(testPath)
	else :
		pathArgs += " -sm={0}/synet{1}.xml -sw={0}/synet{1}.bin".format(testPath, format)
	pathArgs += " -id={1} -od={0}/output -tp={0}/param.xml".format(testPath, imagePath)
	
	trashFile = imagePath + os.path.sep + "descript.ion"
	if os.path.isfile(trashFile) :
		os.remove(trashFile)

	os.environ['LD_LIBRARY_PATH'] = args.bin
	
	out = ""
	if not args.fast and batch == 1 :
		cmd = "{0} -m=convert {1} -tf={2} -cs=1 -bf={3}".format(binPath, pathArgs, format, bf16)
		result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
		out += result.stdout.decode('utf-8')
		if result.returncode != 0 :
			return context.Error("Conversion error in test {0} :\n{1}".format(log, out))
		
	num = 2 if args.fast else 10
	cmd = "{0} -m=compare -e=3 {1} -rn=1 -wt=1 -tt=0 -ie={2} -be={2} -tf={3} -bs={4} -ct={5} -cs=1 -ln={6} -pl={7} -bf={8} -cp={9:d}".format(binPath, pathArgs, num, format, batch, threshold, log, args.performanceLog, bf16, args.comparePrecise)
	result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
	out += result.stdout.decode('utf-8')
	if result.returncode != 0 :
		return context.Error("Compare error in test {0} :\n{1}".format(log, out))

	context.UpdateProgress(log, out[:len(out)-1])
	
	return True

###################################################################################################

def SingleThreadRun(context, beg, end):
	for i in range(beg, end):
		test = context.tests[i]
		if test.framework != "synet" :
			if not RunTest(context, test, 0, 1, "0") :
				return
			if not RunTest(context, test, 1, 1, "0") :
				return
			if test.batch == "1" :
				if not RunTest(context, test, 1, 2, "0") :
					return
		if test.bf16 == "1" :
			if not RunTest(context, test, 1, 1, "1") :
				return
		if test.batch == "1" and test.bf16 == "1" :
			if not RunTest(context, test, 1, 2, "1") :
				return

###################################################################################################

def MultiThreadRun(context):
	threads = []
	random.shuffle(context.tests)
	for t in range(context.args.threads) :
		beg = t * context.block
		end = min(beg + context.block, len(context.tests))
		thread = threading.Thread(target=SingleThreadRun, args=(context, beg, end))
		threads.append(thread)
		thread.start()
	
	for thread in threads :
		thread.join()

###################################################################################################

def main():

	parser = argparse.ArgumentParser(prog="Check", description="Synet tests check script.")
	parser.add_argument("-s", "--src", help="Tests data path.", required=False, type=str, default="./data")
	parser.add_argument("-l", "--list", help="Alternative test list path.", required=False, type=str, default="")	
	parser.add_argument("-b", "--bin", help="Tests binary path.", required=False, type=str, default="./build")
	parser.add_argument("-d", "--dst", help="Output tests path.", required=False, type=str, default="../test/check")
	parser.add_argument("-t", "--threads", help="Tests threads number.", required=False, type=int, default=-1)
	parser.add_argument("-f", "--fast", help="Fast check flag (no model conversion, small number of test images).", required=False, type=bool, default=False)
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, default=[], action="append")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, default=[], action="append")
	parser.add_argument("-pl", "--performanceLog", help="Level of performance log: (0 - no statistics, 1 - averaged, 2 - detailed).", required=False, type=int, default="0")
	parser.add_argument("-bf", "--bf16", help="Run BF16 tests.", required=False, type=bool, default=False)
	parser.add_argument("-it", "--inferenceEngineThreshold", help="Threshold for Inference Engine tests.", required=False, type=float, default=0.000270)
	parser.add_argument("-ot", "--onnxThreshold", help="Threshold for OnnxRuntime tests.", required=False, type=float, default=0.001317)
	parser.add_argument("-bt", "--bf16Threshold", help="Threshold for BF16 tests.", required=False, type=float, default=0.013339)
	parser.add_argument("-cp", "--comparePrecise", help="Compare output precise (element-wise).", required=False, type=bool, default=True)
	context = Context(parser.parse_args())
	
	if not CheckDirs(context) :
		return 1
	
	if not SetTestList(context):
		return 1
	
	ValidateThreadNumber(context)
	
	print("\nSynet start checking in {0} threads: \n".format(context.args.threads))
	
	start = datetime.datetime.now()
	
	if context.args.threads == 1 :
		SingleThreadRun(context, 0, len(context.tests))
	else :
		MultiThreadRun(context)
		
	elapsed = datetime.datetime.now() - start
	
	if not context.error :
		print("All tests finished successfully in {0}:{1:02d}:{2:02d} !\n".format(elapsed.seconds // 3600, elapsed.seconds % 3600 // 60, elapsed.seconds % 60))
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

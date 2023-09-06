import argparse
from asyncio import threads
from itertools import count
import os
from datetime import datetime
from re import I
import threading
import multiprocessing
import subprocess

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

class Context():
	def __init__(self, args):
		self.args = args
		self.tests = []
		self.total = 0
		self.curr = 0
		self.lock = multiprocessing.Lock()
		self.dst = ""
		self.error = False
		
	def SetDst(self, args) :
		now = datetime.now()
		self.dst = "{0}_{1:04d}_{2:02d}_{3:02d}__{4:02d}_{5:02d}".format(args.dst, now.year, now.month, now.day, now.hour, now.minute)
		if args.fast :
			self.dst += "f"
		if args.include != "" or args.exclude != "" :
			self.dst += "d"
		if not os.path.isdir(self.dst):
			try:
				os.makedirs(self.dst)
			except OSError as error: 
				print("Can't create output directory '" , self.dst, "' !")
				return False
		return True
		
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

def GetTestList(context : Context):
	args = context.args
	listPath = args.src + os.path.sep + "tests.txt"
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
		test = Test()
		test.framework = vals[0]
		test.group = vals[1]
		test.name = vals[2]
		test.image = vals[3]
		test.batch = vals[4]
		
		if not (test.framework == "onnx" or test.framework == "inference_engine") :
			return context.Error("Unknown framework: {0} !".format(test.framework))
		
		test.path = test.framework
		if test.group != "root" :
			test.path += os.path.sep + test.group
		test.path += os.path.sep + test.name
		if args.include != "" and test.path.find(args.include) == -1:
			continue
		if args.exclude != "" and test.path.find(args.exclude) != -1:
			continue
		
		if test.batch == "1" :
			context.total += 3
		else :
			context.total += 2
		context.tests.append(test)
	listFile.close()
	if len(context.tests) == 0 :
		return context.Error("There are no tests found for given include({0}) and exclude({1}) filters in file '{2}'!".format(args.include, args.exclude, listPath))
	return True

###################################################################################################

def RunTest(context, test, format, batch):
	if context.error :
		return False
	args = context.args
	binPath = args.bin + os.path.sep + "test_" + test.framework
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
	
	log = ""
	if test.group == "root" :
		log = context.dst + os.path.sep + "c{0}_{1}_f{2}_b{3}.txt".format(test.framework[0], test.name, format, batch)
	else :
		log = context.dst + os.path.sep + "c{0}_{1}__{2}_f{3}_b{4}.txt".format(test.framework[0], test.group, test.name, format, batch)

	threshold = 0.0031
	pathArgs = "-fm={0}/other.dsc -fw={0}/other.dat -sm={0}/synet{1}.xml -sw={0}/synet{1}.bin -id={2} -od={0}/output -tp={0}/param.xml".format(testPath, format, imagePath)
	
	trashFile = imagePath + os.path.sep + "descript.ion"
	if os.path.isfile(trashFile) :
		os.remove(trashFile)
	
	out = ""
	if not args.fast and batch == 1 :
		cmd = "{0} -m=convert {1} -tf={2} -cs=1".format(binPath, pathArgs, format)
		result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
		out += result.stdout.decode('utf-8')
		if result.returncode != 0 :
			return context.Error("Error in test {0} :\n{1}".format(log, out))
		
	num = 2 if args.fast else 10
	cmd = "{0} -m=compare -e=3 {1} -rn=1 -wt=1 -tt=0 -ie={2} -be={2} -tf={3} -bs={4} -ct={5} -cs=1 -ln={6}".format(binPath, pathArgs, num, format, batch, threshold, log)
	result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
	out += result.stdout.decode('utf-8')
	if result.returncode != 0 :
		return context.Error("Error in test {0} :\n{1}".format(log, out))

	context.UpdateProgress(log, out[:len(out)-1])
	
	return True

###################################################################################################

def SingleThreadRun(context, tests, beg, end):
	for i in range(beg, end):
		test = tests[i]
		if not RunTest(context, test, 0, 1) :
			return
		if not RunTest(context, test, 1, 1) :
			return
		if test.batch == "1" :
			if not RunTest(context, test, 1, 2) :
				return

###################################################################################################

def MultiThreadRun(context, tests):
	args = context.args
	count = len(tests)
	tasks = args.threads
	if tasks < 1 or tasks > multiprocessing.cpu_count() :
		tasks = multiprocessing.cpu_count()
	tasks = min(tasks, count)
	block = count // tasks
	tasks = count // block
	
	threads = []
	for t in range(tasks) :
		beg = t * block
		end = min(beg + block, count)
		thread = threading.Thread(target=SingleThreadRun, args=(context, tests, beg, end))
		threads.append(thread)
		thread.start()
	
	for thread in threads :
		thread.join()

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
	context = Context(parser.parse_args())
	
	args = context.args
	
	if not os.path.isdir(args.src):
		print("Test data directory '", args.src, "' is not exist!")
		return 1

	if not os.path.isdir(args.bin):
		print("Binary directory '", args.bin, "' is not exist!")
		return 1
	
	if not context.SetDst(args) :
		return 1
	
	print("\nSynet start checking in {0} threads: \n".format(args.threads))
	
	if not GetTestList(context):
		return 1
	
	if args.threads == 1 :
		SingleThreadRun(context, context.tests, 0, len(context.tests))
	else :
		MultiThreadRun(context, context.tests)
	
	if not context.error :
		print("All test finished succefully!\n")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()

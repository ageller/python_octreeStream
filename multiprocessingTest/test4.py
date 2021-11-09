''' Attempting to write an octree in python

I want this to work with VERY large data sets that can't be stored fully in memory.  So my procedure will be as follows:
- need to read in line-by-line and clear memory every X MB (or maybe every X particles;can I check memory load in python?)
- go down to nodes with containing N particles
- need to write out tree with node sizes and centers and also ending nodes with actual particles
'''

import os
import numpy as np
import json
import h5py
import random
from multiprocessing import Process, Manager

#https://stackoverflow.com/questions/56250514/how-to-tackle-with-error-object-of-type-int32-is-not-json-serializable
#to help with dumping to json
class npEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.int32):
			return int(obj)
		return json.JSONEncoder.default(self, obj)


class octreeStream:
	def __init__(self, inputFile, NMemoryMax = 1e5, NNodeMax = 5000, 
				 header = 0, delim = None, colIndices = {'Coordinates':[0,1,2]},
				 baseDir = 'octreeNodes', Nmax=np.inf, verbose=0, path = None, minWidth=0, 
				 h5PartKey = '', keyList = ['Coordinates'], center = None, cleanDir = False, 
				 Ncores=1):
		'''
			inputFile : path to the file. For now only text files.
			NMemoryMax : the maximum number of particles to save in the memory before writing to a file
			NNodeMax : the maximum number of particles to store in a node before splitting it
			header : the line number of the header (file starts at line 1, 
				set header=0 for no header, and in that case x,y,z are assumed to be the first three columns)
			delim : the delimiter between columns, if set to None, then hdf5 file is assumed
			colIndices : dict with the column numbers for each value in keyList (only necessary for csv files)
			baseDir : the directory to store the octree files
			Nmax : maximum number of particles to include
			verbose : controls how much output to write to the console
			path : the path to the output file
			minWidth : the minimum width that a node can have
			h5PartKey : if needed, can be used to specify which particle type to use, e.g. 'PartType0'
			keyList : Any additional keys that are desired; MUST contain the key to Coordinates first.  If blank, then assume that x,y,z is the first 3 columns in file
			center : options for the user to provide the octree center (can save time)
			cleanDir : if true this will erase the files within that directory before beginning
			Ncores : number of cores for multiprocessing
		'''
		self.nodes = Manager().list() #will contain a list of all nodes with each as a dict


		self.managerDict = Manager().dict()
		self.managerDict['inputFile'] = inputFile
		self.managerDict['NMemoryMax'] = NMemoryMax
		self.managerDict['NNodeMax'] = NNodeMax
		self.managerDict['header'] = header
		self.managerDict['delim'] = delim
		self.managerDict['colIndices'] = colIndices
		self.managerDict['minWidth'] = minWidth
		self.managerDict['h5PartKey'] = h5PartKey
		self.managerDict['keyList'] = keyList
		self.managerDict['center'] = center
		self.managerDict['Nmax'] = Nmax
		self.managerDict['cleanDir'] = cleanDir
		self.managerDict['Ncores'] = Ncores
		self.managerDict['verbose'] = verbose

		if (path is None):
			self.managerDict['path'] = os.path.join(os.getcwd(), baseDir)
		else:
			self.managerDict['path'] = os.path.abspath(path) #to make this windows safe
		print('files will be output to:', self.managerDict['path'])

		self.managerDict['count'] = 0
		self.managerDict['lineN'] = 0
		self.managerDict['arr'] = None #will contain the data from the file


		self.managerDict['width'] = None #will be determined in getSizeCenter


		
	def createNode(self, center, id='', width=0,):
		#node = Manager().dict(x=center[0], y=center[1], z=center[2], width=width, Nparticles=0, id=id, parentNodes=Manager().list(), childNodes=Manager().list(), particles=Manager().list(), needsUpdate=True)
		node = Manager().dict(x=center[0], y=center[1], z=center[2], width=width, Nparticles=0, id=id, parentNodes=[], childNodes=[], particles=[], needsUpdate=True)
		#node = dict(x=center[0], y=center[1], z=center[2], width=width, Nparticles=0, id=id, parentNodes=Manager().list(), childNodes=Manager().list(), particles=Manager().list(), needsUpdate=True)
		self.nodes += [node]
		print('CHECKING NEW NODE', self.nodes[-1])
		return (node, len(self.nodes) - 1)
	
	def findClosestNodeIndexByDistance(self, point, positions):
		#there is probably a faster and more clever way to do this
		#print('checking dist', point.shape, positions.shape, point, positions)
		dist2 = np.sum((positions - point)**2, axis=1)
		return np.argmin(dist2)
	
	def findClosestNode(self, point, parentIndex=None):
		#I am going to traverse the octree to find the closest node
		if (parentIndex is None):
			parentIndex = 0
		print('CHECKING HERE', parentIndex, self.nodes, len(self.nodes))
		for i,n in enumerate(self.nodes):
			print('PRINTING', i, n)
		parent = self.nodes[parentIndex]
		print('checking again', parent['width'])
		childIndices = parent['childNodes']

		while (childIndices != []):
			childPositions = []
			for i in childIndices:
				childPositions.append([self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']])
			parentIndex = childIndices[self.findClosestNodeIndexByDistance(point[0:3], np.array(childPositions))]
			parent = self.nodes[parentIndex]
			childIndices = parent['childNodes']

		return (parent, parentIndex)


	def initialize(self):

		self.managerDict['count'] = 0

		#create the output directory if needed
		if (not os.path.exists(self.managerDict['path'])):
			os.makedirs(self.managerDict['path'])
			
		#remove the files in that directory
		if (self.managerDict['cleanDir']):
			for f in os.listdir(self.managerDict['path']):
				os.remove(os.path.join(self.managerDict['path'], f))

		#create the base node
		(n, index) = self.createNode(self.managerDict['center'], '0', width=self.managerDict['width']) 

		#for some reason when running with multiprocessing, I need to return a value here.   Maybe this is way to make python wait for this to complete before moving on?
		return (n, index)



	def addPointToOctree(self, point):
		#find the node that it belongs in 
		node, index = self.findClosestNode(np.array(point))
		if (self.managerDict['verbose'] > 2):
			print('id, Nparticles', self.nodes[index]['id'], self.nodes[index]['Nparticles'], point)
			
		#add the particle to the node
		self.nodes[index]['particles'] += [point]
		self.nodes[index]['needsUpdate'] = True
		self.nodes[index]['Nparticles'] += 1

		if (self.managerDict['verbose'] > 2):
			print('After, id, Nparticles', self.nodes[index]['id'], self.nodes[index]['Nparticles'])

		#check if we need to split the node
		if (node['Nparticles'] >= self.managerDict['NNodeMax'] and node['width'] >= self.managerDict['minWidth']*2):
			self.createChildNodes(index) 



	def test(self, index):
		print('BEFORE',self.nodes[index]['Nparticles'], self.nodes[index]['childNodes'])
		self.nodes[index]['Nparticles'] += 1
		self.nodes[index]['childNodes'] += [index]
		print('AFTER',self.nodes[index]['Nparticles'], self.nodes[index]['childNodes'])

	def compileOctree(self, inputFile=None, append=False):

		#initialize a few things
		if (not append):
			self.managerDict['center'] = [0,0,0]
			self.managerDict['width'] = 1000
			_ = self.initialize()

		# if (inputFile is None):
		# 	inputFile = self.managerDict['inputFile']

		# #open the input file
		# if (self.managerDict['delim'] is None):
		# 	#assume this is a hdf5 file
		# 	file = h5py.File(os.path.abspath(inputFile), 'r')
		# 	arr = file
		# 	if (self.managerDict['h5PartKey'] != ''):
		# 		arrPart = arr[self.managerDict['h5PartKey']]

		# 	#now build the particle array
		# 	for i, key in enumerate(self.managerDict['keyList']):
		# 		if (i == 0):
		# 			arr = np.array(arrPart[key]) #Coordinates are always first
		# 		else:
		# 			addOn = np.array(arrPart[key])
		# 			arrLen = 1
		# 			if (key == 'Velocities'): #requires special handling because it is a 2D array
		# 				arrLen = 3
		# 			arr = np.hstack((arr, np.reshape(addOn, (len(arr),arrLen))))

		# else:
		# 	#for text files
		# 	file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          
		# 	arr = file

		# self.managerDict['Nmax'] = min(self.managerDict['Nmax'], arr.shape[0])

		ntest = 1
		jobs = []
		for i in range(ntest):
			center = [i,i,i]
			iden = 'test' + str(i)
			width = i*100
			jobs.append(Process(target=self.test, args=(0,)))
			#jobs.append(Process(target=self.addToNodes, args=(center, iden, width,)))
			#jobs.append(Process(target=self.findClosestNode, args=(center,)))
		for j in jobs:
			j.start()
		print('joining')
		for j in jobs:
			j.join()


		#self.iterFileOctree(arr)

		#file.close()


		print('done', self.nodes)

	def iterFileOctree(self, arr):
		#begin the loop to read the file line-by-line
		iStart = self.managerDict['header'];
		self.managerDict['lineN'] = iStart
		while self.managerDict['lineN'] < self.managerDict['Nmax']:
			jobs = []
			for i in range(self.managerDict['Ncores']):
				iEnd = int(np.floor(min(iStart + self.managerDict['NMemoryMax']/self.managerDict['Ncores'], self.managerDict['Nmax'])))
				print(iStart, iEnd, self.managerDict['lineN'], arr.shape[0])
				if (iStart >= iEnd):
					break
				j = Process(target=self.iterLinesOctree, args=(arr[iStart:iEnd], ))
				jobs.append(j)
				iStart = iEnd
				if (iEnd >= arr.shape[0]):
					break

			print('starting jobs', len(jobs), self.managerDict['lineN'], iEnd, self.managerDict['Nmax'])
			for j in jobs:
				j.start()

			print('joining jobs')
			for j in jobs:
				j.join()

			self.managerDict['lineN'] = 2.*self.managerDict['Nmax']

	def iterLinesOctree(self, arr):
		print("checking",arr.shape[0])
		for i in range(arr.shape[0]):
			line = arr[i]

			self.managerDict['lineN'] += 1
			self.managerDict['count'] += 1

			#get the x,y,z from the line 
			if (self.managerDict['delim'] is None):
				point = line
			else:
				lineStrip = line.strip().split(self.managerDict['delim'])
				point = []
				for key in self.managerDict['keyList']:
					indices =  self.managerDict['colIndices'][key]
					if (type(indices) is not list):
						indices = [indices]
					for ii in indices:
						point.append(float(lineStrip[ii]))

			self.addPointToOctree(point)
			
			if (self.managerDict['verbose'] > 0 and (self.managerDict['lineN'] % 100000 == 0)):
				print('line : ', self.managerDict['lineN'])




if __name__ == '__main__':
	oM1 = octreeStream('/Users/ageller/VISUALIZATIONS/FIREdata/m12i_res7100/snapdir_600/snapshot_600.0.hdf5', 
	                 h5PartKey = 'PartType0', keyList = ['Coordinates', 'Density', 'Velocities'],
	                 NNodeMax = 10000, NMemoryMax = 5e4, Nmax=1e5, verbose=3, minWidth=1e-4,
	                 cleanDir = True,
	                 path='/Users/ageller/VISUALIZATIONS/octree_threejs_python/WebGL_octreePartition/src/data/junk/octreeNodes/Gas')

	oM1.compileOctree()

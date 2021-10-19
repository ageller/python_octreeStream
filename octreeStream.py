''' Attempting to write an octree in python

I want this to work with VERY large data sets that can't be stored fully in memory.  So my procedure will be as follows:
- need to read in line-by-line and clear memory every X MB (or maybe every X particles;can I check memory load in python?)
- go down to nodes with containing N particles
- need to write out tree with node sizes and centers and also ending nodes with actual particles
'''

#TODO
# add ability to include other attributes than just positions
# create hdf5 reader; function that handles a line given a position and then a file feeder and array feeder
# will probably also want a way to add a new file or array onto the existing octree

import os
import numpy as np
import pandas as pd
import json
import h5py

#https://stackoverflow.com/questions/56250514/how-to-tackle-with-error-object-of-type-int32-is-not-json-serializable
#to help with dumping to json
class npEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.int32):
			return int(obj)
		return json.JSONEncoder.default(self, obj)

#I'll start with a csv file, though I want to eventually allow for hdf5 files
class octreeStream:
	def __init__(self, inputFile, NMemoryMax = 1e5, NNodeMax = 5000, 
				 header = 0, delim = None, xCol = 0, yCol = 1, zCol = 2,
				 baseDir = 'octreeNodes', Nmax=np.inf, verbose=0, path = None, minWidth=0, 
				 h5keyList = [], center = None):
		'''
			inputFile : path to the file. For now only text files.
			NMemoryMax : the maximum number of particles to save in the memory before writing to a file
			NNodeMax : the maximum number of particles to store in a node before splitting it
			header : the line number of the header (file starts at line 1, 
				set header=0 for no header, and in that case x,y,z are assumed to be the first three columns)
			delim : the delimiter between columns, if set to None, then hdf5 file is assumed
			xCol, yCol, zCol : the columns that have the x,y,z data
			baseDir : the directory to store the octree files
			Nmax : maximum number of particles to include
			verbose : controls how much output to write to the console
			path : the path to the output file
			minWidth : the minimum width that a node can have
			h5KeyList : sequential key list needed to access coordinates, e.g., ['PartType0', 'Coordinates']
			center : options for the user to provide the octree center (can save time)
		'''
		
		self.inputFile = inputFile
		self.NMemoryMax = NMemoryMax
		self.NNodeMax = NNodeMax
		self.header = header
		self.delim = delim
		self.xCol = xCol
		self.yCol = yCol
		self.zCol = zCol
		self.minWidth = minWidth
		self.h5keyList = h5keyList
		self.center = center

		self.nodes = None #will contain a list of all nodes with each as a dict
		self.baseNodePositions = None #will only contain the baseNodes locations as a list of lists (x,y,z)
		self.baseNodeIndices = None #will contain the index for each baseNode within the self.nodes array
		if (path is None):
			self.path = os.path.join(os.getcwd(), baseDir)
		else:
			self.path = os.path.abspath(path) #to make this windows safe
		print('files will be output to:', self.path)

		self.count = 0
		self.Nmax = Nmax

		self.verbose = verbose

		self.center = None #will be determined getSizeCenter
		self.width = None #will be determined in getSizeCenter

		
	def createNode(self, center, id='', width=0,):
		return dict(x=center[0], y=center[1], z=center[2], width=width,
					Nparticles=0, id=id, parentNodes=[], childNodes=[], particles=[], needsUpdate=True)
	
	def findClosestNode(self, point, positions):
		#there is probably a faster and more clever way to do this
		dist2 = np.sum((positions - point)**2, axis=1)
		return np.argmin(dist2)
	
	def createChildNodes(self, index):

		node = self.nodes[index]

		#split the node into 8 separate nodes and add these to self.nodes and self.baseNodePositions
		if (self.verbose > 0):
			print('creating child nodes', index, node['id'], node['Nparticles'], node['width'])


		#check if we need to read in the file (should this be a more careful check?)
		if (len(node['particles']) < self.NNodeMax): 
			self.populateNodeFromFile(node)


		#create the new nodes and add to the baseNodePositions array
		cx = node['x'] + node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] + node['width']/4.
		n1 = self.createNode([cx, cy, cz], node['id']+'_1', width=node['width']/2.)

		cx = node['x'] - node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] + node['width']/4.
		n2 = self.createNode([cx, cy, cz], node['id']+'_2',  width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] + node['width']/4.
		n3 = self.createNode([cx, cy, cz], node['id']+'_3', width=node['width']/2.)
		
		cx = node['x'] - node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] + node['width']/4.
		n4 = self.createNode([cx, cy, cz], node['id']+'_4', width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] - node['width']/4.
		n5 = self.createNode([cx, cy, cz], node['id']+'_5', width=node['width']/2.)

		
		cx = node['x'] - node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] - node['width']/4.
		n6 = self.createNode([cx, cy, cz], node['id']+'_6', width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] - node['width']/4.
		n7 = self.createNode([cx, cy, cz], node['id']+'_7', width=node['width']/2.)
		
		cx = node['x'] - node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] - node['width']/4.
		n8 = self.createNode([cx, cy, cz], node['id']+'_8', width=node['width']/2.)
		
		childIndices = np.array([], dtype='int')
		for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
			#add the parent and child indices to the nodes
			n['parentNodes'] = node['parentNodes'] + [index]
			childIndex = len(self.nodes)
			self.nodes.append(n)
			node['childNodes'].append(childIndex)

			#create these so that I can divide up the parent particles
			if (i == 0):
				childPositions = [[n['x'],n['y'],n['z']]]
			else:
				childPositions = np.append(childPositions, [[n['x'],n['y'],n['z']]], axis=0)
			childIndices = np.append(childIndices, childIndex)

		#divide up the particles 
		for p in node['particles']:
			childIndex = childIndices[self.findClosestNode(np.array([p]), childPositions)]
			self.nodes[childIndex]['particles'].append(p)
			self.nodes[childIndex]['Nparticles'] += 1      

			#add to the baseNodes
			if (childIndex not in self.baseNodeIndices):
				n = self.nodes[childIndex]
				self.baseNodePositions = np.append(self.baseNodePositions, [[n['x'],n['y'],n['z']]], axis=0)
				self.baseNodeIndices = np.append(self.baseNodeIndices, childIndex)
				if (self.verbose > 1):
					print('new baseNode position', n['id'], [[n['x'],n['y'],n['z']]])

		#remove the parent from the self.baseNodes arrays
		i = np.where(self.baseNodeIndices == index)[0]
		if (len(i) > 0 and i != -1):
			self.baseNodePositions = np.delete(self.baseNodePositions, i, axis=0)
			self.baseNodeIndices = np.delete(self.baseNodeIndices,i)
		else:
			print('!!!WARNING, did not find parent in baseNodes', i, index, self.baseNodeIndices)
		
		#remove the particles from the parent
		node['particles'] = []
		node['Nparticles'] = 0
		
		#check if we need to remove a file
		nodeFile = os.path.join(self.path, node['id'] + '.csv')
		if (os.path.exists(nodeFile)):
			os.remove(nodeFile)
			if (self.verbose > 0):
				print('removing file', nodeFile)
			
	def dumpNodesToFiles(self):
		#dump all the nodes to files
		if (self.verbose > 0):
			print('dumping nodes to files ...')
		
		#individual nodes
		for node in self.nodes:
			if ( (node['Nparticles'] > 0) and ('particles' in node) and (node['needsUpdate'])):

				parts = np.array(node['particles'])
				x = parts[:,0]
				y = parts[:,1]
				z = parts[:,2]
				df = pd.DataFrame(dict(x=x, y=y, z=z)).sample(frac=1).reset_index(drop=True) #shuffle the order
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				#check if the file exists
				mode = 'w'
				header = True
				if (os.path.exists(nodeFile)):
					mode = 'a'
					header = False
				df.to_csv(nodeFile, index=False, header=header, mode=mode)
				node['particles'] = []
				node['needsUpdate'] = False
				if (self.verbose > 1):
					print('writing node to file ', node['id'], mode)
				
		#node dict
		with open(os.path.join(self.path, 'octree.json'), 'w') as f:
			json.dump(self.nodes, f, cls=npEncoder)

		self.count = 0
				
		
	def checkNodeFiles(self):
		#check to make sure that only the base Nodes have files
		Nerror = 0
		if (self.baseNodeIndices is None):
			print('Please compile the octree first')
			return

		#first get names of all expected files
		names = []
		Nparts = []
		for index in self.baseNodeIndices:
			names.append(self.nodes[index]['id'] + '.csv')
			Nparts.append(self.nodes[index]['Nparticles'])

		avail = os.listdir(self.path)

		#now check the list of available files
		for fname in avail:
			if ( (fname not in names) and (fname != 'octree.json')):
				print('!!!WARNING: this file should not exist', fname)
				Nerror += 1

		#now check that all expected files exist
		for i, name in enumerate(names):
			if (name not in avail):
				print('!!!WARNING: this file does not exist', name, Nparts[i])
				Nerror += 1

		print('Number of bad files = ', Nerror)
		print('maximum number of particles in a file = ', max(Nparts))
		print('minimum number of particles in a file = ', min(Nparts))
			
	def populateAllNodesFromFiles(self, read = True):
		Nparts = 0
		NbaseNodes = 0

		#read in the octree from the json file
		if (read):
			with open(os.path.join(self.path,'octree.json')) as f:
				self.nodes = json.load(f)
				
		#also recreate the base nodes
		self.baseNodeIndices = np.array([],dtype='int')
		self.baseNodePositions = None
		for index, node in enumerate(self.nodes):
			if (self.nodes[index]['Nparticles'] > 0):
				Nparts += self.nodes[index]['Nparticles']
				NbaseNodes += 1
				self.baseNodeIndices = np.append(self.baseNodeIndices, index)
				if (self.baseNodePositions is None):
					self.baseNodePositions = np.array([[node['x'], node['y'], node['z']]])
				else:
					self.baseNodePositions = np.append(self.baseNodePositions, [[node['x'], node['y'], node['z']]], axis=0)
				self.populateNodeFromFile(self.nodes[index])
		print('Populated octree from files.')
		print(' -- total number of particles = ', Nparts)
		print(' -- total number of nodes = ', len(self.nodes))
		print(' -- total number of base nodes = ', NbaseNodes)
		

	def populateNodeFromFile(self, node):
		nodeFile = os.path.join(self.path, node['id'] + '.csv')
		if (self.verbose > 1):
			print('reading in file', nodeFile)
		df = pd.read_csv(nodeFile)
		node['particles'] += df.values.tolist()
		node['Nparticles'] = len(node['particles'])
		node['needsUpdate'] = True
		self.count += node['Nparticles']

	def shuffleAllParticlesInFiles(self):

		if (self.verbose > 0):
			print('randomizing particle order in data files ... ')

		#read in the octree from the json file
		with open(os.path.join(self.path,'octree.json')) as f:
			self.nodes = json.load(f)
				
		for index, node in enumerate(self.nodes):
			if (self.nodes[index]['Nparticles'] > 0):
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				df = pd.read_csv(nodeFile).sample(frac=1).reset_index(drop=True) #shuffle the order
				df.to_csv(nodeFile, index=False)

	def getSizeCenter(self, inputFile=None):
		#It will be easiest if we can get the center and the size at the start.  This will create overhead to read in the entire file...

		if (self.verbose > 0):
			print('calculating center and size ... ')

		if (inputFile is None):
			inputFile = self.inputFile
			
		#open the input file
		if (self.delim is None):
			#assume this is a hdf5 file
			file = h5py.File(os.path.abspath(inputFile), 'r')
			arr = file
			for key in self.h5keyList:
				arr = arr[key]
			arr = np.array(arr)
			self.center = np.mean(arr, axis=0)
			maxPos = np.max(arr, axis=0)
			minPos = np.min(arr, axis=0)
			self.width = np.max(maxPos - minPos)

		else:
			#for text files
			file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          

			#set up the variables
			#center = np.array([0.,0.,0.])
			maxPos = np.array([0., 0., 0.])
			minPos = np.array([0., 0., 0.])
			#begin the loop to read the file line-by-line
			lineN = 0
			center = np.array([0., 0., 0.])
			for line in file:
				lineN += 1
				if (lineN >= self.header):
					#get the x,y,z from the line 
					point = line.strip().split(self.delim)

					x = float(point[self.xCol])
					y = float(point[self.yCol])
					z = float(point[self.zCol])
					center += np.array([x,y,z])

					maxPos[0] = max([maxPos[0],x])
					maxPos[1] = max([maxPos[1],y])
					maxPos[2] = max([maxPos[2],z])

					minPos[0] = min([minPos[0],x])
					minPos[1] = min([minPos[1],y])
					minPos[2] = min([minPos[2],z])

				if (self.verbose > 0 and (lineN % 10000 == 0)):
					print('line : ', lineN)

				if (lineN > (self.Nmax - self.header - 1)):
					break


			self.center = center/(lineN - self.header)
			#self.center = (maxPos + minPos)/2.
			self.width = np.max(maxPos - minPos)

		file.close()

		if (self.verbose > 0):
			print('have initial center and size', self.center, self.width)

	def initialize(self):

		self.count = 0

		#create the output directory if needed
		if (not os.path.exists(self.path)):
			os.makedirs(self.path)
			
		#initialize the node variables
		self.nodes = [self.createNode(self.center, '0', width=self.width)] #will contain a list of all nodes with each as a dict
		self.baseNodePositions = np.array([self.center]) #will only contain the baseNodes locations as a list of lists (x,y,z)
		self.baseNodeIndices = np.array([0], dtype='int') #will contain the index for each baseNode within the self.nodes array

	def addPointToOctree(self, point):
		#find the node that it belongs in 
		baseIndex = self.baseNodeIndices[self.findClosestNode(np.array([point]),  self.baseNodePositions)]
		if (self.verbose > 2):
			print('index, id, Nparticles',baseIndex, self.nodes[baseIndex]['id'], self.nodes[baseIndex]['Nparticles'])
			
		#add the particle to the node
		self.nodes[baseIndex]['particles'].append(point)
		self.nodes[baseIndex]['needsUpdate'] = True
		self.nodes[baseIndex]['Nparticles'] += 1

		#check if we need to split the node
		if (self.nodes[baseIndex]['Nparticles'] >= self.NNodeMax and self.nodes[baseIndex]['width'] >= self.minWidth*2):
			self.createChildNodes(baseIndex) 

		#if we are beyond the memory limit, then write the nodes to files and clear the particles from the nodes 
		#(also reset the count)
		if (self.count > self.NMemoryMax):
			self.dumpNodesToFiles()


	def compileOctree(self, inputFile=None, append=False):



		#initialize a few things
		if (not append):
			self.getSizeCenter()
			self.initialize()

		if (inputFile is None):
			inputFile = self.inputFile

		#open the input file
		if (self.delim is None):
			#assume this is a hdf5 file
			file = h5py.File(os.path.abspath(inputFile), 'r')
			arr = file
			for key in self.h5keyList:
				arr = arr[key]
		else:
			#for text files
			file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          
			arr = file

		#begin the loop to read the file line-by-line
		lineN = 0
		for line in arr:
			lineN += 1
			if (lineN >= self.header):
				self.count += 1

				#get the x,y,z from the line 
				#get the x,y,z from the line 
				if (self.delim is None):
					point = line
				else:
					point = line.strip().split(self.delim)

				x = float(point[self.xCol])
				y = float(point[self.yCol])
				z = float(point[self.zCol])
				point = [x,y,z]
				
				self.addPointToOctree(point)
				
				if (self.verbose > 0 and (lineN % 10000 == 0)):
					print('line : ', lineN)

				if (lineN > (self.Nmax - self.header - 1)):
					break


		file.close()

		self.dumpNodesToFiles()
		self.shuffleAllParticlesInFiles()

		print('done')
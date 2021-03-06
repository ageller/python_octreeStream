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

#https://stackoverflow.com/questions/56250514/how-to-tackle-with-error-object-of-type-int32-is-not-json-serializable
#to help with dumping to json
class npEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.int32):
			return int(obj)
		if isinstance(obj, np.float32):
			return float(obj)
		return json.JSONEncoder.default(self, obj)

class octreeStream:
	def __init__(self, inputFile, NMemoryMax = 1e5, NNodeMax = 5000, 
				 header = 0, delim = None, colIndices = {'Coordinates':[0,1,2]},
				 baseDir = 'octreeNodes', Nmax=np.inf, verbose=0, path = None, minWidth=0, 
				 h5PartKey = '', keyList = ['Coordinates'], logKey = [False], center = None, cleanDir = False):
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
			logList : a list of true/false values indicating if a key from keyListshould be treated as the log (not Coordinates or Velocities)
			center : options for the user to provide the octree center (can save time)
			cleanDir : if true this will erase the files within that directory before beginning
		'''
		
		self.inputFile = inputFile
		self.NMemoryMax = NMemoryMax
		self.NNodeMax = NNodeMax
		self.header = header
		self.delim = delim
		self.colIndices = colIndices
		self.minWidth = minWidth
		self.h5PartKey = h5PartKey
		self.keyList = keyList
		self.logKey = logKey
		self.center = center
		self.cleanDir = cleanDir

		self.nodes = None #will contain a list of all nodes with each as a dict

		if (path is None):
			self.path = os.path.join(os.getcwd(), baseDir)
		else:
			self.path = os.path.abspath(path) #to make this windows safe
		print('files will be output to:', self.path)

		self.count = 0
		self.Nmax = Nmax

		self.verbose = verbose

		self.width = None #will be determined in getSizeCenter

		
	def createNode(self, center, id='', width=0,):
		return dict(x=center[0], y=center[1], z=center[2], width=width,
					Nparticles=0, id=id, parentNodes=[], childNodes=[], particles=[], needsUpdate=True)
	
	def findClosestNodeIndexByDistance(self, point, positions):
		#there is probably a faster and more clever way to do this
		#print('checking dist', point.shape, positions.shape, point, positions)
		dist2 = np.sum((positions - point)**2, axis=1)
		return np.argmin(dist2)
	
	def findClosestNode(self, point, parentIndex=None):
		#I am going to traverse the octree to find the closest node
		if (parentIndex is None):
			parentIndex = 0
		parent = self.nodes[parentIndex]
		childIndices = parent['childNodes']

		while (childIndices != []):
			childPositions = []
			for i in childIndices:
				childPositions.append([self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']])
			index = childIndices[self.findClosestNodeIndexByDistance(point[0:3], np.array(childPositions))]
			parent = self.nodes[index]
			childIndices = parent['childNodes']

		return parent

	def createChildNodes(self, node):

		#split the node into 8 separate nodes
		if (self.verbose > 0):
			print('creating child nodes', node['id'], node['Nparticles'], node['width'])

		#check if we need to read in the file (should this be a more careful check?)
		if (len(node['particles']) < self.NNodeMax): 
			self.populateNodeFromFile(node)


		#create the new nodes 
		cx = node['x'] + node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] + node['width']/4.
		n1 = self.createNode([cx, cy, cz], node['id']+'1', width=node['width']/2.)

		cx = node['x'] - node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] + node['width']/4.
		n2 = self.createNode([cx, cy, cz], node['id']+'2',  width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] + node['width']/4.
		n3 = self.createNode([cx, cy, cz], node['id']+'3', width=node['width']/2.)
		
		cx = node['x'] - node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] + node['width']/4.
		n4 = self.createNode([cx, cy, cz], node['id']+'4', width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] - node['width']/4.
		n5 = self.createNode([cx, cy, cz], node['id']+'5', width=node['width']/2.)

		
		cx = node['x'] - node['width']/4.
		cy = node['y'] + node['width']/4.
		cz = node['z'] - node['width']/4.
		n6 = self.createNode([cx, cy, cz], node['id']+'6', width=node['width']/2.)
		
		cx = node['x'] + node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] - node['width']/4.
		n7 = self.createNode([cx, cy, cz], node['id']+'7', width=node['width']/2.)
		
		cx = node['x'] - node['width']/4.
		cy = node['y'] - node['width']/4.
		cz = node['z'] - node['width']/4.
		n8 = self.createNode([cx, cy, cz], node['id']+'8', width=node['width']/2.)
		
		childIndices = np.array([], dtype='int')
		for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
			#add the parent and child indices to the nodes
			n['parentNodes'] = node['parentNodes'] + [self.nodes.index(node)]
			childIndex = len(self.nodes)
			self.nodes.append(n)
			node['childNodes'].append(childIndex)

			#create these so that I can divide up the parent particles
			if (i == 0):
				childPositions = np.array([[n['x'],n['y'],n['z']]])
			else:
				childPositions = np.append(childPositions, [[n['x'],n['y'],n['z']]], axis=0)
			childIndices = np.append(childIndices, childIndex)

		#divide up the particles 
		for p in node['particles']:
			child = self.findClosestNode(np.array(p), parentIndex=self.nodes.index(node))
			child['particles'].append(p)
			child['Nparticles'] += 1      

		#check how many particles ended up in each child node
		# if (self.verbose > 0):
		# 	for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
		# 		print('   Child node, Nparticles', n['id'], n['Nparticles'])
		# 		self.checkNodeParticles(node=n)
		
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
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				fmt = ''
				header = ''
				for key,logCheck in zip(self.keyList, self.logKey):
					if (key == 'Coordinates'):
						fmt += '%.8e,%.8e,%.8e,'
						header +='x,y,z,'
					elif(key == 'Velocities'):
						fmt += '%.8e,%.8e,%.8e,'
						header +='vx,vy,vz,'
					else:
						fmt += '%.8e,'
						useKey = key
						if (logCheck):
							useKey = 'log10' + useKey
						header += useKey + ','
				fmt = fmt[:-1] #remove the last ','
				header = header[:-1] #remove the last ','
				mode = 'w'
				if (os.path.exists(nodeFile)):
					mode = 'a'
					header = ''
				with open(nodeFile, mode) as f:
					np.savetxt(f, parts, fmt=fmt, header=header, comments='')


				node['particles'] = []
				node['needsUpdate'] = False
				if (self.verbose > 1):
					print('writing node to file ', node['id'], mode)
				
		#node dict
		with open(os.path.join(self.path, 'octree.json'), 'w') as f:
			json.dump(self.nodes, f, cls=npEncoder)

		self.count = 0
				
	
	def checkNodeParticles(self, node=None, iden=None):
		if (iden is not None):
			for n in self.nodes:
				if (n['id'] == iden):
					node = n
					break

		if (node is not None):
			center = [node['x'], node['y'], node['z']]
			print('      checking node...')
			print('      width = ', node['width'])
			print('      Nparticles = ', node['Nparticles'])
			print('      center = ',center)
			if (node['Nparticles'] > 0):
				if (node['particles'] == [] and node['Nparticles'] > 0):
					self.populateNodeFromFile(node)

				#get the mean position of the particles and the max width
				parts = np.array(node['particles'])[:,0:3]

				meanPosition = np.mean(parts, axis=0)
				maxPosition = np.max(parts, axis=0)
				minPosition = np.min(parts, axis=0)
				dist2 = np.sum((parts - np.array([center]))**2, axis=1)
				width = np.max(np.sqrt(dist2))
				hi = maxPosition - np.array(center)
				lo = np.array(center) - minPosition
				width_linear = np.max(hi - lo)

				print('      mean particle position = ', meanPosition)
				print('      max particle position = ', maxPosition)
				print('      min particle position = ', minPosition)
				print('      max distance for particle positions = ', maxPosition - minPosition)
				print('      width of particles', width, width_linear)

				if (width > node['width']):
					print('      !!!! WARNING, particles are outside width of node')
					wAll = np.sqrt(dist2)
					outside = np.where(wAll > node['width'])[0]
					outside_pick = outside[0]
					#check if there is a closer node... and if not, why not!!??
					for i,n in enumerate(self.nodes):
						if (i == 0):
							allPositions = np.array([[n['x'], n['y'], n['z']]])
						else:
							allPositions = np.append(allPositions, [[n['x'], n['y'], n['z']]], axis=0)
					p = np.array([parts[outside_pick]])[:,0:3]
					dist2 = np.sum((allPositions - p)**2., axis=1)
					print('      checking this particle',p)
					print('      min distance to all, base nodes',min(dist2)**0.5)
		else:
			print('Please specify a node or node id on input')

	def checkNodeFiles(self):
		#check to make sure that only the nodes with Nparticles > 0 have files
		Nerror = 0
		if (self.nodes is None):
			print('Please compile the octree first')
			return

		#first get names of all expected files
		names = []
		Nparts = []
		for n in self.nodes:
			if (n['Nparticles'] > 0):
				names.append(n['id'] + '.csv')
				Nparts.append(n['Nparticles'])

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

		#read in the octree from the json file
		if (read):
			with open(os.path.join(self.path,'octree.json')) as f:
				self.nodes = json.load(f)
				

		NbaseNodes = 0
		for node in self.nodes:
			if (node['Nparticles'] > 0):
				Nparts += node['Nparticles']
				NbaseNodes += 1
				self.populateNodeFromFile(node)
		print('Populated octree from files.')
		print(' -- total number of particles = ', Nparts)
		print(' -- total number of nodes = ', len(self.nodes))
		print(' -- total number of base nodes = ', NbaseNodes)
		

	def populateNodeFromFile(self, node):
		nodeFile = os.path.join(self.path, node['id'] + '.csv')
		if (self.verbose > 1):
			print('reading in file', nodeFile)
		parts = np.genfromtxt(nodeFile, delimiter=',', skip_header=1).tolist()
		node['particles'] += parts
		node['Nparticles'] = len(node['particles'])
		node['needsUpdate'] = True
		self.count += node['Nparticles']

	def shuffleAllParticlesInFiles(self):

		if (self.verbose > 0):
			print('randomizing particle order in data files ... ')

		#read in the octree from the json file
		with open(os.path.join(self.path,'octree.json')) as f:
			self.nodes = json.load(f)
				
		for node in self.nodes:
			if (node['Nparticles'] > 0):
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				if (self.verbose > 1):
					print(nodeFile)
				lines = open(nodeFile).readlines()
				header = lines[0]
				parts = lines[1:]
				random.shuffle(parts)
				lines = [header] + parts
				open(nodeFile, 'w').writelines(lines)

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
			if (self.h5PartKey != ''):
				arr = arr[self.h5PartKey]
			arr = np.array(arr[self.keyList[0]]) #Coordinates are always first
			if (self.center is None):
				self.center = np.mean(arr, axis=0)
			maxPos = np.max(arr - self.center, axis=0)
			minPos = np.min(arr - self.center, axis=0)
			self.width = 2.*np.max(np.abs(np.append(maxPos,minPos)))

		else:
			#for text files
			file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          
			self.iterFileCenter(file)

		file.close()
		if (self.verbose > 0):
			print('have initial center and size', self.center, self.width)

	def iterFileCenter(self, file):
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

				coordIndices = self.colIndices['Coordinates']
				x = float(point[coordIndices[0]])
				y = float(point[coordIndices[1]])
				z = float(point[coordIndices[2]])
				center += np.array([x,y,z])

				maxPos[0] = max([maxPos[0],x])
				maxPos[1] = max([maxPos[1],y])
				maxPos[2] = max([maxPos[2],z])

				minPos[0] = min([minPos[0],x])
				minPos[1] = min([minPos[1],y])
				minPos[2] = min([minPos[2],z])

			if (self.verbose > 0 and (lineN % 100000 == 0)):
				print('line : ', lineN)

			if (lineN > (self.Nmax - self.header - 1)):
				break


		if (self.center is None):
			self.center = center/(lineN - self.header)
		#self.center = (maxPos + minPos)/2.
		maxPos -= self.center
		minPos -= self.center
		self.width = 2.*np.max(np.abs(np.append(maxPos, minPos)))


	def initialize(self):

		self.count = 0

		#create the output directory if needed
		if (not os.path.exists(self.path)):
			os.makedirs(self.path)
			
		#remove the files in that directory
		if (self.cleanDir):
			for f in os.listdir(self.path):
				os.remove(os.path.join(self.path, f))

		#initialize the node variables
		self.nodes = [self.createNode(self.center, '0', width=self.width)] #will contain a list of all nodes with each as a dict


	def addPointToOctree(self, point):
		#find the node that it belongs in 
		node = self.findClosestNode(np.array(point))
		if (self.verbose > 2):
			print('id, Nparticles', node['id'], node['Nparticles'])
			
		#add the particle to the node
		node['particles'].append(point)
		node['needsUpdate'] = True
		node['Nparticles'] += 1

		#check if we need to split the node
		if (node['Nparticles'] >= self.NNodeMax and node['width'] >= self.minWidth*2):
			self.createChildNodes(node) 

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
			if (self.h5PartKey != ''):
				arrPart = arr[self.h5PartKey]

			#now build the particle array
			for i, (key, logCheck) in enumerate(zip(self.keyList, self.logKey)):
				if (i == 0):
					arr = np.array(arrPart[key]) #Coordinates are always first
				else:
					if (logCheck):
						addOn = np.log10(np.array(arrPart[key]))
					else:
						addOn = np.array(arrPart[key])
					arrLen = 1
					if (key == 'Velocities'): #requires special handling because it is a 2D array
						arrLen = 3
					arr = np.hstack((arr, np.reshape(addOn, (len(arr),arrLen))))

		else:
			#for text files
			file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          
			arr = file

		self.iterFileOctree(arr)

		file.close()

		self.dumpNodesToFiles()
		self.shuffleAllParticlesInFiles()

		print('done')

	def iterFileOctree(self, arr):
		#begin the loop to read the file line-by-line
		lineN = 0
		for i in range(arr.shape[0]):
			line = arr[i]
		#for line in arr:
			lineN += 1
			if (lineN >= self.header):
				self.count += 1

				#get the x,y,z from the line 
				if (self.delim is None):
					point = line
				else:
					lineStrip = line.strip().split(self.delim)
					point = []
					for key in self.keyList:
						indices =  self.colIndices[key]
						if (type(indices) is not list):
							indices = [indices]
						for ii in indices:
							point.append(float(lineStrip[ii]))

				self.addPointToOctree(point)
				
				if (self.verbose > 0 and (lineN % 100000 == 0)):
					print('line : ', lineN)

				if (lineN > (self.Nmax - self.header - 1)):
					break




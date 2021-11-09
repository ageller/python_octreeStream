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
		parent = self.nodes[parentIndex]
		childIndices = parent['childNodes']

		while (len(childIndices) > 0):
			childPositions = []
			for i in childIndices:
				childPositions.append([self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']])
			parentIndex = childIndices[self.findClosestNodeIndexByDistance(point[0:3], np.array(childPositions))]
			parent = self.nodes[parentIndex]
			childIndices = parent['childNodes']

		return (parent, parentIndex)

	def createChildNodes(self, parentIndex):

		#split the node into 8 separate nodes
		if (self.managerDict['verbose'] > 0):
			print('creating child nodes', self.nodes[parentIndex]['id'], self.nodes[parentIndex]['Nparticles'], self.nodes[parentIndex]['width'])

		#check if we need to read in the file (should this be a more careful check?)
		if (len(self.nodes[parentIndex]['particles']) < self.managerDict['NNodeMax']): 
			self.populateNodeFromFile(parentIndex)


		#create the new nodes 
		#check to make sure node doesn't already exist, since I'm running in parallel?
		n1, index1 = self.getNodeByID(self.nodes[parentIndex]['id']+'1')
		if (n1 is None):
			cx = self.nodes[parentIndex]['x'] + self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] + self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] + self.nodes[parentIndex]['width']/4.
			n1, index1 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'1', width=self.nodes[parentIndex]['width']/2.)

		n2, index2 = self.getNodeByID(self.nodes[parentIndex]['id']+'2')
		if (n2 is None):
			cx = self.nodes[parentIndex]['x'] - self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] + self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] + self.nodes[parentIndex]['width']/4.
			n2, index2 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'2',  width=self.nodes[parentIndex]['width']/2.)
		
		n3, index3 = self.getNodeByID(self.nodes[parentIndex]['id']+'3')
		if (n3 is None):
			cx = self.nodes[parentIndex]['x'] + self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] - self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] + self.nodes[parentIndex]['width']/4.
			n3, index3 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'3', width=self.nodes[parentIndex]['width']/2.)
		
		n4, index4 = self.getNodeByID(self.nodes[parentIndex]['id']+'4')
		if (n4 is None):
			cx = self.nodes[parentIndex]['x'] - self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] - self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] + self.nodes[parentIndex]['width']/4.
			n4, index4 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'4', width=self.nodes[parentIndex]['width']/2.)
		
		n5, index5 = self.getNodeByID(self.nodes[parentIndex]['id']+'5')
		if (n5 is None):
			cx = self.nodes[parentIndex]['x'] + self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] + self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] - self.nodes[parentIndex]['width']/4.
			n5, index5 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'5', width=self.nodes[parentIndex]['width']/2.)

		n6, index6 = self.getNodeByID(self.nodes[parentIndex]['id']+'6')
		if (n6 is None):
			cx = self.nodes[parentIndex]['x'] - self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] + self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] - self.nodes[parentIndex]['width']/4.
			n6, index6 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'6', width=self.nodes[parentIndex]['width']/2.)
		
		n7, index7 = self.getNodeByID(self.nodes[parentIndex]['id']+'7')
		if (n7 is None):
			cx = self.nodes[parentIndex]['x'] + self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] - self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] - self.nodes[parentIndex]['width']/4.
			n7, index7 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'7', width=self.nodes[parentIndex]['width']/2.)
		
		n8, index8 = self.getNodeByID(self.nodes[parentIndex]['id']+'8')
		if (n8 is None):
			cx = self.nodes[parentIndex]['x'] - self.nodes[parentIndex]['width']/4.
			cy = self.nodes[parentIndex]['y'] - self.nodes[parentIndex]['width']/4.
			cz = self.nodes[parentIndex]['z'] - self.nodes[parentIndex]['width']/4.
			n8, index8 = self.createNode([cx, cy, cz], self.nodes[parentIndex]['id']+'8', width=self.nodes[parentIndex]['width']/2.)
		
		childIndices = np.array([], dtype='int')
		#for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
		for i, n in enumerate([index1, index2, index3, index4, index5, index6, index7, index8]):
			#add the parent and child indices to the nodes
			self.nodes[n]['parentNodes'] = self.nodes[parentIndex]['parentNodes'] + [parentIndex]
			childIndex = len(self.nodes)
			self.nodes[parentIndex]['childNodes'] += [childIndex]

			#create these so that I can divide up the parent particles
			if (i == 0):
				childPositions = np.array([[self.nodes[n]['x'],self.nodes[n]['y'],self.nodes[n]['z']]])
			else:
				childPositions = np.append(childPositions, [[self.nodes[n]['x'],self.nodes[n]['y'],self.nodes[n]['z']]], axis=0)
			childIndices = np.append(childIndices, childIndex)

		#divide up the particles 
		for p in self.nodes[parentIndex]['particles']:
			child, index = self.findClosestNode(np.array(p), parentIndex=parentIndex)
			self.nodes[index]['particles'] += [p]
			self.nodes[index]['Nparticles'] += 1      

		#check how many particles ended up in each child node
		# if (self.managerDict['verbose > 0):
		# 	for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
		# 		print('   Child node, Nparticles', n['id'], n['Nparticles'])
		# 		self.managerDict['checkNodeParticles(node=n)
		
		#remove the particles from the parent
		self.nodes[parentIndex]['particles'] = []
		self.nodes[parentIndex]['Nparticles'] = 0
		
		#check if we need to remove a file
		nodeFile = os.path.join(self.managerDict['path'], self.nodes[parentIndex]['id'] + '.csv')
		if (os.path.exists(nodeFile)):
			os.remove(nodeFile)
			if (self.managerDict['verbose'] > 0):
				print('removing file', nodeFile)
			
	def dumpNodesToFiles(self):
		#dump all the nodes to files
		if (self.managerDict['verbose'] > 0):
			print('dumping nodes to files ...')
		
		#individual nodes
		for node in self.nodes:
			print('checking node for file', node['Nparticles'], node['needsUpdate'], len(node['particles']))
			if ( (node['Nparticles'] > 0) and ('particles' in node) and (node['needsUpdate'])):

				parts = np.array(node['particles'])
				nodeFile = os.path.join(self.managerDict['path'], node['id'] + '.csv')
				fmt = ''
				header = ''
				for key in self.managerDict['keyList']:
					if (key == 'Coordinates'):
						fmt += '%.8e,%.8e,%.8e,'
						header +='x,y,z,'
					elif(key == 'Velocities'):
						fmt += '%.8e,%.8e,%.8e,'
						header +='vx,vy,vz,'
					else:
						fmt += '%.8e,'
						header += key + ','
				fmt = fmt[:-1] #remove the last ','
				header = header[:-1] #remove the last ','
				mode = 'w'
				if (os.path.exists(nodeFile)):
					mode = 'a'
					header = ''
				print('WRITING FILE', nodeFile)
				with open(nodeFile, mode) as f:
					np.savetxt(nodeFile, parts, fmt=fmt, header=header, comments='')


				node['particles'] = []
				node['needsUpdate'] = False
				if (self.managerDict['verbose'] > 1):
					print('writing node to file ', node['id'], mode)
				
		#node dict
		with open(os.path.join(self.managerDict['path'], 'octree.json'), 'w') as f:
			json.dump(list(self.nodes), f, cls=npEncoder)

		self.managerDict['count'] = 0
				
	
	def checkNodeParticles(self, node=None, iden=None, index=None):
		if (index is not None):
			node = self.nodes[index]

		if (index is None and iden is not None):
			node, index = self.getNodeByID(iden)

		if (index is not None):
			center = [self.nodes[index]['x'], self.nodes[index]['y'], self.nodes[index]['z']]
			print('      checking node...')
			print('      width = ', self.nodes[index]['width'])
			print('      Nparticles = ', self.nodes[index]['Nparticles'])
			print('      center = ',center)
			if (self.nodes[index]['Nparticles'] > 0):
				if (self.nodes[index]['particles'] == [] and self.nodes[index]['Nparticles'] > 0):
					self.populateNodeFromFile(index)

				#get the mean position of the particles and the max width
				parts = np.array(self.nodes[index]['particles'])[:,0:3]

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

				if (width > self.nodes[index]['width']):
					print('      !!!! WARNING, particles are outside width of node')
					wAll = np.sqrt(dist2)
					outside = np.where(wAll > self.nodes[index]['width'])[0]
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

	def getNodeByID(self, iden):
		node = None
		index = 0
		for i,n in enumerate(self.nodes):
			if (n['ID'] == iden):
				node = n
				index = i
				break
		return node, index

	def checkNodeFiles(self):
		#check to make sure that only the nodes with Nparticles > 0 have files
		Nerror = 0
		if (len(self.nodes) == 0):
			print('Please compile the octree first')
			return

		#first get names of all expected files
		names = []
		Nparts = []
		for n in self.nodes:
			if (n['Nparticles'] > 0):
				names.append(n['id'] + '.csv')
				Nparts.append(n['Nparticles'])

		avail = os.listdir(self.managerDict['path'])

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
			with open(os.path.join(self.managerDict['path'],'octree.json')) as f:
				self.nodes = json.load(f)
				

		NbaseNodes = 0
		for index, node in enumerate(self.nodes):
			if (node['Nparticles'] > 0):
				Nparts += node['Nparticles']
				NbaseNodes += 1
				self.populateNodeFromFile(index)
		print('Populated octree from files.')
		print(' -- total number of particles = ', Nparts)
		print(' -- total number of nodes = ', len(self.nodes))
		print(' -- total number of base nodes = ', NbaseNodes)
		

	def populateNodeFromFile(self, index):
		nodeFile = os.path.join(self.managerDict['path'], self.nodes[index]['id'] + '.csv')
		if (self.managerDict['verbose'] > 1):
			print('reading in file', nodeFile)
		parts = np.genfromtxt(nodeFile, delimiter=',', skip_header=1).tolist()
		self.nodes[index]['particles'] += parts
		self.nodes[index]['Nparticles'] = len(self.nodes[index]['particles'])
		self.nodes[index]['needsUpdate'] = True
		self.managerDict['count'] += self.nodes[index]['Nparticles']

	def shuffleAllParticlesInFiles(self):

		if (self.managerDict['verbose'] > 0):
			print('randomizing particle order in data files ... ')

		#read in the octree from the json file
		with open(os.path.join(self.managerDict['path'],'octree.json')) as f:
			self.nodes = json.load(f)
				
		for node in self.nodes:
			if (node['Nparticles'] > 0):
				nodeFile = os.path.join(self.managerDict['path'], node['id'] + '.csv')
				if (self.managerDict['verbose'] > 1):
					print(nodeFile)
				lines = open(nodeFile).readlines()
				header = lines[0]
				parts = lines[1:]
				random.shuffle(parts)
				lines = [header] + parts
				open(nodeFile, 'w').writelines(lines)

	def getSizeCenter(self, inputFile=None):
		#It will be easiest if we can get the center and the size at the start.  This will create overhead to read in the entire file...

		if (self.managerDict['verbose'] > 0):
			print('calculating center and size ... ')

		if (inputFile is None):
			inputFile = self.managerDict['inputFile']
			
		#open the input file
		if (self.managerDict['delim'] is None):
			#assume this is a hdf5 file
			file = h5py.File(os.path.abspath(inputFile), 'r')
			arr = file
			if (self.managerDict['h5PartKey'] != ''):
				arr = arr[self.managerDict['h5PartKey']]
			arr = np.array(arr[self.managerDict['keyList'][0]]) #Coordinates are always first
			if (self.managerDict['center'] is None):
				self.managerDict['center'] = np.mean(arr, axis=0)
			maxPos = np.max(arr - self.managerDict['center'], axis=0)
			minPos = np.min(arr - self.managerDict['center'], axis=0)
			self.managerDict['width'] = 2.*np.max(np.abs(np.append(maxPos,minPos)))

		else:
			#for text files
			file = open(os.path.abspath(inputFile), 'r') #abspath converts to windows format          
			self.iterFileCenter(file)

		file.close()
		if (self.managerDict['verbose'] > 0):
			print('have initial center and size', self.managerDict['center'], self.managerDict['width'])

	def iterFileCenter(self, file):
		#set up the variables
		#center = np.array([0.,0.,0.])
		maxPos = np.array([0., 0., 0.])
		minPos = np.array([0., 0., 0.])
		#begin the loop to read the file line-by-line
		self.managerDict['lineN'] = 0
		center = np.array([0., 0., 0.])

		for line in file:
			self.managerDict['lineN'] += 1
			if (self.managerDict['lineN'] >= self.managerDict['header']):
				#get the x,y,z from the line 
				point = line.strip().split(self.managerDict['delim'])

				coordIndices = self.managerDict['colIndices']['Coordinates']
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

			if (self.managerDict['verbose'] > 0 and (self.managerDict['lineN'] % 100000 == 0)):
				print('line : ', self.managerDict['lineN'])

			if (self.managerDict['lineN'] > (self.managerDict['Nmax'] - self.managerDict['header'] - 1)):
				break


		if (self.managerDict['center'] is None):
			self.managerDict['center'] = center/(self.managerDict['lineN'] - self.managerDict['header'])
		#self.managerDict['center = (maxPos + minPos)/2.
		maxPos -= self.managerDict['center']
		minPos -= self.managerDict['center']
		self.managerDict['width'] = 2.*np.max(np.abs(np.append(maxPos, minPos)))


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
		print('BEFORE',self.nodes[index]['Nparticles'])
		self.nodes[index]['Nparticles'] += 1
		print('AFTER',self.nodes[index]['Nparticles'])

	def compileOctree(self, inputFile=None, append=False):

		#initialize a few things
		if (not append):
			#self.getSizeCenter()
			self.managerDict['center'] = [0,0,0]
			self.managerDict['width'] = 1000
			_ = self.initialize()



		if (inputFile is None):
			inputFile = self.managerDict['inputFile']

		#open the input file
		if (self.managerDict['delim'] is None):
			#assume this is a hdf5 file
			file = h5py.File(os.path.abspath(inputFile), 'r')
			arr = file
			if (self.managerDict['h5PartKey'] != ''):
				arrPart = arr[self.managerDict['h5PartKey']]

			#now build the particle array
			for i, key in enumerate(self.managerDict['keyList']):
				if (i == 0):
					arr = np.array(arrPart[key]) #Coordinates are always first
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


		self.managerDict['Nmax'] = min(self.managerDict['Nmax'], arr.shape[0])

		self.iterFileOctree(arr)

		# file.close()

		# self.shuffleAllParticlesInFiles()

		# print('done')

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
				#j = Process(target=self.test, args=(0,))
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

			#for testing
			self.managerDict['lineN'] = iEnd
			#now dump to files
			self.dumpNodesToFiles()

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
			
			if (self.managerDict['verbose'] > 0 and (self.managerDict['lineN'] % 1000 == 0)):
				print('line : ', self.managerDict['lineN'])




if __name__ == '__main__':
	oM1 = octreeStream('/Users/ageller/VISUALIZATIONS/FIREdata/m12i_res7100/snapdir_600/snapshot_600.0.hdf5', 
	                 h5PartKey = 'PartType0', keyList = ['Coordinates', 'Density', 'Velocities'],
	                 NNodeMax = 10000, NMemoryMax = 5e4, Nmax=1e5, verbose=2, minWidth=1e-4,
	                 cleanDir = True,
	                 Ncores=4,
	                 path='/Users/ageller/VISUALIZATIONS/octree_threejs_python/WebGL_octreePartition/src/data/junk/octreeNodes/Gas')

	oM1.compileOctree()

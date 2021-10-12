''' Attempting to write an octree in python

I want this to work with VERY large data sets that can't be stored fully in memory.  So my procedure will be as follows:
- need to read in line-by-line and clear memory every X MB (or maybe every X particles;can I check memory load in python?)
- go down to nodes with containing N particles
- need to write out tree with node sizes and centers and also ending nodes with actual particles
'''

import os
import numpy as np
import pandas as pd
import json

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
				 header = 1, delim = ',', xCol = 0, yCol = 1, zCol = 2,
				 center = [0., 0., 0.], baseDir = 'octreeNodes', lineNmax=np.inf):
		'''
			inputFile : path to the file. For now only text files.
			NMemoryMax : the maximum number of particles to save in the memory before writing to a file
			NNodeMax : the maximum number of particles to store in a node before splitting it
			header : the line number of the header (file starts at line 1, 
				set header=0 for no header, and in that case x,y,z are assumed to be the first three columns)
			delim : the delimiter between columns
			xCol, yCol, zCol : the columns that have the x,y,z data
			center : the center of the particles [x,y,z]
			baseDir : the directory to store the octree files
			lineNmax : maximum number of lines to read in
		'''
		
		self.inputFile = inputFile
		self.NMemoryMax = NMemoryMax
		self.NNodeMax = NNodeMax
		self.header = header
		self.delim = delim
		self.xCol = xCol
		self.yCol = yCol
		self.zCol = zCol
		self.center = center
		self.baseDir = baseDir
		
		self.nodes = None #will contain a list of all nodes with each as a dict
		self.baseNodePositions = None #will only contain the baseNodes locations as a list of lists (x,y,z)
		self.baseNodeIndices = None #will contain the index for each baseNode within the self.nodes array
		self.path = os.path.join(os.getcwd(),self.baseDir)

		self.count = 0
		self.lineNmax = lineNmax

		self.verbose = 0
		
	def createNode(self, center, id='', xWidth=0, yWidth=0, zWidth=0):
		return dict(x=center[0], y=center[1], z=center[2], xWidth=xWidth, yWidth=yWidth, zWidth=zWidth,
					Nparticles=0, id=id, parentNodes=[], childNodes=[], particles=[])
	
	def findClosestNode(self, point, positions):
		#there is probably a faster and more clever way to do this
		dist2 = np.sum((positions - point)**2, axis=1)
		return np.argmin(dist2)
	
	def createChildNodes(self, index):
		#split the node into 8 separate nodes and add these to self.nodes and self.baseNodePositions
		if (self.verbose > 0):
			print('creating child nodes', index)

		#update the parent widths
		def updateWidth(node, key, arr):
			node[key+'Width'] = max([max(arr) - node[key], node[key] - min(arr)])
			
		def updateParentWidth(node, parent, key):
			if ((parent[key] + parent[key+'Width']) < (node[key] + node[key+'Width'])):
				 parent[key+'Width'] = (node[key] + node[key+'Width']) - parent[key]
			if ((parent[key] - parent[key+'Width']) > (node[key] - node[key+'Width'])):
				 parent[key+'Width'] = parent[key] - (node[key] - node[key+'Width'])  
					
		if ('particles' not in self.nodes[index]):
			self.populateNodeFromFile(self.nodes[index])
		parts = np.array(self.nodes[index]['particles'])
		x = parts[:,0]
		y = parts[:,1]
		z = parts[:,2]
		updateWidth(self.nodes[index], 'x', x)
		updateWidth(self.nodes[index], 'y', y)
		updateWidth(self.nodes[index], 'z', z)
		
		#and propagate this upwards
		for p in self.nodes[index]['parentNodes']:
			updateParentWidth(self.nodes[index], self.nodes[p], 'x')
			updateParentWidth(self.nodes[index], self.nodes[p], 'y')
			updateParentWidth(self.nodes[index], self.nodes[p], 'z')
				

		#create the new nodes and add to the baseNodePositions array
		cx = self.nodes[index]['x'] + self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] + self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] + self.nodes[index]['zWidth']/2.
		n1 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_1', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)
		
		cx = self.nodes[index]['x'] - self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] + self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] + self.nodes[index]['zWidth']/2.
		n2 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_2', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)
		
		cx = self.nodes[index]['x'] + self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] - self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] + self.nodes[index]['zWidth']/2.
		n3 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_3', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)           
		
		cx = self.nodes[index]['x'] - self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] - self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] + self.nodes[index]['zWidth']/2.
		n4 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_4', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)   
		
		cx = self.nodes[index]['x'] + self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] + self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] - self.nodes[index]['zWidth']/2.
		n5 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_5', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)
		
		cx = self.nodes[index]['x'] - self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] + self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] - self.nodes[index]['zWidth']/2.
		n6 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_6', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.) 
		
		cx = self.nodes[index]['x'] + self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] - self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] - self.nodes[index]['zWidth']/2.
		n7 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_7', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)          
		
		cx = self.nodes[index]['x'] - self.nodes[index]['xWidth']/2.
		cy = self.nodes[index]['y'] - self.nodes[index]['yWidth']/2.
		cz = self.nodes[index]['z'] - self.nodes[index]['zWidth']/2.
		n8 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_8', 
							 xWidth=self.nodes[index]['xWidth']/2.,
							 yWidth=self.nodes[index]['yWidth']/2.,
							 zWidth=self.nodes[index]['zWidth']/2.)   
		
		#childPositions = np.array([[]])
		childIndices = np.array([], dtype='int')
		for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
			n['parentNodes'] = self.nodes[index]['parentNodes'] + [index]
			self.nodes.append(n)
			self.nodes[index]['childNodes'].append(len(self.nodes) - 1)
			self.baseNodePositions = np.append(self.baseNodePositions, [[n['x'],n['y'],n['z']]], axis=0)
			self.baseNodeIndices = np.append(self.baseNodeIndices, len(self.nodes) - 1)
			if (i == 0):
				childPositions = [[n['x'],n['y'],n['z']]]
			else:
				childPositions = np.append(childPositions, [[n['x'],n['y'],n['z']]], axis=0)
			childIndices = np.append(childIndices, len(self.nodes) - 1)
			if (self.verbose > 1):
				print('new baseNode position',[[n['x'],n['y'],n['z']]])
			
		#divide up the particles (I suppose I could do this just for the children, but I already have this function...)
		for p in self.nodes[index]['particles']:
			childIndex = childIndices[self.findClosestNode(np.array([p]), childPositions)]
			self.nodes[childIndex]['particles'].append(p)
			self.nodes[childIndex]['Nparticles'] += 1            
			
		#remove the parent center from the self.baseNodePositions
		i = np.where(self.baseNodeIndices == index)[0]
		if (len(i) > 0 and i != -1):
			self.baseNodePositions = np.delete(self.baseNodePositions, i, axis=0)
			self.baseNodeIndices = np.delete(self.baseNodeIndices,i)
		
		#remove the particles from the parent
		self.nodes[index]['particles'] = []
		self.nodes[index]['Nparticles'] = 0
		
		#check if we need to remove a file
		nodeFile = os.path.join(self.path, self.nodes[index]['id'] + '.csv')
		if (os.path.exists(nodeFile)):
			os.remove(nodeFile)
			if (self.verbose > 0):
				print('removing file', nodeFile)
			
	def dumpNodesToFiles(self):
		#dump all the nodes to files
		if (self.verbose > 0):
			print('dumping nodes to files')
		
		#individual nodes
		for node in self.nodes:
			if ( (node['Nparticles'] > 0) and ('particles' in node)):
				parts = np.array(node['particles'])
				x = parts[:,0]
				y = parts[:,1]
				z = parts[:,2]
				df = pd.DataFrame(dict(x=x, y=y, z=z)).sample(frac=1).reset_index(drop=True) #shuffle the order
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				df.to_csv(nodeFile, index=False)
				node['particles'] = []
				del node['particles']
				
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
		for index in self.baseNodeIndices:
			names.append(self.nodes[index]['id'] + '.csv')

		#now check the list
		for fname in os.listdir(self.path):
			if ( (fname not in names) and (fname != 'octree.json')):
				print('!!!WARNING: this file should not exist', fname)
				Nerror += 1
		print('Number of bad files = ', Nerror)
		
			
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
		if (self.verbose > 0):
			print('reading in file', nodeFile)
		df = pd.read_csv(nodeFile)
		node['particles'] = df.values.tolist()
		node['Nparticles'] = len(node['particles'])
		self.count += node['Nparticles']
		
	def compileOctree(self):

		#create the output directory if needed
		if (not os.path.exists(self.path)):
			os.mkdir(self.path)
			
		#initialize the node variables
		self.nodes = [self.createNode(self.center, '0')] #will contain a list of all nodes with each as a dict
		self.baseNodePositions = np.array([self.center]) #will only contain the baseNodes locations as a list of lists (x,y,z)
		self.baseNodeIndices = np.array([0], dtype='int') #will contain the index for each baseNode within the self.nodes array

		#open the input file
		file = open(os.path.abspath(self.inputFile), 'r') #abspath converts to windows format          

		#begin the loop to read the file line-by-line
		self.count = 0
		lineN = 0
		for line in file:
			self.count += 1
			lineN += 1
			if (self.count >= self.header):
				#get the x,y,z from the line 
				split = line.strip().split(self.delim)
				x = float(split[self.xCol])
				y = float(split[self.yCol])
				z = float(split[self.zCol])
				point = [x,y,z]
				
				#find the node that it belongs in 
				baseIndex = self.baseNodeIndices[self.findClosestNode(np.array([point]),  self.baseNodePositions)]
				if (self.verbose > 1):
					print('lineN, index, id, Nparticles',lineN, baseIndex, self.nodes[baseIndex]['id'], 
					  self.nodes[baseIndex]['Nparticles'])

				#check if there are particles in the node in memory; if not, read in the file
				if ('particles' not in self.nodes[baseIndex]):
					#read in the file (also add to the count)
					self.populateNodeFromFile(self.nodes[baseIndex])
					
				#add the particle to the node
				self.nodes[baseIndex]['particles'].append(point)
				self.nodes[baseIndex]['Nparticles'] += 1

				#check if we need to split the node
				if (self.nodes[baseIndex]['Nparticles'] >= self.NNodeMax):
					self.createChildNodes(baseIndex) 

				#if we are beyond the memory limit, then write the nodes to files and clear the particles from the nodes 
				#(also reset the count)
				if (self.count > self.NMemoryMax):
					self.dumpNodesToFiles()
				
				if (lineN > self.lineNmax):
					self.dumpNodesToFiles()
					break


		file.close()
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
				 baseDir = 'octreeNodes', lineNmax=np.inf, verbose=0, path = None):
		'''
			inputFile : path to the file. For now only text files.
			NMemoryMax : the maximum number of particles to save in the memory before writing to a file
			NNodeMax : the maximum number of particles to store in a node before splitting it
			header : the line number of the header (file starts at line 1, 
				set header=0 for no header, and in that case x,y,z are assumed to be the first three columns)
			delim : the delimiter between columns
			xCol, yCol, zCol : the columns that have the x,y,z data
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
		
		self.nodes = None #will contain a list of all nodes with each as a dict
		self.baseNodePositions = None #will only contain the baseNodes locations as a list of lists (x,y,z)
		self.baseNodeIndices = None #will contain the index for each baseNode within the self.nodes array
		if (path is None):
			self.path = os.path.join(os.getcwd(), baseDir)
		else:
			self.path = os.path.abspath(path) #to make this windows safe
		print('files will be output to:', self.path)

		self.count = 0
		self.lineNmax = lineNmax

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
		#split the node into 8 separate nodes and add these to self.nodes and self.baseNodePositions
		if (self.verbose > 0):
			print('creating child nodes', index, self.nodes[index]['x'], self.nodes[index]['y'], self.nodes[index]['z'], self.nodes[index]['width'])


		if ('particles' not in self.nodes[index]):
			self.populateNodeFromFile(self.nodes[index])


		#create the new nodes and add to the baseNodePositions array
		cx = self.nodes[index]['x'] + self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] + self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] + self.nodes[index]['width']/4.
		n1 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_1', width=self.nodes[index]['width']/2.)

		cx = self.nodes[index]['x'] - self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] + self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] + self.nodes[index]['width']/4.
		n2 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_2',  width=self.nodes[index]['width']/2.)
		
		cx = self.nodes[index]['x'] + self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] - self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] + self.nodes[index]['width']/4.
		n3 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_3', width=self.nodes[index]['width']/2.)
		
		cx = self.nodes[index]['x'] - self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] - self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] + self.nodes[index]['width']/4.
		n4 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_4', width=self.nodes[index]['width']/2.)
		
		cx = self.nodes[index]['x'] + self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] + self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] - self.nodes[index]['width']/4.
		n5 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_5', width=self.nodes[index]['width']/2.)

		
		cx = self.nodes[index]['x'] - self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] + self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] - self.nodes[index]['width']/4.
		n6 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_6', width=self.nodes[index]['width']/2.)
		
		cx = self.nodes[index]['x'] + self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] - self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] - self.nodes[index]['width']/4.
		n7 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_7', width=self.nodes[index]['width']/2.)
		
		cx = self.nodes[index]['x'] - self.nodes[index]['width']/4.
		cy = self.nodes[index]['y'] - self.nodes[index]['width']/4.
		cz = self.nodes[index]['z'] - self.nodes[index]['width']/4.
		n8 = self.createNode([cx, cy, cz], self.nodes[index]['id']+'_8', width=self.nodes[index]['width']/2.)
		
		#add the parent and child indices to the nodes
		childIndices = np.array([], dtype='int')
		for i, n in enumerate([n1, n2, n3, n4, n5, n6, n7, n8]):
			n['parentNodes'] = self.nodes[index]['parentNodes'] + [index]
			childIndex = len(self.nodes)
			self.nodes.append(n)
			self.nodes[index]['childNodes'].append(childIndex)
			self.baseNodePositions = np.append(self.baseNodePositions, [[n['x'],n['y'],n['z']]], axis=0)
			self.baseNodeIndices = np.append(self.baseNodeIndices, childIndex)

			#create these so that I can divide up the parent particles
			if (i == 0):
				childPositions = [[n['x'],n['y'],n['z']]]
			else:
				childPositions = np.append(childPositions, [[n['x'],n['y'],n['z']]], axis=0)
			childIndices = np.append(childIndices, childIndex)
			if (self.verbose > 1):
				print('new baseNode position',[[n['x'],n['y'],n['z']]])
			
		#divide up the particles 
		for p in self.nodes[index]['particles']:
			childIndex = childIndices[self.findClosestNode(np.array([p]), childPositions)]
			self.nodes[childIndex]['particles'].append(p)
			self.nodes[childIndex]['Nparticles'] += 1            
			
		#remove the parent from the self.baseNode* arrays
		i = np.where(self.baseNodeIndices == index)[0]
		if (len(i) > 0 and i != -1):
			self.baseNodePositions = np.delete(self.baseNodePositions, i, axis=0)
			self.baseNodeIndices = np.delete(self.baseNodeIndices,i)
		else:
			print('!!!WARNING, did not find parent in baseNodes', i, index, self.baseNodeIndices)
		
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
			if ( (node['Nparticles'] > 0) and ('particles' in node) and (node['needsUpdate'])):
				parts = np.array(node['particles'])
				x = parts[:,0]
				y = parts[:,1]
				z = parts[:,2]
				df = pd.DataFrame(dict(x=x, y=y, z=z)).sample(frac=1).reset_index(drop=True) #shuffle the order
				nodeFile = os.path.join(self.path, node['id'] + '.csv')
				df.to_csv(nodeFile, index=False)
				node['particles'] = []
				node['needsUpdate'] = False;
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
		if (self.verbose > 1):
			print('reading in file', nodeFile)
		df = pd.read_csv(nodeFile)
		node['particles'] = df.values.tolist()
		node['Nparticles'] = len(node['particles'])
		node['needsUpdate'] = True
		self.count += node['Nparticles']
		
	def getSizeCenter(self):
		#It will be easiest if we can get the center and the size at the start.  This will create overhead to read in the entire file...

		#open the input file
		file = open(os.path.abspath(self.inputFile), 'r') #abspath converts to windows format          

		#set up the variables
		#center = np.array([0.,0.,0.])
		maxPos = np.array([0., 0., 0.])
		minPos = np.array([0., 0., 0.])
		#begin the loop to read the file line-by-line
		lineN = 0
		for line in file:
			lineN += 1
			if (lineN >= self.header):
				#get the x,y,z from the line 
				split = line.strip().split(self.delim)
				x = float(split[self.xCol])
				y = float(split[self.yCol])
				z = float(split[self.zCol])
				#center += np.array([x,y,z])

				maxPos[0] = max([maxPos[0],x])
				maxPos[1] = max([maxPos[1],y])
				maxPos[2] = max([maxPos[2],z])

				minPos[0] = min([minPos[0],x])
				minPos[1] = min([minPos[1],y])
				minPos[2] = min([minPos[2],z])
			if (lineN > (self.lineNmax - self.header - 1)):
				break

		file.close()

		#self.center = center/(lineN - self.header)
		self.center = (maxPos + minPos)/2.
		self.width = np.max(maxPos - minPos)

		if (self.verbose > 0):
			print('have initial center and size', self.center, self.width)


	def appendToFile(self, node, part):
		nodeFile = os.path.join(self.path, node['id'] + '.csv')
		line = str(part['x'])+','+str(part['y'])+','+str(part['z'])+'\n'
		with open(nodeFile, 'a') as f:
			f.write(line.replace(' ',''))


	def compileOctree(self):

		#first get the size and center
		self.getSizeCenter()

		#create the output directory if needed
		if (not os.path.exists(self.path)):
			os.mkdir(self.path)
			
		#initialize the node variables
		self.nodes = [self.createNode(self.center, '0', width=self.width)] #will contain a list of all nodes with each as a dict
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
			if (lineN >= self.header):
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
					
				#add the particle to the node
				#check if there are particles in the node in memory;
				if ('particles' in self.nodes[baseIndex]):
					self.nodes[baseIndex]['particles'].append(point)
				else:
					#this node has been saved to a file, I will try to simply append the value onto the end of the file
					self.appendToFile(self.nodes[baseIndex], dict(x=x, y=y, z=z))
					#read in the file (also add to the count)
					#self.populateNodeFromFile(self.nodes[baseIndex])

				self.nodes[baseIndex]['Nparticles'] += 1

				#check if we need to split the node
				if (self.nodes[baseIndex]['Nparticles'] >= self.NNodeMax):
					self.createChildNodes(baseIndex) 

				#if we are beyond the memory limit, then write the nodes to files and clear the particles from the nodes 
				#(also reset the count)
				if (self.count > self.NMemoryMax):
					self.dumpNodesToFiles()
				
				if (self.verbose > 0 and (lineN % 10000 == 0)):
					print('line : ', lineN)

				if (lineN > (self.lineNmax - self.header - 1)):
					self.dumpNodesToFiles()
					break


		file.close()
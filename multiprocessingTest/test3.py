from multiprocessing import Process, Manager
import numpy as np

class octreeStream:
	def __init__(self):
		self.nodes = Manager().list() #will contain a list of all nodes with each as a dict
		self.managerDict = Manager().dict()
		self.managerDict['center'] = [0,0,0]
		self.managerDict['width'] = 1000

	def createNode(self, center, id='', width=0,):
		#node = Manager().dict(x=center[0], y=center[1], z=center[2], width=width, Nparticles=0, id=id, parentNodes=Manager().list(), childNodes=Manager().list(), particles=Manager().list(), needsUpdate=True)

		node = dict(x=center[0], y=center[1], z=center[2], width=width, Nparticles=0, id=id, parentNodes=Manager().list(), childNodes=Manager().list(), particles=Manager().list(), needsUpdate=True)

		self.nodes.append(node)
		print('CHECKING NEW NODE', len(self.nodes), self.nodes[-1])
		return (node, len(self.nodes) - 1)


	def addToNodes(self, center, iden, width):
		self.createNode(center, iden, width)


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
		print('CHECKING PARENT', parent)
		childIndices = parent['childNodes']
		#childIndices = self.nodes[parentIndex]['childNodes']
		print('CHECKING CHILDINDICES', childIndices)

		while (len(childIndices) > 0):
			childPositions = []
			for i in childIndices:
				childPositions.append([self.nodes[i]['x'], self.nodes[i]['y'], self.nodes[i]['z']])
			parentIndex = childIndices[self.findClosestNodeIndexByDistance(point[0:3], np.array(childPositions))]
			parent = self.nodes[parentIndex]
			childIndices = parent['childNodes']

		return (parent, parentIndex)

	def initialize(self):

		#self.managerDict['count'] = 0

		#create the output directory if needed
		# if (not os.path.exists(self.managerDict['path'])):
		# 	os.makedirs(self.managerDict['path'])
			
		# #remove the files in that directory
		# if (self.managerDict['cleanDir']):
		# 	for f in os.listdir(self.managerDict['path']):
		# 		os.remove(os.path.join(self.managerDict['path'], f))

		#create the base node
		#_ = self.createNode(self.managerDict['center'], '0', width=self.managerDict['width']) 
		n1, index1 = self.createNode([0,0,0], 'test', width=100)
		return (n1, index1)

	def test(self):
		_ = self.initialize()
		#n1, index1 = self.createNode([0,0,0], 'test', width=100)
		print('check',self.nodes[-1])
		print('STARTING JOBS-----------------------------')

		ntest = 1
		jobs = []
		for i in range(ntest):
			center = [i,i,i]
			iden = 'test' + str(i)
			width = i*100
			#jobs.append(Process(target=self.addToNodes, args=(center, iden, width,)))
			jobs.append(Process(target=self.findClosestNode, args=(center,)))
		for j in jobs:
			j.start()
		for j in jobs:
			j.join()

		print('done', self.nodes)



if __name__ == '__main__':
	o = octreeStream()
	o.test()


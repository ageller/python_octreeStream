import multiprocessing as mp
from time import sleep


class A(object):
    def __init__(self, *args, **kwargs):
        # do other stuff
        managerDict = mp.Manager().dict()
        managerDict['test'] = [1,2,3,4]
        self.manager = managerDict

    def do_something(self, i):
        sleep(0.2)
        print('%s * %s = %s' % (i, i, i*i))

    def run(self):
        processes = []

        for i in range(10):
            p = mp.Process(target=self.do_something, args=(i,))
            processes.append(p)

        [x.start() for x in processes]


if __name__ == '__main__':
    a = A()
    a.run()

'''
from octreeStreamMultiprocessing import octreeStream as octreeStreamMultiprocessing
oM1 = octreeStreamMultiprocessing('/Users/ageller/VISUALIZATIONS/FIREdata/m12i_res7100/snapdir_600/snapshot_600.0.hdf5', 
                 h5PartKey = 'PartType0', keyList = ['Coordinates', 'Density', 'Velocities'],
                 NNodeMax = 10000, NMemoryMax = 5e4, Nmax=1e5, verbose=1, minWidth=1e-4,
                 cleanDir = True,
                 path='/Users/ageller/VISUALIZATIONS/octree_threejs_python/WebGL_octreePartition/src/data/junk/octreeNodes/Gas')
oM1.compileOctree()
'''
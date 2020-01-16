try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = 0
        self._sumsq = epsilon
        self._count = epsilon
        self.shape = shape

    def mean(self):
        return self._sum / self._count
    
    def std(self):
        return np.sqrt(np.maximum((self._sumsq / self._count) - self.mean()**2, 1e-2))

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        # update
        self._sum += totalvec[0:n].reshape(self.shape)
        self._sumsq += totalvec[n:2*n].reshape(self.shape)
        self._count += totalvec[2*n]

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.std(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean(), rms.std()]

        assert np.allclose(ms1, ms2)

def test_dist():
    np.random.seed(0)
    p1,p2,p3=(np.random.randn(3,1), np.random.randn(4,1), np.random.randn(5,1))
    q1,q2,q3=(np.random.randn(6,1), np.random.randn(7,1), np.random.randn(8,1))

    # p1,p2,p3=(np.random.randn(3), np.random.randn(4), np.random.randn(5))
    # q1,q2,q3=(np.random.randn(6), np.random.randn(7), np.random.randn(8))

    comm = MPI.COMM_WORLD
    assert comm.Get_size()==2
    if comm.Get_rank()==0:
        x1,x2,x3 = p1,p2,p3
    elif comm.Get_rank()==1:
        x1,x2,x3 = q1,q2,q3
    else:
        assert False

    rms = RunningMeanStd(epsilon=0.0, shape=(1,))

    rms.update(x1)
    rms.update(x2)
    rms.update(x3)

    bigvec = np.concatenate([p1,p2,p3,q1,q2,q3])

    def checkallclose(x,y):
        print(x,y)
        return np.allclose(x,y)

    assert checkallclose(
        bigvec.mean(axis=0),
        rms.mean(),
    )
    assert checkallclose(
        bigvec.std(axis=0),
        rms.std(),
    )

if __name__ == "__main__":
    # Run with mpirun -np 2 python <filename>
    test_dist()

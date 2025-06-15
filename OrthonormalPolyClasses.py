import numpy as np
import scipy as sp
from math import comb



class Jacobi:


    def __init__(self, alpha, beta, kMax):
        self.alpha = alpha
        self.beta = beta
        self.kMax = kMax
        kArray = np.arange(1,max(kMax,2))
        twoNAB = 2*kArray + alpha + beta
        self.Aks = np.divide(np.multiply(twoNAB+1, twoNAB+2),
                             2 * np.multiply(kArray+1, kArray + alpha + beta + 1))
        self.Bks = np.divide((alpha**2-beta**2)*(twoNAB+1),
                             2 * np.multiply(np.multiply(kArray+1, kArray + alpha + beta + 1), twoNAB))
        self.Cks = np.divide(np.multiply(np.multiply(kArray + alpha, kArray + beta), (twoNAB+2)),
                              np.multiply(np.multiply(kArray+1, kArray + alpha + beta + 1), twoNAB))



    def __call__(self, x, desiredDeg):
        return self.calcAtX(x)[desiredDeg]/comb(self.alpha+desiredDeg,desiredDeg)

    def calcAtX(self, x):
        x = np.asarray(x)
        solutionValues = np.ones((max(self.kMax, 1) + 1,)+x.shape)
        solutionValues[1] = (((self.alpha + self.beta) / 2 + 1) * x + (self.alpha - self.beta) / 2)
        for degree in range(2, self.kMax + 1):
            coefIdx = degree - 2
            solutionValues[degree] = ((self.Aks[coefIdx] * x + self.Bks[coefIdx]) * solutionValues[degree - 1]
                                      - self.Cks[coefIdx] * solutionValues[degree - 2])
        return solutionValues

    def calcNormalizedAtX(self, x):
        solutionValues = self.calcAtX(x)
        kArray = np.arange(self.kMax + 1)
        solutionValuesShape = list(solutionValues.shape)
        for i in range(1,len(solutionValuesShape)):
            solutionValuesShape[i] = 1
        normalizationFactor = np.divide(1,sp.special.comb(kArray + self.alpha, kArray))
        normalizationFactor = np.reshape(normalizationFactor,tuple(solutionValuesShape),copy=False)
        return np.multiply(solutionValues, normalizationFactor)


class Disk:

    def __init__(self, alpha, gamma, kMax):
        self.alpha = alpha
        self.gamma = gamma
        self.kMax = kMax
        self.Jacobi = Jacobi(alpha, gamma, kMax)


    def __call__(self, r, theta, desiredDeg):
        return self.calcAtRTheta(r, theta)[desiredDeg]

    def calcAtRTheta(self, r, theta):
        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            rMesh, thetaMesh = np.meshgrid(r, theta)
            return (np.power(rMesh, self.gamma) * np.cos(thetaMesh * self.gamma) *
                    self.Jacobi.calcNormalizedAtX(2 * (rMesh ** 2) - 1))
        else:
            return np.power(r,self.gamma) * np.cos(theta*self.gamma) * self.Jacobi.calcNormalizedAtX(2*(r**2) - 1)


class DiskCombi:
    def __init__(self, alpha, coefficientArray):
        self.alpha = alpha
        self.kMax = coefficientArray.shape[1] - 1
        self.gammaMax = coefficientArray.shape[0] - 1
        self.DiskList = [Disk(self.alpha, gamma, self.kMax) for gamma in range(self.gammaMax + 1)]
        self.coefficientArray = coefficientArray

    def __call__(self, r, theta):
        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            rMesh, thetaMesh = np.meshgrid(r, theta)
            meshDims = len(rMesh.shape)
            solArray = np.array([diskInst.calcAtRTheta(rMesh, thetaMesh) for diskInst in self.DiskList])
        else:
            meshDims = len(r.shape)
            solArray = np.array([diskInst.calcAtRTheta(r, theta) for diskInst in self.DiskList])
        solDimLength = len(solArray.shape) - meshDims
        solDims = tuple([axisIdx for axisIdx in range(solDimLength)])
        coefsDesiredShape = list(solArray.shape)
        for idx in range(solDimLength, len(coefsDesiredShape)):
            coefsDesiredShape[idx] = 1
        desiredCoefsArray = np.reshape(self.coefficientArray, tuple(coefsDesiredShape), copy=True)
        return np.sum(np.multiply(solArray, desiredCoefsArray), axis=solDims)




diskPoly = Disk(1,2,3)
# print(diskPoly(np.sqrt(1/2),0,2))

coefArray = np.array([[0.5,0,0.25],[0,0,0],[0,0.25,0]])
# coefArray = np.random.random((3,3))
coefArray /= np.sum(coefArray)
diskCombi = DiskCombi(1,coefArray)
# print(diskCombi(np.sqrt(1/2),0))

innerProductArray = np.linspace(0,1,5,endpoint=True)
sqrtIPA = np.sqrt(innerProductArray)
angleArray = np.linspace(0, 3 * np.pi, 8, endpoint=True)
outputArray = diskCombi(sqrtIPA, angleArray)
print(outputArray)


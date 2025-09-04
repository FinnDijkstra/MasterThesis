import time

import numpy as np
import scipy as sp
from math import comb
# from threadpoolctl import threadpool_info
# import plot2dCharts


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
            jacobiVals = self.Jacobi.calcNormalizedAtX(2 * (r ** 2) - 1)
            # thetaShape = tuple(1 for _ in range(len(theta.shape)))
            # jacobiReshape = thetaShape + r.shape + (self.kMax+1,)
            # jacobiValsV2 = jacobiVals.T.reshape(jacobiReshape)
            # rParts = np.power(rMesh, self.gamma)*jacobiValsV2
            # outputsTry1 = rParts * np.cos(thetaMesh * self.gamma)
            # thetaShape = tuple(1 for _ in range(len(theta.shape)))
            # jacobiReshape = jacobiVals.shape + thetaShape
            # jacobiValsV2 = jacobiVals.reshape(jacobiReshape)
            outputVals = np.zeros((self.kMax+1,)+rMesh.shape)
            for curK in range(self.kMax + 1):
                outputVals[curK] = (np.power(rMesh, self.gamma) *
                                        np.cos(thetaMesh * self.gamma) * jacobiVals[curK])
            # outputVals = np.array(outputValsList)
            x=1
            return outputVals
            # return (np.power(rMesh, self.gamma) * np.cos(thetaMesh * self.gamma) *jacobiValsV2
            #         )
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
            solArray = np.array([diskInst.calcAtRTheta(r, theta) for diskInst in self.DiskList])
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

    def allValues(self, r, theta):
        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            rMesh, thetaMesh = np.meshgrid(r, theta)
            solArray = np.array([diskInst.calcAtRTheta(r, theta) for diskInst in self.DiskList])
        else:
            solArray = np.array([diskInst.calcAtRTheta(r, theta) for diskInst in self.DiskList])
        # for gamma in range(self.gammaMax+1):
        #     for k in range(self.kMax+1):
        #         for ridx in range(0):
        #             for thetaidx in range((theta.shape[0] - 1) / 2):
        #                 if solArray[gamma,k,ridx,thetaidx] != solArray[gamma,k,ridx,-1-thetaidx]:
        #                     print("bad pair")
        return solArray


    def sumAtRow(self, r, theta, gamma):
        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            rCorrect, thetaCorrect = np.meshgrid(r, theta)
        else:
            rCorrect = r
            thetaCorrect = theta
        values = self.DiskList[gamma].calcAtRTheta(rCorrect, thetaCorrect)
        coefs = self.coefficientArray[gamma]
        reshapeShape = [1 for _ in range(len(rCorrect.shape)+1)]
        reshapeShape[0] = coefs.shape[0]
        coefsCorrect = coefs.reshape(reshapeShape)
        return np.sum(np.multiply(values,coefsCorrect), axis=0)


    def findMinimalParams(self,r,theta):
        r = np.asarray(r)
        theta = np.asarray(theta)
        rShape = r.shape
        thetaShape = theta.shape
        kNr = self.kMax+1
        gammaNr = self.gammaMax+1
        solOutputs = self.allValues(r,theta)
        solOutputsReshaped = solOutputs.reshape((gammaNr*kNr,) +thetaShape+ rShape)
        flatMin = np.min(solOutputsReshaped, axis=0)
        flatMinIndices = np.zeros((theta.shape[0],r.shape[0]),dtype=int)
        minVals = np.zeros_like(flatMinIndices, dtype=np.float64)
        # trustableRadius = np.ones_like(r,dtype=bool)
        # trustableAngles = np.ones_like(theta,dtype=bool)
        invalidCoordsCounter = 0
        for ridx in range(r.shape[0]):
            for thetaidx in range(theta.shape[0]):
                curMin = flatMin[thetaidx,ridx]

                solsCur = solOutputsReshaped[:,thetaidx,ridx]
                validIndices = np.argwhere(solsCur<curMin+1e-10)
                minimalIndex = np.min(validIndices)
                flatMinIndices[thetaidx,ridx] = minimalIndex

                if curMin > -1e-2:
                    invalidCoordsCounter += 1
                if ridx == 0:
                    curMin = -1
                minVals[thetaidx, ridx] = curMin
        flat_sorted = np.sort(minVals.ravel())

        # Determine the (x+1)th largest value
        clip_value = flat_sorted[-max(invalidCoordsCounter+1,(r.shape[0]*theta.shape[0]//10000 + 1))]

        # Clip top x values to the (x+1)th value
        minVals = np.minimum(minVals, clip_value)


        # trustAbleArray = trustableAngles[:,None] | trustableRadius[None,:]
        # clipForMin = np.max(minVals[trustAbleArray])
        # untrustableArray = ~trustAbleArray
        # # untrustableRadius = ~trustableRadius
        # minVals[untrustableArray] = np.clip(minVals[untrustableArray],-1,clipForMin)

        # validIndices = np.argwhere(solOutputsReshaped<flatMin+1e-10)
        # flatMinIndices = np.min(validIndices,axis=0)
        gammaIndices = flatMinIndices // kNr
        kIndices = flatMinIndices % kNr
        # kIndices[:,0] = 0
        # gammaIndices[0,0] = 0

        # for ridx in range(r.shape[0]):
        #     for thetaidx in range(1,(theta.shape[0] - 1) // 2):
        #         curR = r[ridx]
        #         curTheta = theta[thetaidx]
        #         curThetaMin = theta[-1-thetaidx]
        #         gamma1 = gammaIndices[thetaidx, ridx]
        #         gamma2 = gammaIndices[-1-thetaidx, ridx]
        #         k1 = kIndices[thetaidx, ridx]
        #         k2 = kIndices[-1-thetaidx, ridx]
        #         gamma3 = gamma1-gamma2
        #         k3 = k1-k2
        #         desiredVal = solOutputsReshaped[flatMinIndices[thetaidx,ridx], thetaidx,ridx]
        #         # desiredVal2 = solOutputsReshaped[flatMinIndices[-1-thetaidx,ridx], -1-thetaidx,ridx]
        #         # leftVal = solOutputs[gamma1,k1,thetaidx,ridx]
        #         # rightVal = solOutputs[gamma2,k2,-1-thetaidx,ridx]
        #
        #         x=1
        #         if gamma1 != gamma2 or k1 != k2:
        #             print("bad pair")
        return gammaIndices, kIndices, minVals




class FastRadialEstimator:
    def __init__(self, slow_function, resolution,gamma):
        self.slow_function = slow_function  # Callable: f(r), r in [0,1]
        self.resolution = resolution

        # Precompute points at evenly spaced radii
        self.r_grid = np.linspace(0, 1, resolution, endpoint=True)
        self.values = self.slow_function(self.r_grid,0,gamma)[0]

    def __call__(self, r_input):
        r_input = np.asarray(r_input)
        r_abs = np.abs(r_input)
        # Clip inputs to [0, 1] to stay within interpolation range
        r_clipped = np.clip(r_abs, 0, 1)

        # Normalize to get fractional index
        scaled = r_clipped * (self.resolution - 1)
        i = np.clip(np.floor(scaled).astype(int),0, self.resolution - 1)
        idx2 = np.clip(np.ceil(scaled).astype(int), 0, self.resolution - 1)
        frac = scaled - i

        # Avoid going out of bounds on the upper edge
        # i = np.clip(i, 0, self.resolution - 1)

        # Interpolate linearly
        result = (1 - frac) * self.values[i] + frac * self.values[idx2]
        return result


class FastDiskCombiEstimator:
    def __init__(self, alpha, coefficientArray, resolution):
        self.alpha = alpha
        self.kMax = coefficientArray.shape[1] - 1
        self.gammaMax = coefficientArray.shape[0] - 1
        self.coefficientArray = coefficientArray
        self.diskCombi = DiskCombi(alpha,coefficientArray)
        self.fastRadialEstimatorList = [FastRadialEstimator(self.diskCombi.sumAtRow, resolution, gammaIdx)
                                        for gammaIdx in range(self.gammaMax+1)]

    def __call__(self, r, theta):
        r = np.asarray(r)
        theta = np.asarray(theta)
        if r.shape != theta.shape:
            rCorrect, thetaCorrect = np.meshgrid(r, theta)
        else:
            rCorrect = r
            thetaCorrect = theta
        cosFactors = np.array([np.cos(gammaIdx*thetaCorrect) for gammaIdx in range(self.gammaMax+1)])
        radialFactors = np.array([FRE(rCorrect) for FRE in self.fastRadialEstimatorList])
        return np.sum(np.multiply(cosFactors,radialFactors),axis=0)
        # cosFactors = [np.cos(gammaIdx * thetaCorrect) for gammaIdx in range(self.gammaMax + 1)]
        # radialFactors = [FRE(rCorrect) for FRE in self.fastRadialEstimatorList]
        # return sum(cosFactors[gammaIdx]*radialFactors[gammaIdx] for gammaIdx in range(self.gammaMax+1))



def bench(n=6, reps=600):
    coefs = np.random.random((15,15))
    coefs /= np.sum(coefs)
    estimatorDisk = FastDiskCombiEstimator(1,coefs,1000)
    diskPoly = DiskCombi(1,coefs)
    t1 = 0.0
    t2 = 0.0
    totalDiff = 0.0
    for _ in range(reps):
        A = np.random.random((n, n))
        B = np.random.random((n, n))*2*np.pi
        t0 = time.perf_counter()
        C1 = estimatorDisk(A,B)
        t1 += time.perf_counter() - t0
        t0 = time.perf_counter()
        C2 = diskPoly(A,B)
        t2 += time.perf_counter() - t0
        totalDiff += np.sum(np.abs(C1-C2))
    print(f"Fast {t1}s, Original {t2}s, diff {totalDiff/n/n}")
    return


if __name__ == "__main__":
    testSpeed = True
    if testSpeed:
        for I in range(5):
            bench()
    else:
        rRes = 500 + 1
        thetaRes = 1000 + 1
        rGrid = np.linspace(0, 1, rRes, endpoint=True)
        r2Grid = np.sqrt(rGrid)
        thetaGrid = np.linspace(0, 2 * np.pi, thetaRes, endpoint=True)
        rMesh,thetaMesh  = np.meshgrid(rGrid, thetaGrid)
        r2Mesh, theta2Mesh = np.meshgrid(r2Grid, thetaGrid)
        cosMesh = np.cos(thetaMesh)
        sinMesh = np.sin(thetaMesh)
        xMesh = cosMesh*rMesh
        yMesh = sinMesh*rMesh
        coefArray = np.ones((50, 20)) / 4000
        complexDimension = 4
        totalDisk = DiskCombi(complexDimension-2, coefArray)
        # bestGamma, bestK = totalDisk.findMinimalParams(rGrid, thetaGrid)
        bestGamma, bestK, minimalValues = totalDisk.findMinimalParams(r2Grid,thetaGrid)
        # plot2dCharts.groupedPolarPlot(rMesh, thetaMesh, bestGamma, bestK, 5, 5)
        graphTitle = "Parameters for best disk polynomial\n" + fr"in $\mathbb{{C}}^{{{complexDimension}}}$ "
        # plot2dCharts.groupedPolarPlot(r2Mesh,theta2Mesh,bestGamma,bestK,10,10,graphTitle,
        #                               legendBool=False, minVals = minimalValues, plot3dBool = True, comDim=complexDimension)
        bestGammaMask = (bestGamma>0)
        bestKMask = (bestK >0)
        print(np.sum(bestKMask*bestGammaMask))
        x=1

# diskPoly = Disk(1,2,3)
# # print(diskPoly(np.sqrt(1/2),0,2))
#
# coefArray = np.array([[0.5,0,0.25],[0,0,0],[0,0.25,0]])
# # coefArray = np.random.random((3,3))
# coefArray /= np.sum(coefArray)
# diskCombi = DiskCombi(1,coefArray)
# # print(diskCombi(np.sqrt(1/2),0))
#
# innerProductArray = np.linspace(0,1,5,endpoint=True)
# sqrtIPA = np.sqrt(innerProductArray)
# angleArray = np.linspace(0, 3 * np.pi, 8, endpoint=True)
# outputArray = diskCombi(sqrtIPA, angleArray)
# print(outputArray)

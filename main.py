# from typing import final
import time
import gurobipy as gp
from gurobipy import GRB
import math
import scipy
import numpy as np
import fractions
import random
import scipy.stats as st
from scipy.special import hyp2f1, loggamma, binom
# import sympy as sym
import pandas as pd
import json
import os
import datetime
from multiprocessing import Pool, cpu_count
from itertools import combinations
import improvedBQPNLP


from matplotlib import pyplot as plt

import SphereBasics
from line_profiler import LineProfiler
from numpy.polynomial import Polynomial

from SphereBasics import complexWeightPolyCreator, complexDoubleCap, radialIntegrator, \
    integratedWeightPolyCreator, createRandomPointsComplex

from OrthonormalPolyClasses import Jacobi, Disk, DiskCombi, FastRadialEstimator, FastDiskCombiEstimator

nrOfVars = 10
nrOfTests = 200
dimension = 4
distanceForbidden = 0
jacobiType = "new"

polar2z = lambda r, theta: r * np.exp(1j * theta)
z2polar = lambda z: (np.abs(z), np.angle(z))

def thetaBasedOnThetapart(tpVal):
    return (-tpVal)/(1-tpVal)

def gammaFunction(nVar):
    if nVar % 0.5 != 0 or nVar <= 0:
        print("Gamma currently not implemented for non multiples of 0.5")
        return 1
    if nVar % 1 == 0:
        return math.factorial(math.floor(nVar)-1)
    oddNr = math.floor(nVar)
    muFactor = 1
    for i in range(oddNr):
        muFactor *= 2*i + 1
    finalValue = math.sqrt(math.pi)*muFactor/(2**oddNr)
    return finalValue


def jacobiValue(degree, alphaJac, betaJac, location):
    if degree == 0:
        return 1
    if location == 1:
        return gammaFunction(alphaJac + degree + 1)/(gammaFunction(degree + 1)*gammaFunction(alphaJac + 1))
    finalValue = gammaFunction(alphaJac + degree + 1)
    finalValue /= gammaFunction(degree + 1)
    finalValue /= gammaFunction(degree + alphaJac + betaJac + 1)
    finalValue *= sum(math.comb(degree, sumIter)*gammaFunction(degree + alphaJac + betaJac + sumIter + 1)
                      / gammaFunction(alphaJac + sumIter + 1) * (((location - 1) / 2)**sumIter)
                      for sumIter in range(degree+1))
    return finalValue

def normedJacobiValue(degree, alpha, beta, location):
    # if location == -10:
    #     functDegree = 2*degree
    # else:
    #     functDegree = degree
    if degree == 0 or location == 1:
        return 1
    return scipy.special.eval_jacobi(degree, alpha, beta, location)/scipy.special.eval_jacobi(degree, alpha, beta, 1)


def realDiskPoly(alpha, gamma, k, radius, angle):
    return (radius**gamma)*math.cos(gamma*angle*math.pi)*normedJacobiValue(k, alpha, gamma, float(2*(radius**2)-1))


def vectorizeRDP(alpha, gamma, k, zArray):
    jacobiLocs = 2*(np.abs(zArray)**2)-1
    gammaFactor = np.real(np.power(zArray,gamma))
    jacobiFactor = scipy.special.eval_jacobi(k, alpha, gamma, 1)
    jacobiVals = scipy.special.eval_jacobi(k, alpha, gamma, jacobiLocs)
    diskVals = gammaFactor*jacobiVals/jacobiFactor
    return diskVals


def imaginaryDiskPoly(alpha, gamma, k, radius, angle):
    return (radius**abs(gamma))*math.sin(gamma*angle*math.pi)*normedJacobiValue(k, alpha, abs(gamma), float(2*(radius**2)-1))


def negAngle(gamma, angle):
    totAngle = gamma*angle - 1/2
    totAngle = totAngle % 2.0
    return totAngle <= 1



def makeModel(name):
    m = gp.Model(f"3.11 implementation")
    m.setParam("OutputFlag", 1)
    # m.setParam(GRB.Param.Presolve, 0)
    m.setParam(GRB.Param.DisplayInterval, 10)
    # m.setParam(GRB.Param.Method, 2)
    # m.setParam("ConcurrentMethod", 2)
    # m.setParam(GRB.Param.BarIterLimit, 0)

    alpha = (dimension-3.0)/2.0
    fList = m.addVars(nrOfVars, vtype=GRB.CONTINUOUS, lb=0, name="clusterWeights")
    sphereMeasure = 2*(math.pi**(dimension/2)/gammaFunction(dimension/2))
    jacobiList = [normedJacobiValue(jacobiIter, alpha, alpha, distanceForbidden) for jacobiIter in range(nrOfVars)]
    # m.addConstr((gp.quicksum((fList[fIter] * jacobiList[fIter])
    #                                           for fIter in range(nrOfVars)) == 0),
    #                                                 name=f"Jacobi Constraint")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) <= 0,
                name="Jacobi_ConstraintS")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) >= 0,
                name="Jacobi_ConstraintL")

    m.addConstr((1 == gp.quicksum(fList[fIter]
                                  for fIter in range(nrOfVars))),
                name=f"Measure Constraint")

    objective = (sphereMeasure**2 * fList[0])
    m.setObjective(fList[0], GRB.MAXIMIZE)
    m.update()
    m.optimize()
    print(m.ObjVal)



def makeModelv2(alpha, beta, newForbidden):
    m = gp.Model(f"Gen Theta implementation")
    m.setParam("OutputFlag", 1)
    # m.setParam(GRB.Param.Presolve, 0)
    m.setParam(GRB.Param.DisplayInterval, 10)
    # m.setParam(GRB.Param.Method, 2)
    # m.setParam("ConcurrentMethod", 2)
    # m.setParam(GRB.Param.BarIterLimit, 0)

    # alpha = (dimension-3.0)/2.0
    fList = m.addVars(nrOfVars, vtype=GRB.CONTINUOUS, lb=0, name="clusterWeights")
    sphereMeasure = 2*(math.pi**(dimension/2)/gammaFunction(dimension/2))
    jacobiList = [normedJacobiValue(jacobiIter, alpha, beta, newForbidden) for jacobiIter in range(nrOfVars)]
    # m.addConstr((gp.quicksum((fList[fIter] * jacobiList[fIter])
    #                                           for fIter in range(nrOfVars)) == 0),
    #                                                 name=f"Jacobi Constraint")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) <= 0,
                name="Jacobi_ConstraintS")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) >= 0,
                name="Jacobi_ConstraintL")

    m.addConstr((1 == gp.quicksum(fList[fIter]
                                  for fIter in range(nrOfVars))),
                name=f"Measure Constraint")

    # objective = (sphereMeasure**2 * fList[0])
    m.setObjective(fList[0], GRB.MAXIMIZE)
    m.update()
    m.optimize()
    print(m.ObjVal)



def makeModelCascading(alpha, oldForbidden, newForbidden):
    m = gp.Model(f"Gen Theta implementation")
    m.setParam("OutputFlag", 1)
    # m.setParam(GRB.Param.Presolve, 0)
    m.setParam(GRB.Param.DisplayInterval, 10)
    # m.setParam(GRB.Param.Method, 2)
    # m.setParam("ConcurrentMethod", 2)
    # m.setParam(GRB.Param.BarIterLimit, 0)

    # alpha = (dimension-3.0)/2.0
    fList = m.addVars(nrOfVars, vtype=GRB.CONTINUOUS, lb=0, name="clusterWeights")
    sphereMeasure = 2*(math.pi**(dimension/2)/gammaFunction(dimension/2))
    jacobiMaster = [[] for _ in range(nrOfVars)]
    radZ = abs(oldForbidden)
    if radZ == 0:
        angleZ = 1
    else:
        angleZ = oldForbidden / radZ
    for totDegree in range(nrOfVars):
        for mDegree in range(totDegree+1):
            nDegree = totDegree - mDegree
            jacobiStep1 = normedJacobiValue(min(mDegree,nDegree), alpha, abs(mDegree-nDegree), newForbidden)
            curRad = radZ**(abs(mDegree-nDegree))
            curAngle = angleZ**(mDegree-nDegree)
            jacobiMaster[totDegree].append(curRad*curAngle*jacobiStep1)

    jacobiList = [normedJacobiValue(jacobiIter, alpha, 0, newForbidden) for jacobiIter in range(nrOfVars)]
    # m.addConstr((gp.quicksum((fList[fIter] * jacobiList[fIter])
    #                                           for fIter in range(nrOfVars)) == 0),
    #                                                 name=f"Jacobi Constraint")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) <= 0,
                name="Jacobi_ConstraintS")
    m.addConstr(gp.quicksum(fList[fIter] * jacobiList[fIter] for fIter in range(nrOfVars)) >= 0,
                name="Jacobi_ConstraintL")

    m.addConstr((1 == gp.quicksum(fList[fIter]
                                  for fIter in range(nrOfVars))),
                name=f"Measure Constraint")

    # objective = (sphereMeasure**2 * fList[0])
    m.setObjective(fList[0], GRB.MAXIMIZE)
    m.update()
    m.optimize()
    print(m.ObjVal)


def findMinFixedGammaBlock(forbiddenRadius, forbiddenAngle, alpha, gamma, kDepth):
    if negAngle(gamma, forbiddenAngle)  or forbiddenRadius == 1:
        return (forbiddenRadius**gamma) * math.cos(gamma*forbiddenAngle*math.pi), 0
    else:
        rdpValList = [0] * kDepth
        for i in range(kDepth):
            rdpValList[i] = realDiskPoly(alpha, gamma, i, forbiddenRadius, forbiddenAngle)
        minVal = min(rdpValList)
        return minVal, rdpValList.index(minVal)


def findMinFixedGammaVar(forbiddenRadius, forbiddenAngle, alpha, gamma):
    if negAngle(gamma, forbiddenAngle) or forbiddenRadius == 1:
        return (forbiddenRadius**gamma) * math.cos(gamma*forbiddenAngle*math.pi), 0
    else:
        minVal = 1
        curIdx = 1
        while curIdx < 1000:
            curVal = realDiskPoly(alpha, gamma, curIdx, forbiddenRadius, forbiddenAngle)
            if curVal <= minVal:
                minVal = curVal
                curIdx += 1
            else:
                return minVal, curIdx-1
        return minVal, curIdx-1



def findMinBlock(forbiddenRadius, forbiddenAngle, alpha, kDepth, gammaDepth=0):
    minValTup = (1, 0, 0)
    if gammaDepth == 0 and 0 < forbiddenRadius < 1:
        gammaCal, gammaCalK = findMinFixedGammaBlock(forbiddenRadius, forbiddenAngle, alpha, 0, kDepth)
        gammaDepth = math.ceil(math.log(-gammaCal, forbiddenRadius))
        minValTup = (gammaCal, 0, gammaCalK)
    elif gammaDepth == 0 and forbiddenRadius == 1:
        gammaDepth = fractions.Fraction(forbiddenAngle).denominator + 1
        kDepth = 1
    elif forbiddenRadius == 0:
        gammaCal, gammaCalK = findMinFixedGammaBlock(forbiddenRadius, forbiddenAngle, alpha, 0, kDepth)
        minValTup = (gammaCal, 0, gammaCalK)
        return minValTup
    else:
        gammaCal, gammaCalK = findMinFixedGammaBlock(forbiddenRadius, forbiddenAngle, alpha, 0, kDepth)
        minValTup = (gammaCal, 0, gammaCalK)
    for curGamma in range(1, gammaDepth):
        curGammaMin, curGammaMinK = findMinFixedGammaBlock(forbiddenRadius, forbiddenAngle, alpha, curGamma, kDepth)
        if curGammaMin < minValTup[0]:
            minValTup = (curGammaMin, curGamma, curGammaMinK)
    return minValTup


def findMinVar(forbiddenRadius, forbiddenAngle, alpha, gammaDepth=0):
    minValTup = (1, 0, 0)
    # Spiral down
    if gammaDepth == 0 and 0 < forbiddenRadius < 1:
        gammaCal, gammaCalK = findMinFixedGammaVar(forbiddenRadius, forbiddenAngle, alpha, 0)
        gammaDepth = math.ceil(math.log(-gammaCal, forbiddenRadius))
        minValTup = (gammaCal, 0, gammaCalK)
    # Circle
    elif gammaDepth == 0 and forbiddenRadius == 1:
        gammaDepth = fractions.Fraction(forbiddenAngle).denominator + 1
        kDepth = 1
    # Simple
    elif forbiddenRadius == 0:
        gammaCal, gammaCalK = findMinFixedGammaVar(forbiddenRadius, forbiddenAngle, alpha, 0)
        minValTup = (gammaCal, 0, gammaCalK)
        return minValTup
    # gammaMax preditermined
    else:
        gammaCal, gammaCalK = findMinFixedGammaVar(forbiddenRadius, forbiddenAngle, alpha, 0)
        minValTup = (gammaCal, 0, gammaCalK)
    MinimumFound = False
    betterMinimaFound = 0
    for curGamma in range(1, gammaDepth):
        curGammaMin, curGammaMinK = findMinFixedGammaVar(forbiddenRadius, forbiddenAngle, alpha, curGamma)
        if curGammaMin < minValTup[0]:
            minValTup = (curGammaMin, curGamma, curGammaMinK)
    #         if MinimumFound:
    #             betterMinimaFound += 1
    #             MinimumFound = False
    #     else:
    #         MinimumFound = True
    # if betterMinimaFound > 0:
    #     print(f"{betterMinimaFound + 1} local minima found for R: {forbiddenRadius}, theta: {forbiddenAngle}")
    return minValTup




# Press the green button in the gutter to run the script.
def tierListPrinter(Dictionary, nrPrinted=0):
    if nrPrinted == 0:
        nrPrinted = len(Dictionary)
    for key, value in Dictionary.items():
        leaderboardValue = value[3]*value[4]
        Dictionary[key].append(leaderboardValue)
    sortedDict = sorted(Dictionary.items(), key=lambda item: item[1][-1], reverse=True)
    print("The leaderbord of how unique the minimum is (calculated by gamma*k):")
    for placement in range(nrPrinted):
        print(f"Nr ({placement+1}): Test nr {sortedDict[placement][0]} with a score of {sortedDict[placement][1][-1]}")
        print(f"\t Radius {sortedDict[placement][1][1]}, Angle {sortedDict[placement][1][2]}"
              f", gamma {sortedDict[placement][1][3]}, k {sortedDict[placement][1][4]}.")
    sortedDict1 = sorted(Dictionary.items(), key=lambda item: item[1][-3], reverse=True)
    print("The leaderbord of how many gamma needed to be checked:")
    for placement in range(nrPrinted):
        print(
            f"Nr ({placement + 1}): Test nr {sortedDict1[placement][0]} with a score of {sortedDict1[placement][1][-1]}")
        print(f"\t Radius {sortedDict1[placement][1][1]}, Angle {sortedDict1[placement][1][2]}"
              f", gamma {sortedDict1[placement][1][3]}, k {sortedDict1[placement][1][4]}.")
    sortedDict2 = sorted(Dictionary.items(), key=lambda item: item[1][-2], reverse=True)
    print("The leaderbord of how many k needed to be checked:")
    for placement in range(nrPrinted):
        print(
            f"Nr ({placement + 1}): Test nr {sortedDict2[placement][0]} with a score of {sortedDict2[placement][1][-1]}")
        print(f"\t Radius {sortedDict2[placement][1][1]}, Angle {sortedDict2[placement][1][2]}"
              f", gamma {sortedDict2[placement][1][3]}, k {sortedDict2[placement][1][4]}.")


def characteristicMatrix(nrOfPoints):
    nrOfSets = 2**nrOfPoints
    setIndices = np.arange(nrOfSets)
    charMatrix = np.empty((nrOfPoints, nrOfSets), dtype=bool)
    for pointIdx in range(nrOfPoints):
        relativeIndices = setIndices//(2**pointIdx)
        charIndices = relativeIndices % 2
        charMatrix[pointIdx] = charIndices
    return charMatrix


# possible alterations to fix model:
# is it a_0>=(sum_{0<=i<=inf}(a_i))^2 or a_0>=(sum_{1<=i<=inf}(a_i))^2
# is it sum_{i in sets}(lambda_i)==1 or sum_{i in sets of v}(lambda_i)==1 for all v in V
# is obj sum_{0<=i<=inf}(a_i) or sum_{1<=i<=inf}(a_i)
# do we do Disk(v,v) = sum_{i in sets}(lambda_i I_{v in set i})

def modelBQP(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, pointMatrix=None, nrOfPoints=2):
    alpha = complexDimension - 2
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)
    if pointMatrix is None:
        pointMatrix = SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints,
                                                             baseZ=forbiddenZ)[0]
    else:
        nrOfPoints = pointMatrix.shape[0]
    charMatrix = characteristicMatrix(nrOfPoints)
    nrOfSets = charMatrix.shape[1]
    nrOfPoints = pointMatrix.shape[0]
    innerProductMatrix = pointMatrix @ pointMatrix.conj().T
    diskPolyValueTensor = np.zeros((gammaMax, kMax, nrOfPoints, nrOfPoints))
    for gamma in range(gammaMax):
        for k in range(kMax):
            diskPolyValueTensor[gamma][k] = vectorizeRDP(alpha, gamma, k, innerProductMatrix)
    m = gp.Model(f"BQP model for forbidden innerproduct {forbiddenZ} in dimension {complexDimension}, "
                 f"with {nrOfPoints} BQP points,"
                 f" gamma checked {gammaMax}, k checked {kMax}")
    m.setParam("OutputFlag", 1)
    m.setParam(GRB.Param.DisplayInterval, 10)

    diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.CONTINUOUS, lb=0, name="diskpoly weights")
    setWeights = m.addVars(nrOfSets, vtype=GRB.CONTINUOUS, lb=0, name="set weights")

    m.addConstr(setWeights.sum() == 1, name="convex description of sets")
    m.addConstr(diskPolyWeights[0,0] - diskPolyWeights.sum()*diskPolyWeights.sum() >= 0, "SDP contraint")
    m.addConstr(gp.quicksum(diskPolyValueTensor[gammaIdx,kIdx,1,0]*diskPolyWeights[gammaIdx,kIdx]
                            for gammaIdx in range(gammaMax) for kIdx in range(kMax)) == 0,
                "Forbidden Innerproduct Constraint")
    constraintMatrx = [[0 for j in range(nrOfPoints)] for i in range(nrOfPoints)]
    for mainPointIdx in range(nrOfPoints):
        for secondPointIdx in range(mainPointIdx+1):
            charVector = charMatrix[mainPointIdx]*charMatrix[secondPointIdx]
            constraintMatrx[mainPointIdx][secondPointIdx] = m.addConstr(gp.quicksum(diskPolyValueTensor[gammaIdx, kIdx, mainPointIdx, secondPointIdx] *
                                    diskPolyWeights[gammaIdx, kIdx]
                                    for gammaIdx in range(gammaMax) for kIdx in range(kMax))
                        ==
                        gp.quicksum(setWeights[setIdx]*charVector[setIdx] for setIdx in range(nrOfSets)),
                        f"BPQ  Constraint vectors {mainPointIdx}, {secondPointIdx}")
    m.setObjective(diskPolyWeights.sum(), GRB.MAXIMIZE)
    m.update()
    m.optimize()
    setSum = 0
    setWeightsArray = np.zeros(nrOfSets)
    for i in range(nrOfSets):
        setSum += setWeights[i].x
        setWeightsArray[i] = setWeights[i].x
    diskSum = 0
    diskWeightsArray = np.zeros((gammaMax, kMax))
    for gammaIdx in range(gammaMax):
        for kIdx in range(kMax):
            diskSum += diskPolyWeights[gammaIdx,kIdx].x
            diskWeightsArray[gammaIdx,kIdx] = diskPolyWeights[gammaIdx,kIdx].x
    # print(diskSum)
    # print(diskSum**2)
    # print((1/2)**(complexDimension-1))
    # print("Done")
    adjustedDWA = diskWeightsArray/diskSum
    return m, diskWeightsArray, setWeightsArray, adjustedDWA


def modelMultipleBQP(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1):
    alpha = complexDimension - 2
    gammaMax = int(gammaMax + 1)
    kMax = int(kMax + 1)
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)
    if listOfPointSets is None:
        listOfPointSets = []
        for pointsSetIdx in range(nrOfPointSets):
            listOfPointSets.append(SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints,
                                                                 baseZ=forbiddenZ)[0])
    else:
        nrOfPointSets = len(listOfPointSets)
        # nrOfPoints = listOfPointSets[0].shape[0]
    charMatricesList = []
    setSizeList = []
    nrOfPointsList = []
    IPMList = []
    diskPolyValueTensorList = []
    diskPolysByGamma = [Disk(complexDimension-2, curGamma, kMax-1) for curGamma in range(gammaMax)]

    for pointSetIdx in range(nrOfPointSets):
        pointSet = listOfPointSets[pointSetIdx]
        nrOfPoints = pointSet.shape[0]
        charMatrix = characteristicMatrix(nrOfPoints)
        nrOfSets = charMatrix.shape[1]

        innerProductMatrix = pointSet @ pointSet.conj().T
        ipmRads, ipmAngles = z2polar(innerProductMatrix)
        diskPolyValueTensor = np.zeros((gammaMax, kMax, nrOfPoints, nrOfPoints))
        for gamma in range(gammaMax):
            diskPolyValueTensor[gamma] = diskPolysByGamma[gamma].calcAtRTheta(ipmRads, ipmAngles)
            # for k in range(kMax):
            #     # diskPolyValueTensor[gamma][k] = vectorizeRDP(alpha, gamma, k, innerProductMatrix)
            #     # diskPolyValueTensor[gamma][k] = (createDiskPolyCosineless(complexDimension, k, gamma)(ipmRads) *
            #     #                                  np.cos(ipmAngles * gamma))
            #     diskPolyValueTensor[gamma][k] = (createDiskPolyCosineless(complexDimension, k, gamma)(ipmRads)*
            #                                      np.cos(ipmAngles*gamma))
        charMatricesList.append(charMatrix)
        setSizeList.append(nrOfSets)
        nrOfPointsList.append(nrOfPoints)
        IPMList.append(innerProductMatrix)
        diskPolyValueTensorList.append(diskPolyValueTensor)
    forbiddenInnerproductVals = np.zeros((gammaMax,kMax))
    for gamma in range(gammaMax):
        forbiddenInnerproductVals[gamma] = diskPolysByGamma[gamma].calcAtRTheta(forbiddenRadius, forbiddenAngle)
        # for k in range(kMax):
        #     # diskPolyValueTensor[gamma][k] = vectorizeRDP(alpha, gamma, k, innerProductMatrix)
        #
        #     forbiddenInnerproductVals[gamma][k] = (createDiskPolyCosineless(complexDimension, k, gamma)(forbiddenRadius) *
        #                                      np.cos(forbiddenAngle * gamma))
    m = gp.Model(f"BQP model for forbidden innerproduct {forbiddenZ} in dimension {complexDimension}, "
                 f"with {nrOfPoints} BQP points,"
                 f" gamma checked {gammaMax}, k checked {kMax}")
    m.setParam("OutputFlag", 0)
    m.setParam(GRB.Param.FeasibilityTol, 1e-8)
    # m.setParam(GRB.Param.DisplayInterval, 10)
    diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.CONTINUOUS, lb=0, name="diskpoly weights")
    m.addConstr(diskPolyWeights[0, 0] - diskPolyWeights.sum() * diskPolyWeights.sum() >= 0, "SDP contraint")
    m.addConstr(gp.quicksum(forbiddenInnerproductVals[gammaIdx, kIdx] * diskPolyWeights[gammaIdx, kIdx]
                            for gammaIdx in range(gammaMax) for kIdx in range(kMax)) == 0,
                f"Forbidden Innerproduct Constraint")

    for pointSetIdx in range(nrOfPointSets):

        curNrOfSets = setSizeList[pointSetIdx]
        curCharMatrix = charMatricesList[pointSetIdx]
        curDPVT = diskPolyValueTensorList[pointSetIdx]
        curNrOfPoints = nrOfPointsList[pointSetIdx]

        setWeights = m.addVars(curNrOfSets, vtype=GRB.CONTINUOUS, lb=0, name=f"set weights pointset {pointSetIdx}")

        m.addConstr(setWeights.sum() == 1, name=f"convex description of sets pointset {pointSetIdx}")


        # constraintMatrx = [[0 for j in range(nrOfPoints)] for i in range(nrOfPoints)]
        for mainPointIdx in range(curNrOfPoints):
            for secondPointIdx in range(mainPointIdx+1):
                charVector = curCharMatrix[mainPointIdx]*curCharMatrix[secondPointIdx]

                # constraintMatrx[mainPointIdx][secondPointIdx] =
                m.addConstr(gp.quicksum(curDPVT[gammaIdx, kIdx, mainPointIdx, secondPointIdx] *
                                        diskPolyWeights[gammaIdx, kIdx]
                                        for gammaIdx in range(gammaMax) for kIdx in range(kMax))
                            ==
                            gp.quicksum(setWeights[setIdx]*charVector[setIdx] for setIdx in range(curNrOfSets)),
                            f"BPQ  Constraint vectors {mainPointIdx}, {secondPointIdx}, pointset {pointSetIdx}")
    m.setObjective(diskPolyWeights.sum(), GRB.MAXIMIZE)
    lbFeasibleBool = False
    powerIdx = (complexDimension - 1)
    while not lbFeasibleBool:
        lbConstr = m.addConstr(diskPolyWeights.sum() >= (1 / 2) ** powerIdx,
                                "Lower bound for obj due to double cap to fix weird behaviour")
        m.update()
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            m.remove(lbConstr)
            powerIdx += 1
        else:
            lbFeasibleBool = True
    # setSum = 0
    # setWeightsArray = np.zeros(nrOfSets)
    # for i in range(nrOfSets):
    #     setSum += setWeights[i].x
    #     setWeightsArray[i] = setWeights[i].x
    diskSum = 0
    diskWeightsArray = np.zeros((gammaMax, kMax))
    for gammaIdx in range(gammaMax):
        for kIdx in range(kMax):
            diskSum += diskPolyWeights[gammaIdx,kIdx].x
            diskWeightsArray[gammaIdx,kIdx] = diskPolyWeights[gammaIdx,kIdx].x
    setSizeList = []
    setTotal = 0
    AddedPoints = 1
    for pointset in listOfPointSets:
        curSize = len(pointset)
        setSizeList.append(curSize)
        setTotal += curSize
        AddedPoints += curSize-2
    # print(f"Objective Iteration {AddedPoints}: {diskSum}. (total points {setTotal}, {setSizeList})")
    # print(diskSum**2)
    # print((1/2)**(complexDimension-1))
    # print("Done")
    adjustedDWA = diskWeightsArray/diskSum
    return m, diskWeightsArray, adjustedDWA, diskSum, diskWeightsArray



def finalBQPModel(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1):
    alpha = complexDimension - 2
    gammaMax = int(gammaMax + 1)
    kMax = int(kMax + 1)
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)
    if listOfPointSets is None:
        listOfPointSets = []
        for pointsSetIdx in range(nrOfPointSets):
            listOfPointSets.append(SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints,
                                                                            baseZ=forbiddenZ)[0])
    else:
        nrOfPointSets = len(listOfPointSets)
    charMatricesList = []
    setSizeList = []
    nrOfPointsList = []
    IPMList = []
    diskPolyValueTensorList = []
    diskPolysByGamma = [Disk(alpha, curGamma, kMax-1) for curGamma in range(gammaMax)]

    for pointSetIdx in range(nrOfPointSets):
        pointSet = listOfPointSets[pointSetIdx]
        nrOfPoints = pointSet.shape[0]
        charMatrix = characteristicMatrix(nrOfPoints)
        nrOfSets = charMatrix.shape[1]

        innerProductMatrix = pointSet @ pointSet.conj().T
        ipmRads, ipmAngles = z2polar(innerProductMatrix)
        diskPolyValueTensor = np.zeros((gammaMax, kMax, nrOfPoints, nrOfPoints))
        for gamma in range(gammaMax):
            diskPolyValueTensor[gamma] = diskPolysByGamma[gamma].calcAtRTheta(ipmRads, ipmAngles)
        charMatricesList.append(charMatrix)
        setSizeList.append(nrOfSets)
        nrOfPointsList.append(nrOfPoints)
        IPMList.append(innerProductMatrix)
        diskPolyValueTensorList.append(diskPolyValueTensor)
    forbiddenInnerproductVals = np.zeros((gammaMax,kMax))
    for gamma in range(gammaMax):
        forbiddenInnerproductVals[gamma] = diskPolysByGamma[gamma].calcAtRTheta(forbiddenRadius, forbiddenAngle)
    m = gp.Model(f"BQP model for forbidden innerproduct {forbiddenZ} in dimension {complexDimension}, "
                 f"with {nrOfPoints} BQP points,"
                 f" gamma checked {gammaMax}, k checked {kMax}")
    m.setParam("OutputFlag", 0)
    m.setParam(GRB.Param.FeasibilityTol, 1e-9)
    m.setParam(GRB.Param.BarConvTol, 1e-12)
    m.setParam(GRB.Param.BarQCPConvTol, 1e-12)
    m.setParam(GRB.Param.MIPGap, 1e-12)
    m.setParam(GRB.Param.IntFeasTol, 1e-9)
    m.setParam(GRB.Param.PSDTol, 1e-12)
    m.setParam(GRB.Param.OptimalityTol, 1e-9)
    # m.setParam(GRB.Param.DisplayInterval, 10)
    diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.CONTINUOUS, lb=0, name="diskpoly weights")
    m.addConstr(diskPolyWeights[0, 0] - diskPolyWeights.sum() * diskPolyWeights.sum() >= 0, "SDP contraint")
    m.addConstr(gp.quicksum(forbiddenInnerproductVals[gammaIdx, kIdx] * diskPolyWeights[gammaIdx, kIdx]
                            for gammaIdx in range(gammaMax) for kIdx in range(kMax)) == 0,
                f"Forbidden Innerproduct Constraint")

    for pointSetIdx in range(nrOfPointSets):

        curNrOfSets = setSizeList[pointSetIdx]
        curCharMatrix = charMatricesList[pointSetIdx]
        curDPVT = diskPolyValueTensorList[pointSetIdx]
        curNrOfPoints = nrOfPointsList[pointSetIdx]

        setWeights = m.addVars(curNrOfSets, vtype=GRB.CONTINUOUS, lb=0, name=f"set weights pointset {pointSetIdx}")

        m.addConstr(setWeights.sum() == 1, name=f"convex description of sets pointset {pointSetIdx}")

        for mainPointIdx in range(curNrOfPoints):
            for secondPointIdx in range(mainPointIdx+1):
                charVector = curCharMatrix[mainPointIdx]*curCharMatrix[secondPointIdx]

                m.addConstr(gp.quicksum(curDPVT[gammaIdx, kIdx, mainPointIdx, secondPointIdx] *
                                        diskPolyWeights[gammaIdx, kIdx]
                                        for gammaIdx in range(gammaMax) for kIdx in range(kMax))
                            ==
                            gp.quicksum(setWeights[setIdx]*charVector[setIdx] for setIdx in range(curNrOfSets)),
                            f"BPQ  Constraint vectors {mainPointIdx}, {secondPointIdx}, pointset {pointSetIdx}")
    m.setObjective(diskPolyWeights.sum(), GRB.MAXIMIZE)
    lbFeasibleBool = False
    powerIdx = (complexDimension - 1)
    while not lbFeasibleBool:
        lbConstr = m.addConstr(diskPolyWeights.sum() >= (1 / 2) ** powerIdx,
                                "Lower bound for obj due to double cap to fix weird behaviour")
        m.update()
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            m.remove(lbConstr)
            powerIdx += 1
        else:
            lbFeasibleBool = True
    diskSum = 0
    diskWeightsArray = np.zeros((gammaMax, kMax))
    for gammaIdx in range(gammaMax):
        for kIdx in range(kMax):
            diskSum += diskPolyWeights[gammaIdx,kIdx].x
            diskWeightsArray[gammaIdx,kIdx] = diskPolyWeights[gammaIdx,kIdx].x
    setSizeList = []
    setTotal = 0
    AddedPoints = 1
    for pointset in listOfPointSets:
        curSize = len(pointset)
        setSizeList.append(curSize)
        setTotal += curSize
        AddedPoints += curSize-2
    adjustedDWA = diskWeightsArray/diskSum
    return adjustedDWA, diskSum, diskWeightsArray



def createDiskPolyCosineless(dim, k, gamma):
    doubleAngleR = Polynomial((-1,0,2),[0,1],[0,1])
    rPower = Polynomial([0]*gamma+[1],[0,1],[0,1])
    startingJacobi = scipy.special.jacobi(k,dim-2,gamma)
    JacobiRightVal = np.float64(startingJacobi(1))
    normalizedJacobi = startingJacobi/JacobiRightVal
    normalizedDoubleAngleJacobi = Polynomial(normalizedJacobi.coef[::-1])(doubleAngleR)
    return normalizedDoubleAngleJacobi*rPower


def stripNonSpin(polyTest, testDim, resolutiontest=2000, maxDeg=20):
    if type(polyTest) is Polynomial:
        maxCoef = polyTest(1)
        rSpacetest = np.linspace(0,1,resolutiontest)
        testArray = polyTest(rSpacetest)
    elif type(polyTest) is np.ndarray:
        resolutiontest = polyTest.shape[0]
        maxCoef = polyTest[-1]
        rSpacetest = np.linspace(0, 1, resolutiontest)
        testArray = polyTest
    else:
        print("invalid input")
        return


    weightPolytest = complexWeightPolyCreator(testDim)
    kTest = 0
    totCoef = 0
    coefList = []
    while totCoef < maxCoef -0.000001 and kTest < maxDeg:
        kPoly = createDiskPolyCosineless(testDim, kTest, 0)
        kPolyNormFunc = weightPolytest * kPoly * kPoly
        kPolyNorm = np.sum(kPolyNormFunc(rSpacetest))/resolutiontest
        testKFunc = weightPolytest * kPoly
        testNorm = np.sum(testKFunc(rSpacetest)*testArray)/resolutiontest
        coefK = max(0,testNorm/kPolyNorm)
        coefList.append(coefK)
        totCoef += coefK
        kTest += 1
    return coefList

def c_k(n: int, k: int) -> float:
    """
    Coefficient c_k in the Jacobi expansion of K_S(s):
      K_S(s) = sum_{k>=0} c_k * P_k^{(n-2,0)}(2 s^2 - 1)

    c_k = 2^{-2(n-1)} * ((2k+n-1)/(n-1))^{3/2}
          * [ ((n-1)_k / k!) * 2F1(-k, k+n-1; n; 1/2) ]^2

    Args:
        n: ambient complex dimension (n >= 2)
        k: degree (k >= 0)

    Returns:
        c_k as a Python float
    """
    if n < 2 or k < 0:
        raise ValueError("Require n >= 2 and k >= 0.")

    # Hypergeometric truncates for integer k
    H = hyp2f1(-k, k + n - 1, n, 0.5)

    # Log-safe ((n-1)_k / k!) = Gamma(n-1+k)/Gamma(n-1)/Gamma(k+1)
    lg = loggamma(n - 1 + k) - loggamma(n - 1) - loggamma(k + 1)
    poch_ratio = np.exp(lg)

    pref = 2.0 ** ( -2* (n -1)) * ((2 * k + n - 1) / (n - 1)) ** 1.0
    return float(pref * ((poch_ratio**2) * (H **2) ))


def c_coeffs(n: int, K: int) -> np.ndarray:
    """Vector of [c_0, ..., c_K]."""
    return np.array([c_k(n, k) for k in range(K + 1)])

def alternateStripNonSpin(arrayTest, testDim, radiusArray, weightsArray, maxDeg=20):
    kTest = 0
    resolutiontest = arrayTest.shape[0]
    totCoef = 0
    coefList = []
    while totCoef < 1 - 0.000001 and kTest < maxDeg:
        kPoly = createDiskPolyCosineless(testDim, kTest, 0)
        kPolyNormFunc = kPoly * kPoly
        kPolyNorm = np.sum(kPolyNormFunc(radiusArray)) / resolutiontest
        # testKFunc = weightsArray * kPoly
        testNorm = np.sum(np.multiply(kPoly(radiusArray), arrayTest)) / resolutiontest
        coefK = max(0, testNorm / kPolyNorm)
        coefList.append(coefK)
        totCoef += coefK
        kTest += 1
    return coefList


def polyFromCoefs(coefs, polySet):
    return Polynomial(sum(coefs[curDeg]*polySet[curDeg]
                                        for curDeg in range(len(coefs))).coef,
                                    window=[0,1], domain=[0,1])


def halfOpenRectsToCoefs(rIntervals, thetaIntervals, kMax, gammaMax, complexDim, rRes=200):
    kStop = kMax + 1
    gammaStop = gammaMax + 1
    coefTotal = kStop*gammaStop
    # theta intervals are defined over [0,pi] due to symmetry constraint of a zonal function
    angularMods = np.zeros(gammaStop)
    angularMods[0] = 2*np.sum(thetaIntervals[:,1]-thetaIntervals[:,0])/(2*np.pi)
    for gammaIdx in range(1,gammaStop):
        angularMods[gammaIdx] = 2*np.sum(np.sin(gammaIdx*thetaIntervals[:,1])
                                         -np.sin(gammaIdx*thetaIntervals[:,0]))/(2*np.pi*gammaIdx)
    constantMeasure = lambda r: -(1-r**2)**(complexDim-1)
    intSizes = constantMeasure(rIntervals[:,1])-constantMeasure(rIntervals[:,0])
    constMod = np.sum(intSizes)
    functionMeasure = lambda r: 2*(complexDim-1)*r*(1-r**2)**(complexDim-2)
    radialNumerators = np.zeros((gammaStop,kStop))
    intCoefConstructor = np.ones((gammaStop,kStop))/coefTotal
    diskCombiPolys = DiskCombi(complexDim-2,intCoefConstructor)
    for rIntervalIdx in range(rIntervals.shape[0]):
        # setup integration of the interval by creating the radial points and their weighted measure
        (rStart,rStop) = rIntervals[rIntervalIdx]
        radiusArray = np.linspace(rStart,rStop,rRes,endpoint=False) + (rStop-rStart)/(2*rRes)
        measurePoints = functionMeasure(radiusArray)/rRes*(rStop-rStart)
        measureNumericalInt = np.sum(measurePoints)
        measurePoints *= intSizes[rIntervalIdx]/measureNumericalInt

        # calc values of all relavant polys at the points in the interval
        combiVals = diskCombiPolys.allValues(radiusArray, 0)
        radialNumerators += np.dot(combiVals, measurePoints)[:, :, 0]
    numerators = radialNumerators*angularMods[:,None]
    squareNumerators = np.square(numerators)
    kArray = np.arange(kStop)
    gammaArray = np.arange(gammaStop)
    kMesh, gammaMesh = np.meshgrid(kArray,gammaArray)

    chiFunc = np.where(gammaMesh == 0, 1, 0.5)

    denomNest = (2.0 * kMesh + complexDim + gammaMesh - 1.0)
    c1 = binom(kMesh + complexDim - 2.0, kMesh)
    c2 = binom(kMesh + complexDim + gammaMesh - 2.0, kMesh + gammaMesh)
    denominators = chiFunc * (complexDim-1)/(denomNest*c1*c2)
    results = squareNumerators/denominators
    comparisonDC = c_coeffs(3,21)
    print(np.sum(results))
    print(np.sqrt(results[0,0]))
    return results





def facetInequalityBQPGammaless(dim,startingPointset, startingCoefficients,
                                startingFacet=0, maxDryRun=-1, stopEarly=True, useRadialEstimator=True):

    pointSetSize = startingPointset.shape[0]
    significantCoefs = np.argwhere(startingCoefficients > 1e-4)
    idxOfLastSignificantCoef = np.max(significantCoefs,axis=0)
    # startingPolynomial = polyFromCoefs(startingCoefficients, polynomialSet)
    if useRadialEstimator:
        startingPolynomial = FastDiskCombiEstimator(dim-2,
                                                    startingCoefficients[:idxOfLastSignificantCoef[0]+1,
                                                                            :idxOfLastSignificantCoef[1]+1]
                                                    ,resolution=1000)
        # # WIP, estimator needs to be made that accept cos(\theta\gamma) stuff
        # # HOW TO:
        # # Make radial estimators for each gamma used, then mult by cos(\theta\gamma) then sum results
        # # Result:
        # # Non recursive function for disk polynomials (aka profit)
        # polyToEstimate = DiskCombi(dim - 2, startingCoefficients[:idxOfLastSignificantCoef[0] + 1,
        #                                         :idxOfLastSignificantCoef[1] + 1])
        # startingPolynomial = polyToEstimate
    else:
        startingPolynomial = DiskCombi(dim-2,startingCoefficients[:idxOfLastSignificantCoef[0]+1,
                                             :idxOfLastSignificantCoef[1]+1])
    startingGuess = np.concat([startingPointset.T.real, startingPointset.T.imag])
    flattenedStartingGuess = startingGuess.ravel()
    baseFlattenedStartingGuess = flattenedStartingGuess.copy()
    shapeStartingGuess = startingGuess.shape
    # shapeStartingSet = startingPointset.shape
    # startingInnerproduct = ((flattenedStartingGuess[:pointSetSize*dim]-1j*flattenedStartingGuess[pointSetSize*dim:]).T @
    #                         (flattenedStartingGuess[:pointSetSize*dim]+1j*flattenedStartingGuess[pointSetSize*dim:]))
    # startingInnerproductv2 = (flattenedStartingGuess.reshape(shapeStartingGuess)[:dim] - 1j * flattenedStartingGuess.reshape(shapeStartingGuess)[
    #                                                                             dim:]).T @ (
    #                                    flattenedStartingGuess.reshape(shapeStartingGuess)[: dim] + 1j * flattenedStartingGuess.reshape(shapeStartingGuess)[
    #                                                                                        dim:])
    validIneqs, facetIneqs, betas = facetReader("bqp6.dat")
    recomplexify = lambda S: ((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))
    polyVals = lambda S:startingPolynomial(*(z2polar((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))))

    bestFacet = 0
    sol = startingGuess
    lastImprovement = 0
    facetIdx = startingFacet + 1
    if validIneqs:
        nrOfFacets = betas.shape[0]
        if maxDryRun == -1:
            maxDryRun = nrOfFacets
        # relativeSizes = np.max(np.max(np.abs(facetIneqs),axis=1), axis=1)
        for facetNr in range(nrOfFacets):
            startingLocations = createRandomPointsComplex(dim,6, includeInner=False)[0]
            startingGuess = np.concat([startingLocations.T.real, startingLocations.T.imag])
            flattenedStartingGuess = startingGuess.ravel()
            facetIdx = -nrOfFacets + facetNr + startingFacet
            objective = lambda S: betas[facetIdx] - np.sum(
                np.multiply(facetIneqs[facetIdx],polyVals(S)))
            constraint = lambda S: np.sum(np.square(S.reshape(shapeStartingGuess)),axis=0)-1

            relativeSizeV1 = np.sqrt(np.sum(np.abs(facetIneqs[facetIdx])))
            currentCutOff = -1.5 * (-1/np.log2(nrOfFacets+2)+1/np.log2(facetNr+2))

            succesBase = False
            succesRand = False
            baseObj = 0
            randObj = 0
            baseSol = baseFlattenedStartingGuess
            randSol = flattenedStartingGuess

            t0 = time.perf_counter()
            # randompoints as startingpoints
            try:
                res = scipy.optimize.minimize(
                    objective, flattenedStartingGuess,
                    method='trust-constr',
                    constraints={'type': 'eq', 'fun': constraint},
                    tol=1e-8,
                    options={"maxiter":500}
                )
            except:
                succesRand = False
            else:
                relativeObjective = res.fun / relativeSizeV1  # / relativeSizes[facetIdx]
                if relativeObjective < bestFacet and res.constr_violation < 1e-8:
                    randSol = res.x.reshape(shapeStartingGuess)
                    randObj = relativeObjective
                    succesRand = True
                else:
                    succesRand = False
            t1 = time.perf_counter()
            resImpr = improvedBQPNLP.solve_problem(flattenedStartingGuess,shapeStartingGuess, startingPolynomial, facetIneqs, facetIdx)
            t2 = time.perf_counter()
            print(f"Old: {t1-t0} \n New:{t2-t1} \n Objectives: {res.fun} vs {resImpr.fun}")
            x=1
            # input points as startingpoints
            try:
                res = scipy.optimize.minimize(
                    objective, baseFlattenedStartingGuess,
                    method='trust-constr',
                    constraints={'type': 'eq', 'fun': constraint},
                    tol=1e-8,
                    options={"maxiter":500}
                )
            except:
                succesBase = False
            else:
                relativeObjective = res.fun / relativeSizeV1 # / relativeSizes[facetIdx]
                if relativeObjective < bestFacet and res.constr_violation < 1e-8:
                    baseSol = res.x.reshape(shapeStartingGuess)
                    baseObj = relativeObjective
                    succesBase = True
                else:
                    succesBase = False

            # Find best sol out of starting options
            resObj = baseObj
            resSol = baseSol
            succesRes = succesBase
            if succesRand:
                if randObj < baseObj:
                    resObj = randObj
                    resSol = randSol
                    succesRes = succesRand

            # Update best sol
            if succesRes:
                lastImprovement = 0
                bestFacet = resObj
                sol = resSol
            else:
                lastImprovement += 1

            # Stop if criteria are met
            if bestFacet < 0 and stopEarly:
                if bestFacet < currentCutOff or lastImprovement >= maxDryRun:
                    break

        if bestFacet < 0:
            return True, (sol[:dim]+1j*sol[dim:]).T, (facetIdx+1) % nrOfFacets
        else:
            return False, startingGuess.T, (facetIdx+1) % nrOfFacets



def optimizeSingleFacet(args):
    facetNr, nrOfFacets, startingFacet, shapeStartingGuess, betas, facetIneqs, startingPolynomial, dim = args
    polyVals = lambda S: startingPolynomial(*(z2polar((S.reshape(shapeStartingGuess)[:dim] -
                                                       1j * S.reshape(shapeStartingGuess)[dim:]).T @
                                                      (S.reshape(shapeStartingGuess)[:dim] +
                                                       1j * S.reshape(shapeStartingGuess)[dim:]))))
    facetIdx = -nrOfFacets + facetNr + startingFacet
    startingLocations = createRandomPointsComplex(dim, 6, includeInner=False)[0]
    startingGuess = np.concatenate([startingLocations.T.real, startingLocations.T.imag])
    flattenedStartingGuess = startingGuess.ravel()

    objective = lambda S: betas[facetIdx] - np.sum(np.multiply(facetIneqs[facetIdx], polyVals(S)))
    constraint = lambda S: np.sum(np.square(S.reshape(shapeStartingGuess)), axis=0) - 1

    try:
        res = scipy.optimize.minimize(
            objective, flattenedStartingGuess,
            method='trust-constr',
            constraints={'type': 'eq', 'fun': constraint},
            tol=1e-8,
            options={"maxiter": 50}
        )
        return (facetNr, facetIdx, res.fun, res.x.reshape(shapeStartingGuess), False)
    except Exception:
        return (facetNr, facetIdx, None, None, True)



def parrallelFacetInequalityBQPGammaless(dim,startingPointset, fixedPointsets, startingCoefficients, polynomialSet,
                                startingFacet=0):
    pointSetSize = startingPointset.shape[0]

    # startingPolynomial = polyFromCoefs(startingCoefficients, polynomialSet)
    startingPolynomialv0 = DiskCombi(dim-2,startingCoefficients)
    startingPolynomial = FastRadialEstimator(startingPolynomialv0, 10001)
    startingGuess = np.concat([startingPointset.T.real, startingPointset.T.imag])
    flattenedStartingGuess = startingGuess.ravel()
    shapeStartingGuess = startingGuess.shape
    shapeStartingSet = startingPointset.shape
    startingInnerproduct = ((flattenedStartingGuess[:pointSetSize*dim]-1j*flattenedStartingGuess[pointSetSize*dim:]).T @
                            (flattenedStartingGuess[:pointSetSize*dim]+1j*flattenedStartingGuess[pointSetSize*dim:]))
    startingInnerproductv2 = (flattenedStartingGuess.reshape(shapeStartingGuess)[:dim] - 1j * flattenedStartingGuess.reshape(shapeStartingGuess)[
                                                                                dim:]).T @ (
                                       flattenedStartingGuess.reshape(shapeStartingGuess)[: dim] + 1j * flattenedStartingGuess.reshape(shapeStartingGuess)[
                                                                                           dim:])
    validIneqs, facetIneqs, betas = facetReader("bqp6.dat")
    recomplexify = lambda S: ((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))
    polyVals = lambda S:startingPolynomial(*(z2polar((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))))

    bestFacet = 0
    sol = startingGuess
    lastImprovement = 0
    facetIdx = startingFacet + 1

    if validIneqs:
        nrOfFacets = betas.shape[0]
        args_list = [
            (facetNr, nrOfFacets, startingFacet, shapeStartingGuess, betas, facetIneqs, startingPolynomial, dim)
            for facetNr in range(nrOfFacets)
        ]
        relativeSizes = np.max(np.max(np.abs(facetIneqs),axis=1), axis=1)
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(optimizeSingleFacet, args_list)
        for facetNr, facetIdx, relObj, cursol, failed in sorted(results, key=lambda x: x[0]):
            if failed:
                lastImprovement += 1
            else:
                if relObj < bestFacet:
                    bestFacet = relObj
                    sol = cursol
                    lastImprovement = 0
                else:
                    lastImprovement += 1
        if bestFacet < 0:
            return True, (sol[:dim]+1j*sol[dim:]).T, (facetIdx+1) % nrOfFacets
        else:
            return False, startingGuess.T, (facetIdx+1) % nrOfFacets


def iterativeProbabilisticImprovingBPQ(dim, resolution=201, outputRes=300, maxIter=10, maxDeg=0, relativityScale=-1,
                                       maxSetSize=10, uniformCoordinateWise=False,
                                       reverseWeighted=False, spreadPoints=False, basePoly=None,
                                       forbiddenRad=0, forbiddenTheta=0, sequentialEdges=False,
                                       printBool=False, plotBool=False):

    if maxDeg == 0:
        maxDeg = 4 * (dim - 1)
    if relativityScale == -1:
        relativityScale = (maxDeg-1)/maxDeg
        # relativityScale = 1
    plotRadii = np.linspace(0, 1, resolution, endpoint=True)
    # maxIter = 10
    polyList = [createDiskPolyCosineless(dim, kIdx, 0) for kIdx in range(maxDeg + 1)]
    weightPoly = complexWeightPolyCreator(dim)
    weightInt = integratedWeightPolyCreator(dim) + 1
    heighestWeightPoint = np.sqrt(1 / (2 * (dim - 2) + 1))
    heighestWeight = weightPoly(heighestWeightPoint)
    scaledWeightPoly = weightPoly / heighestWeight
    if basePoly is None:
        cDC = SphereBasics.complexDoubleCapv2(dim, outputRes, resolution)
        radiusSpace = np.linspace(0, 1, outputRes, endpoint=True)
        polyEstCDC = Polynomial.fit(radiusSpace, cDC, 4 * (dim - 1))
        coefsPolyCDC = stripNonSpin(polyEstCDC, dim, 10 * outputRes)
        basePoly = sum(polyList[kIdx] * coefsPolyCDC[kIdx] for kIdx in range(len(coefsPolyCDC)))

    pointSet1 = np.zeros((2, dim), dtype=np.complex128)
    pointSet1[0, 0] = 1
    pointSet1[1, 1] = polar2z(forbiddenRad, -forbiddenTheta)
    pointSet1[1, 1] = np.sqrt(1-forbiddenRad**2)



    _, _,  coefsThetav0, bestObjVal,unscaledcoefsThetav0 = modelMultipleBQP(forbiddenRad,
                                                                            forbiddenTheta, dim, 0, maxDeg,
                                           listOfPointSets=[pointSet1])
    coefsForKv0 = coefsThetav0[0]



    scalingPoly = (1 - relativityScale) * Polynomial([1], window=[0,1],domain=[0,1]) + relativityScale * basePoly
    scalingPolyV2 = 1 - relativityScale * basePoly
    polyV0 = sum(polyList[kIdx] * coefsForKv0[kIdx] for kIdx in range(maxDeg+1))
    # differencePolyv0 = polyV0 - basePoly
    # differenceDensity0 = differencePolyv0*weightPoly
    # differenceIntegral0 = differenceDensity0.integ(lbnd=0)
    # remainingRadius = 1

    # for i in range(2):
    #     differenceSize0 = differenceIntegral0(remainingRadius)
    #     pdf0 = differenceDensity0/differenceSize0
    #     probability = SphereBasics.polyPdf(a=0, b=remainingRadius,
    #                                                 name=f'Probability Distrubution Init')
    #     probabilityPdf = probability(poly=pdf0)
    #     value = probability.rvs()
    #     pointSet1[2,i] = value
    #     remainingRadius = np.sqrt(remainingRadius**2-value**2)
    listOfPointSets = [pointSet1]
    fullPointSets = []
    resultingpolyList = [polyV0]
    coefThetaList = [coefsThetav0]
    unscaledCoefThetaList = [unscaledcoefsThetav0]
    weightPolyList = [complexWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    weightIntsList = [integratedWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    facetStart = 0
    if spreadPoints:
        innerProductList = np.array([0,1])
        # distantPoly = Polynomial([0,0,1,-2,1], window=[0,1], domain=[0,1])
    else:
        innerProductList = []
        # distantPoly = Polynomial([1], window=[0, 1], domain=[0, 1])
    for iterationNr in range(maxIter):
        setSpecificIterNr = iterationNr % (maxSetSize-2)
        coefsPrevTheta = coefThetaList[iterationNr]
        polyPrev = resultingpolyList[iterationNr]
        differencePoly = polyPrev - basePoly
        scaledDifference = differencePoly * scalingPolyV2
        # differenceDensity = scaledDifference * weightPoly
        differenceDensity = scaledDifference * (1-scaledWeightPoly)
        # scaledDifference, scaledRemainder = np.polydiv(differencePoly.coef, scalingPoly.coef)
        # polyScaledDifference = Polynomial(scaledDifference, window=[0,1],domain=[0,1])
        # polyScaledRemainder = Polynomial(scaledRemainder, window=[0,1],domain=[0,1])
        # PolyScaledFactor = polyScaledDifference*basePoly
        if plotBool:
            fig, ax = plt.subplots()
            v1Line, = ax.plot(plotRadii, polyPrev(plotRadii), label="Previous")

            v3Line, = ax.plot(plotRadii, basePoly(plotRadii), label='Base')
            v4Line, = ax.plot(plotRadii, scaledWeightPoly(plotRadii), label='Weight')
            ax.legend(handles=[v1Line,  v3Line, v4Line], loc='upper left',
                      bbox_to_anchor=(0.01, 0.99))
            plt.show()
            fig, ax = plt.subplots()
            v2Line, = ax.plot(plotRadii, differencePoly(plotRadii), label='Difference')
            v5Line, = ax.plot(plotRadii, scaledDifference(plotRadii), label='Scaled Difference')
            v6Line, = ax.plot(plotRadii, differenceDensity(plotRadii), label='Scaled Difference Density')
            # scaledWeightPoly
            ax.legend(handles=[ v2Line, v5Line,v6Line], loc='upper left',
                      bbox_to_anchor=(0.01, 0.99))
            plt.show()

        differenceIntegral = differenceDensity.integ(lbnd=0)
        remainingRadius = 1
        nrOfNextPoints = 3 + setSpecificIterNr
        if setSpecificIterNr == 0 and iterationNr != 0:
            fullPointSets.append(listOfPointSets[iterationNr])
            prevPointSet = listOfPointSets[0]
        else:
            prevPointSet = listOfPointSets[iterationNr]
        pointSetNew = np.zeros((nrOfNextPoints, dim),dtype=np.complex128)
        pointSetNew[:nrOfNextPoints - 1] = prevPointSet
        if uniformCoordinateWise:
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim,dtype=np.complex128)
            relativeSizes = np.random.random(size=dim-1)
            for i in range(min(dim, nrOfNextPoints )):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    coordinateRadius = remainingRadius*relativeSizes[i]
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)
        elif reverseWeighted:
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim, dtype=np.complex128)
            relativeSizes = np.random.random(size=dim - 1)
            for i in range(min(dim, nrOfNextPoints)):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    maxInt = weightInt(remainingRadius)
                    scaledWeightInt = weightInt/maxInt
                    unTransformedFactor = remainingRadius * relativeSizes[i]
                    coordinateRadius = remainingRadius*scaledWeightInt(unTransformedFactor)
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)
        elif spreadPoints:
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim, dtype=np.complex128)
            relativeSizes = np.random.random(size=dim - 1)
            for i in range(min(dim, nrOfNextPoints)):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    pieceWisePolyBreakpoints = np.zeros(2*innerProductList.shape[0]-1)
                    for leftIdx in range(innerProductList.shape[0]):
                        for shiftIdx in range(2):
                            pwpbIdx = 2*leftIdx+shiftIdx
                            if pwpbIdx < pieceWisePolyBreakpoints.shape[0]:
                                pieceWisePolyBreakpoints[pwpbIdx] = (1/2*innerProductList[leftIdx]+
                                                                     1/2*innerProductList[leftIdx+shiftIdx])
                    # spreadInt = distantPoly.integ(lbnd=0)
                    # intHeight = spreadInt(remainingRadius)
                    # scaledAndShiftedSpreadInt = spreadInt / intHeight - relativeSizes[i]
                    # roots = scaledAndShiftedSpreadInt.roots()
                    # realMask = (np.abs(np.imag(roots)) < 0.1**5)
                    # realMaskv2 = (np.imag(roots) == 0)
                    # if np.any(realMask != realMaskv2):
                    #     print(roots[(realMask - realMaskv2)])
                    # positiveRealMask = (np.real(roots) >= 0)
                    # nonEliminated = np.real(roots[positiveRealMask*realMask])
                    # desiredRoot = np.argmin(nonEliminated)
                    # pieceWiseTupleList = []
                    # x = sym.symbols('x', positive=True)
                    # for piecewiseIdx in range(pieceWisePolyBreakpoints.shape[0]):
                    #     if piecewiseIdx % 2 == 0:
                    #         pieceWiseTupleList.append((x-pieceWisePolyBreakpoints[piecewiseIdx],
                    #                                    x <= pieceWisePolyBreakpoints[piecewiseIdx+1]))
                    #     else:
                    #         pieceWiseTupleList.append((-x + pieceWisePolyBreakpoints[piecewiseIdx+ 1],
                    #                                    x <= pieceWisePolyBreakpoints[piecewiseIdx + 1]))
                    # pieceWiseFunction = sym.Piecewise(*pieceWiseTupleList)
                    #
                    sizesOfParts = pieceWisePolyBreakpoints[:-1]-pieceWisePolyBreakpoints[1:]
                    intsOfParts = np.square(sizesOfParts)
                    radiusIdx = np.searchsorted(pieceWisePolyBreakpoints, remainingRadius)
                    if radiusIdx == intsOfParts.shape[0]:
                        radiusIdx -= 1
                        cumsumOfInts = np.cumsum(intsOfParts)
                        totalOfInt = np.sum(cumsumOfInts[- 1])
                    else:
                        if radiusIdx % 2 == 1:
                            intRadContribution = np.square(remainingRadius-pieceWisePolyBreakpoints[radiusIdx-1])
                        else:
                            radSquaredDistance = np.square(-remainingRadius+pieceWisePolyBreakpoints[radiusIdx])
                            intRadContribution = intsOfParts[radiusIdx] - radSquaredDistance
                        cumsumOfInts = np.cumsum(intsOfParts)
                        totalOfInt = np.sum(cumsumOfInts[radiusIdx-1])+intRadContribution
                    relativeOfInt = totalOfInt*relativeSizes[i]
                    newIdx = np.searchsorted(cumsumOfInts, relativeOfInt)
                    if newIdx % 2 == 1:
                        coordinateRadius = (np.sqrt(relativeOfInt-cumsumOfInts[newIdx-1])
                                            + pieceWisePolyBreakpoints[newIdx-1])
                    else:
                        coordinateRadius = -(np.sqrt(-relativeOfInt+cumsumOfInts[newIdx])
                                            + pieceWisePolyBreakpoints[newIdx])
                    # coordinateRadius = nonEliminated[desiredRoot]
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                insertIdx = np.searchsorted(innerProductList, coordinateRadius)

                # Insert x at the correct index
                innerProductList = np.insert(innerProductList, insertIdx, coordinateRadius)
                # distantPoly *= Polynomial([coordinateRadius**2,-2*coordinateRadius,1], window=[0,1],domain=[0,1])
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)
        elif sequentialEdges:
            prevPoint = pointSetNew[nrOfNextPoints - 1]
            basis = onb_with_vector(prevPoint)


            newPointCoefRads = np.zeros_like(prevPoint)
            newPointCoefRads[0] = forbiddenRad
            remainingRadius = np.sqrt(1 - forbiddenRad ** 2)
            coordinateAngleCoefs = st.uniform.rvs(scale=2 * math.pi, size=dim)
            coordinateAngleCoefs[0] = -forbiddenTheta
            surfaceMeasure = lambda r, decr: 1-(1 - r ** 2) ** (dim - 2 - decr)
            radialRealizations = st.uniform.rvs(size=dim-2)
            for radIdx in range(dim-2):
                realization = radialRealizations[radIdx]
                normalizationFactor = surfaceMeasure(remainingRadius,radIdx)
                radialCoef = np.power(1-np.power(1-realization*normalizationFactor,1/(dim-2-radIdx)),1/2)
                newPointCoefRads[radIdx+1] = radialCoef
                remainingRadius = np.sqrt(remainingRadius**2-radialCoef**2)
                if remainingRadius <= 0:
                    break
            newPointCoefRads[-1] = remainingRadius
            newPointComplexCoefs = polar2z(newPointCoefRads, coordinateAngleCoefs)
            newPoint = np.dot(basis, newPointComplexCoefs)

        else:
            FirstCoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=2)
            overUnderRest = np.random.randint(0, 2, size=min(dim, nrOfNextPoints - 1))

            newPoint = np.zeros(dim,dtype=np.complex128)
            for i in range(min(dim, nrOfNextPoints - 1)):
                if remainingRadius <= 0:
                    break
                if i <= 1:
                    differenceSizeIdx = differenceIntegral(remainingRadius)
                    pdfIdx = differenceDensity / differenceSizeIdx
                    probability = SphereBasics.polyPdf(a=0, b=remainingRadius, poly=pdfIdx,
                                                       name=f'Probability Distrubution Init')
                    radiusCur = probability.rvs()
                    coordinate = radiusCur * np.exp(1j * FirstCoordinateAngles[i])
                else:
                    # use i'th point as w
                    comparisonPoint = pointSetNew[i]
                    # preditermined coordinates have inner product p(v,w) below
                    innerProductWithPrev = np.vdot(comparisonPoint[:i], newPoint[:i])
                    absIPWP = np.abs(innerProductWithPrev)
                    # w_i
                    PrevLastCoordinateRadius = np.abs(comparisonPoint[i])
                    # |<v,w>| in [|p(v,w)| - |v_max w_i|, |p(v,w)| - |v_max w_i|]
                    lbIP = max(0, absIPWP - PrevLastCoordinateRadius * remainingRadius)
                    ubIP = min(1, absIPWP + PrevLastCoordinateRadius * remainingRadius)
                    # adjust probability distribution within range of possibilities
                    differenceSizeIdx = differenceIntegral(ubIP) - differenceIntegral(lbIP)
                    pdfIdx = differenceDensity / differenceSizeIdx
                    probability = SphereBasics.polyPdf(a=lbIP, b=ubIP, poly=pdfIdx,
                                                       name=f'Probability Distrubution innerproduct')
                    desiredInnerproduct = probability.rvs()
                    # |<v,w>| must be adjusted by v_i*w_i with the number below
                    desiredLastCoordinateFactor = desiredInnerproduct - absIPWP
                    if i == dim - 1:
                        # cannot funnel remaining radius to another coordinate
                        radiusCur = remainingRadius
                    else:
                        # |v_i| must be in [t/w_i,v_max]
                        radiusLB = np.abs(desiredLastCoordinateFactor) / PrevLastCoordinateRadius
                        # choose using weighted distrubution what part of the radius stays on this coordinate
                        curWeightPoly = weightPolyList[i]
                        curIntPoly = weightIntsList[i]
                        curPossibilitySize = curIntPoly(remainingRadius) - curIntPoly(radiusLB)
                        curWeightPoly /= curPossibilitySize
                        radiusProbability = SphereBasics.polyPdf(a=radiusLB, b=remainingRadius, poly=curWeightPoly,
                                                                 name=f'radiusPart distribution')
                        radiusCur = radiusProbability.rvs()
                    # with |v_i| = radiusCur, a relative phase is found to get desired innerproduct,
                    # randomly choosing for [0,pi] or [-pi,0]
                    relativeAngle = (2 * overUnderRest[i] - 1) * np.arccos(
                        (desiredLastCoordinateFactor ** 2 -
                         (PrevLastCoordinateRadius * radiusCur) ** 2)
                        / (2 * (PrevLastCoordinateRadius * radiusCur)))
                    # the true phase is found by multiplying the relative phase with the phase of the p(v,w)
                    coordinate = radiusCur * np.exp(1j * relativeAngle) * (innerProductWithPrev / absIPWP)
                    # coordinate = radiusCur*np.exp(1j*coordinateAngles[i])
                newPoint[i] = coordinate
                remainingRadius = np.sqrt(remainingRadius ** 2 - radiusCur ** 2)
            if nrOfNextPoints <= dim:
                newPoint[nrOfNextPoints - 1] = remainingRadius
        pointSetNew[nrOfNextPoints - 1] = newPoint
        listOfPointSets.append(pointSetNew)
        newInnerproducts = pointSetNew @ pointSetNew.conj().T
        newRadii, _ = z2polar(newInnerproducts)
        for i in range(newRadii.shape[0]):
            if newRadii[i][i] < 1-0.000001 or newRadii[i][i] > 1+0.000001:
                print(f"Radius error: {newRadii[i][i]}")
        # plt.hist(newRadii.flatten(),bins=np.linspace(0,1,21,endpoint=True))
        # plt.show()
        # print(newRadii)
        (_, _, coefsCurTheta, curObjVal,
         unscaledCoefsCurTheta) = modelMultipleBQP(0, 0, dim, 0, maxDeg,
                                                   listOfPointSets=[listOfPointSets[iterationNr+1]]+fullPointSets)
        if listOfPointSets[iterationNr+1].shape[0] == 6:
            foundBool, ineqPointset, facetStart = facetInequalityBQPGammaless(dim, listOfPointSets[iterationNr+1],
                                                                              fullPointSets,
                                                                                unscaledCoefsCurTheta,polyList,
                                                                              startingFacet=facetStart)
            if foundBool:
                (_,_,coefsFacetTheta, facetObjVal,
                 unscaledCoefsFacetTheta) = modelMultipleBQP(0, 0, dim, 0,
                                                                maxDeg,
                                                             listOfPointSets=[ineqPointset] + fullPointSets)


                print(f"facet ineqs gave us {facetObjVal}, instead of {curObjVal}")
                if facetObjVal < curObjVal:
                    curObjVal = facetObjVal
                    listOfPointSets[iterationNr+1] = ineqPointset
                    coefsCurTheta = coefsFacetTheta
                    unscaledCoefsCurTheta = unscaledCoefsFacetTheta
                else:
                    print(f"facets did not improve the solution ({facetObjVal} new, {curObjVal} old)")
                if len(fullPointSets)>0:
                    inputPointSets = [ineqPointset] + fullPointSets
                    combinedPointSets = [np.concat([inputPointSets[PSidx], inputPointSets[PS2idx]])
                                         for PSidx in range(len(inputPointSets)-1)
                                         for PS2idx in range(PSidx+1, len(inputPointSets))]
                    (_, _, coefsFacetThetav2, facetObjValv2,
                     unscaledCoefsFacetThetaV2) = modelMultipleBQP(0, 0, dim, 0,
                                                                             maxDeg, listOfPointSets=combinedPointSets)
                    print(f"Combined facet ineqs gave us {facetObjValv2}, instead of {curObjVal}")
                    if facetObjValv2 < curObjVal:
                        curObjVal = facetObjValv2
                        listOfPointSets[iterationNr + 1] = ineqPointset
                        coefsCurTheta = coefsFacetThetav2
                        unscaledCoefsCurTheta = unscaledCoefsFacetTheta
                    else:
                        print(f"facets did not improve the solution ({facetObjVal} new, {curObjVal} old)")
            else:
                print(f"did not find good facet (cur sol value {curObjVal})")

        # for kIdx in range(maxDeg+1):
        #     if coefsCurTheta[0][kIdx] < 0.0001:
        #         coefsCurTheta[0][kIdx] = 0
        if curObjVal < bestObjVal:
            if printBool:
                if curObjVal < bestObjVal-0.01 and curObjVal<0.29:
                    print(f"current objective value: {curObjVal}")
                    print("Large Improvement with new Innerproducts:")
                    print(newInnerproducts)
        bestObjVal = curObjVal
        coefThetaList.append(coefsCurTheta)
        unscaledCoefThetaList.append(unscaledCoefsCurTheta)
        polyVcur = sum(polyList[kIdx] * coefsCurTheta[0][kIdx] for kIdx in range(maxDeg+1))
        resultingpolyList.append(polyVcur)
        # if bestObjVal < (1/2)**(dim-1):
        #     print("objValError")
    inputParameters = {"dim":dim, "resolution":resolution, "outputRes":outputRes, "maxIter":maxIter,
                       "maxDeg":maxDeg, "relativityScale":relativityScale, "maxSetSize":maxSetSize,
                       "uniformCoordinateWise":uniformCoordinateWise, "reverseWeighted":reverseWeighted,
                       "spreadPoints":spreadPoints, "basePoly":basePoly.coef.tolist()}
    usedPointSets = fullPointSets + [listOfPointSets[-1]]
    # usedPointSetLists = [pointset.tolist() for pointset in usedPointSets]
    usedPointSetDict = {f"point set {setNr}": usedPointSets[setNr].tolist() for setNr in range(len(usedPointSets))}
    coefsThetaListsList = [coefArray.tolist() for coefArray in coefThetaList]
    unscaledCoefsThetaListsList = [coefArray.tolist() for coefArray in unscaledCoefThetaList]
    outputDict = {"objective":bestObjVal, "coeficients disk polynomials list":coefsThetaListsList,
                  "unscaled coeficients":unscaledCoefsThetaListsList,
                   "used point sets":usedPointSetDict,
                  "input parameters":inputParameters} # "resulting polynomials list":resultingpolyList,
    return outputDict



def finalBQPMethod(dim, maxDeg=0, maxGamma=0, forbiddenRad=0, forbiddenTheta=0,
                    sizeOfBQPSet=10, setAmount=10, setLinks=1,
                    sequentialEdges=False, uniformCoordinateWise=False, reverseWeighted=False, spreadPoints=False,
                    compareToLowerBound=False, trulyUniformOnSphere=False, improvementWithFacets = False,
                    relativityScale=-1, basePoly=None,
                    printBool=False):
    # Make sure all correct information is available
    if maxDeg == 0:
        maxDeg = 4 * (dim - 1)
    if reverseWeighted:
        if relativityScale == -1:
            relativityScale = (maxDeg - 1) / maxDeg
    if spreadPoints:
        innerProductList = np.array([0, 1])
    else:
        innerProductList = []

    # Create radial measures and their densities for each dimension
    weightPolyList = [complexWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    weightIntsList = [integratedWeightPolyCreator(dim - depth) for depth in range(dim - 1)]



    # Start making point sets for simplest set with inner product t
    pointSet1 = np.zeros((2, dim), dtype=np.complex128)
    pointSet1[0, 0] = 1
    pointSet1[1, 1] = polar2z(forbiddenRad, -forbiddenTheta)
    pointSet1[1, 1] = np.sqrt(1 - forbiddenRad ** 2)

    # Run once to get a baseline
    coefsThetav0, bestObjVal, unscaledcoefsThetav0 = finalBQPModel(forbiddenRad,
                                                                            forbiddenTheta, dim, maxGamma, maxDeg,
                                                                            listOfPointSets=[pointSet1])
    # Disk poly from baseline
    diskPolyBQPLess = DiskCombi(dim-2,unscaledcoefsThetav0)

    # Initialize the lists for BQP sets, disk polynomials and their coefficients in both normalizations
    listOfPointSets = [pointSet1]
    fullPointSets = []
    resultingpolyList = [diskPolyBQPLess]
    coefThetaList = [coefsThetav0]
    unscaledCoefThetaList = [unscaledcoefsThetav0]

    # Set counters to start
    facetStart = 0
    setIdx = 0
    pointsInCurSet = 2
    iterationNr = 0

    while setIdx < setAmount:
        # Preparation for next point and set
        remainingRadius = 1
        nrOfNextPoints = pointsInCurSet + 1
        if pointsInCurSet == 2 and setIdx != 0:
            fullPointSets.append(listOfPointSets[iterationNr])
            prevPointSet = listOfPointSets[0]
        else:
            prevPointSet = listOfPointSets[iterationNr]
        pointSetNew = np.zeros((nrOfNextPoints, dim), dtype=np.complex128)
        pointSetNew[:nrOfNextPoints - 1] = prevPointSet

        # straight up: random argument, magnitude of coordinate is realization * remaining radius
        if uniformCoordinateWise:
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim, dtype=np.complex128)
            relativeSizes = np.random.random(size=dim - 1)
            for i in range(min(dim, nrOfNextPoints)):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    coordinateRadius = remainingRadius * relativeSizes[i]
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)

        # with radius r, realization R and remaining radius b, then r = b*((1-(b*R)^2)/(1-b^2))^(n-1)
        # This creates a distribution that errs on the side of the middle a lot
        elif reverseWeighted:
            weightInt = weightIntsList[0]
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim, dtype=np.complex128)
            relativeSizes = np.random.random(size=dim - 1)
            for i in range(min(dim, nrOfNextPoints)):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    maxInt = weightInt(remainingRadius)
                    scaledWeightInt = weightInt / maxInt
                    unTransformedFactor = remainingRadius * relativeSizes[i]
                    coordinateRadius = remainingRadius * scaledWeightInt(unTransformedFactor)
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)

        # density the probability is the squared error of a possible inner product and
        # already existing the inner products.
        # This should produce a diverse set of inner products
        elif spreadPoints:
            CoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=dim)
            newPoint = np.zeros(dim, dtype=np.complex128)
            relativeSizes = np.random.random(size=dim - 1)
            for i in range(min(dim, nrOfNextPoints)):
                if remainingRadius <= 0:
                    break
                if i + 1 < min(dim, nrOfNextPoints):
                    pieceWisePolyBreakpoints = np.zeros(2 * innerProductList.shape[0] - 1)
                    for leftIdx in range(innerProductList.shape[0]):
                        for shiftIdx in range(2):
                            pwpbIdx = 2 * leftIdx + shiftIdx
                            if pwpbIdx < pieceWisePolyBreakpoints.shape[0]:
                                pieceWisePolyBreakpoints[pwpbIdx] = (1 / 2 * innerProductList[leftIdx] +
                                                                     1 / 2 * innerProductList[leftIdx + shiftIdx])
                    sizesOfParts = pieceWisePolyBreakpoints[:-1] - pieceWisePolyBreakpoints[1:]
                    intsOfParts = np.square(sizesOfParts)
                    radiusIdx = np.searchsorted(pieceWisePolyBreakpoints, remainingRadius)
                    if radiusIdx == intsOfParts.shape[0]:
                        radiusIdx -= 1
                        cumsumOfInts = np.cumsum(intsOfParts)
                        totalOfInt = np.sum(cumsumOfInts[- 1])
                    else:
                        if radiusIdx % 2 == 1:
                            intRadContribution = np.square(
                                remainingRadius - pieceWisePolyBreakpoints[radiusIdx - 1])
                        else:
                            radSquaredDistance = np.square(-remainingRadius + pieceWisePolyBreakpoints[radiusIdx])
                            intRadContribution = intsOfParts[radiusIdx] - radSquaredDistance
                        cumsumOfInts = np.cumsum(intsOfParts)
                        totalOfInt = np.sum(cumsumOfInts[radiusIdx - 1]) + intRadContribution
                    relativeOfInt = totalOfInt * relativeSizes[i]
                    newIdx = np.searchsorted(cumsumOfInts, relativeOfInt)
                    if newIdx % 2 == 1:
                        coordinateRadius = (np.sqrt(relativeOfInt - cumsumOfInts[newIdx - 1])
                                            + pieceWisePolyBreakpoints[newIdx - 1])
                    else:
                        coordinateRadius = -(np.sqrt(-relativeOfInt + cumsumOfInts[newIdx])
                                             + pieceWisePolyBreakpoints[newIdx])
                else:
                    coordinateRadius = remainingRadius
                coordinateValue = coordinateRadius * np.exp(1j * CoordinateAngles[i])
                newPoint[i] = coordinateValue
                insertIdx = np.searchsorted(innerProductList, coordinateRadius)

                # Insert x at the correct index
                innerProductList = np.insert(innerProductList, insertIdx, coordinateRadius)
                remainingRadius = np.sqrt(remainingRadius ** 2 - coordinateRadius ** 2)

        # Spreads points equally over the sphere
        # under the constraint that they must have inner product t with the previous point.
        elif sequentialEdges:
            prevPoint = pointSetNew[nrOfNextPoints - 2]
            basis = onb_with_vector(prevPoint)

            newPointCoefRads = np.zeros_like(prevPoint)
            newPointCoefRads[0] = forbiddenRad
            remainingRadius = np.sqrt(1 - forbiddenRad ** 2)
            coordinateAngleCoefs = st.uniform.rvs(scale=2 * math.pi, size=dim)
            coordinateAngleCoefs[0] = -forbiddenTheta
            surfaceMeasure = lambda r, decr: 1 - (1 - r ** 2) ** (dim - 2 - decr)
            radialRealizations = st.uniform.rvs(size=dim - 2)
            for radIdx in range(dim - 2):
                realization = radialRealizations[radIdx]
                normalizationFactor = surfaceMeasure(remainingRadius, radIdx)
                radialCoef = np.power(1 - np.power(1 - realization * normalizationFactor, 1 / (dim - 2 - radIdx)),
                                      1 / 2)
                newPointCoefRads[radIdx + 1] = radialCoef
                remainingRadius = np.sqrt(remainingRadius ** 2 - radialCoef ** 2)
                if remainingRadius <= 0:
                    break
            newPointCoefRads[-1] = remainingRadius
            newPointComplexCoefs = polar2z(newPointCoefRads, coordinateAngleCoefs)
            newPoint = np.dot(basis, newPointComplexCoefs)

        # If the coefficients for the lower bound have been provided, it subtracts the coefficients and
        # places inner products where the functions differ a lot
        elif compareToLowerBound:
            # WIP TO BE FIXED
            differenceIntegral = lambda r: r**2
            differenceDensity = lambda r: 2 * r
            FirstCoordinateAngles = st.uniform.rvs(scale=2 * math.pi, size=2)
            overUnderRest = np.random.randint(0, 2, size=min(dim, nrOfNextPoints - 1))

            newPoint = np.zeros(dim, dtype=np.complex128)
            for i in range(min(dim, nrOfNextPoints - 1)):
                if remainingRadius <= 0:
                    break
                if i <= 1:
                    differenceSizeIdx = differenceIntegral(remainingRadius)
                    pdfIdx = differenceDensity / differenceSizeIdx
                    probability = SphereBasics.polyPdf(a=0, b=remainingRadius, poly=pdfIdx,
                                                       name=f'Probability Distrubution Init')
                    radiusCur = probability.rvs()
                    coordinate = radiusCur * np.exp(1j * FirstCoordinateAngles[i])
                else:
                    # use i'th point as w
                    comparisonPoint = pointSetNew[i]
                    # preditermined coordinates have inner product p(v,w) below
                    innerProductWithPrev = np.vdot(comparisonPoint[:i], newPoint[:i])
                    absIPWP = np.abs(innerProductWithPrev)
                    # w_i
                    PrevLastCoordinateRadius = np.abs(comparisonPoint[i])
                    # |<v,w>| in [|p(v,w)| - |v_max w_i|, |p(v,w)| - |v_max w_i|]
                    lbIP = max(0, absIPWP - PrevLastCoordinateRadius * remainingRadius)
                    ubIP = min(1, absIPWP + PrevLastCoordinateRadius * remainingRadius)
                    # adjust probability distribution within range of possibilities
                    differenceSizeIdx = differenceIntegral(ubIP) - differenceIntegral(lbIP)
                    pdfIdx = differenceDensity / differenceSizeIdx
                    probability = SphereBasics.polyPdf(a=lbIP, b=ubIP, poly=pdfIdx,
                                                       name=f'Probability Distrubution innerproduct')
                    desiredInnerproduct = probability.rvs()
                    # |<v,w>| must be adjusted by v_i*w_i with the number below
                    desiredLastCoordinateFactor = desiredInnerproduct - absIPWP
                    if i == dim - 1:
                        # cannot funnel remaining radius to another coordinate
                        radiusCur = remainingRadius
                    else:
                        # |v_i| must be in [t/w_i,v_max]
                        radiusLB = np.abs(desiredLastCoordinateFactor) / PrevLastCoordinateRadius
                        # choose using weighted distrubution what part of the radius stays on this coordinate
                        curWeightPoly = weightPolyList[i]
                        curIntPoly = weightIntsList[i]
                        curPossibilitySize = curIntPoly(remainingRadius) - curIntPoly(radiusLB)
                        curWeightPoly /= curPossibilitySize
                        radiusProbability = SphereBasics.polyPdf(a=radiusLB, b=remainingRadius, poly=curWeightPoly,
                                                                 name=f'radiusPart distribution')
                        radiusCur = radiusProbability.rvs()
                    # with |v_i| = radiusCur, a relative phase is found to get desired innerproduct,
                    # randomly choosing for [0,pi] or [-pi,0]
                    relativeAngle = (2 * overUnderRest[i] - 1) * np.arccos(
                        (desiredLastCoordinateFactor ** 2 -
                         (PrevLastCoordinateRadius * radiusCur) ** 2)
                        / (2 * (PrevLastCoordinateRadius * radiusCur)))
                    # the true phase is found by multiplying the relative phase with the phase of the p(v,w)
                    coordinate = radiusCur * np.exp(1j * relativeAngle) * (innerProductWithPrev / absIPWP)
                    # coordinate = radiusCur*np.exp(1j*coordinateAngles[i])
                newPoint[i] = coordinate
                remainingRadius = np.sqrt(remainingRadius ** 2 - radiusCur ** 2)
            if nrOfNextPoints <= dim:
                newPoint[nrOfNextPoints - 1] = remainingRadius

        # Chooses a point according to the surface measure
        elif trulyUniformOnSphere:
            newPointCoefRads = np.zeros(dim)
            coordinateAngleCoefs = st.uniform.rvs(scale=2 * math.pi, size=dim)
            surfaceMeasure = lambda r, decr: 1 - (1 - r ** 2) ** (dim - 2 - decr)
            radialRealizations = st.uniform.rvs(size=dim - 1)
            for radIdx in range(dim - 2):
                realization = radialRealizations[radIdx]
                normalizationFactor = surfaceMeasure(remainingRadius, radIdx)
                radialCoef = np.power(1 - np.power(1 - realization * normalizationFactor, 1 / (dim - 2 - radIdx)),
                                      1 / 2)
                newPointCoefRads[radIdx + 1] = radialCoef
                remainingRadius = np.sqrt(remainingRadius ** 2 - radialCoef ** 2)
                if remainingRadius <= 0:
                    break
            newPointCoefRads[-1] = remainingRadius
            newPoint = polar2z(newPointCoefRads, coordinateAngleCoefs)

        # Failsafe
        else:
            newPoint = np.ones(dim)/np.sqrt(dim)
        print("point made, running problem")
        # Add point to pointset
        pointSetNew[nrOfNextPoints - 1] = newPoint

        # Process pointsets
        listOfPointSets.append(pointSetNew)
        newInnerproducts = pointSetNew @ pointSetNew.conj().T
        newRadii, _ = z2polar(newInnerproducts)
        for i in range(newRadii.shape[0]):
            if newRadii[i][i] < 1 - 0.000001 or newRadii[i][i] > 1 + 0.000001:
                print(f"Radius error: {newRadii[i][i]}")

        # Append sets together according to setLinks
        inputPointSets = [listOfPointSets[iterationNr + 1]] + fullPointSets
        if setLinks == 1:
            pointSetsForModel = inputPointSets
        elif len(fullPointSets) < setLinks:
            if improvementWithFacets:
                pointSetsForModel = [np.concat(inputPointSets)]
            else:
                pointSetsForModel = [np.concat([inputPointSets[0][:2]]+[inputPointSets[setIdx][2:]
                                                                        for setIdx in range(len(inputPointSets))])]
        else:
            pointSetsForModel = []
            if improvementWithFacets:
                for comb in combinations(range(len(inputPointSets)), setLinks):
                    pointSetsForModel.append(np.concat([inputPointSets[combIdx] for combIdx in comb]))
            else:
                for comb in combinations(range(len(inputPointSets)), setLinks):
                    pointSetsForModel.append(np.concat([inputPointSets[0][:2]]+
                                                       [inputPointSets[combIdx][2:] for combIdx in comb]))
        # Run model with new BQP sets
        (coefsCurTheta, curObjVal,
         unscaledCoefsCurTheta) = finalBQPModel(forbiddenRad, forbiddenTheta, dim,
                                                maxGamma, maxDeg,
                                                listOfPointSets=pointSetsForModel)
        # If the set is full and we want to improve with NLP, then improve the pointset and run it again to verify
        # the program found an improvement
        if improvementWithFacets:
            if listOfPointSets[iterationNr + 1].shape[0] == 6:
                print(f"Before face optimization obj {curObjVal}")
                (foundBool, ineqPointset,
                 facetStart) = facetInequalityBQPGammaless(dim, listOfPointSets[iterationNr + 1],
                                                                                  unscaledCoefsCurTheta,
                                                                                  startingFacet=facetStart)
                if foundBool:
                    inputPointSets = [ineqPointset] + fullPointSets
                    if setLinks == 1:
                        pointSetsForModel = inputPointSets
                    elif len(fullPointSets) < setLinks:
                        pointSetsForModel = [np.concat(inputPointSets)]
                    else:
                        pointSetsForModel = []
                        for comb in combinations(range(len(inputPointSets)), setLinks):
                            pointSetsForModel.append(np.concat([inputPointSets[combIdx] for combIdx in comb]))
                    (coefsFacetTheta, facetObjVal,
                     unscaledCoefsFacetTheta) = finalBQPModel(forbiddenRad, forbiddenTheta,
                                                                dim, maxGamma, maxDeg,
                                                                listOfPointSets=pointSetsForModel)

                    print(f"facet ineqs gave us {facetObjVal}, instead of {curObjVal}")
                    if facetObjVal < curObjVal:
                        curObjVal = facetObjVal
                        listOfPointSets[iterationNr + 1] = ineqPointset
                        coefsCurTheta = coefsFacetTheta
                        unscaledCoefsCurTheta = unscaledCoefsFacetTheta
                    else:
                        print(f"facets did not improve the solution ({facetObjVal} new, {curObjVal} old)")
                else:
                    print(f"did not find good facet (cur sol value {curObjVal})")

        if curObjVal < bestObjVal:
            if printBool:
                if curObjVal < bestObjVal - 0.01 and curObjVal < 0.29:
                    print(f"current objective value: {curObjVal}")
                    print("Large Improvement with new Innerproducts:")
                    print(newInnerproducts)
        bestObjVal = curObjVal
        coefThetaList.append(coefsCurTheta)
        unscaledCoefThetaList.append(unscaledCoefsCurTheta)
        # polyVcur = sum(polyList[kIdx] * coefsCurTheta[0][kIdx] for kIdx in range(maxDeg + 1))
        # resultingpolyList.append(polyVcur)

        # Counter incrementation
        iterationNr += 1
        print(f"Made set {setIdx} point {nrOfNextPoints} (obj {bestObjVal})")
        if nrOfNextPoints == sizeOfBQPSet:
            setIdx += 1
            pointsInCurSet = 2
        else:
            pointsInCurSet = nrOfNextPoints



    inputParameters = {"dim": dim, "maxDeg": maxDeg, "maxGamma": maxGamma,
                       "forbiddenRad":forbiddenRad,"forbiddenTheta":forbiddenTheta,
                       "sizeOfBQPSet":sizeOfBQPSet, "setAmount":setAmount, "setLinks":setLinks,
                       "improvementWithFacets":improvementWithFacets,"sequentialEdges":sequentialEdges,
                       "uniformCoordinateWise": uniformCoordinateWise, "reverseWeighted": reverseWeighted,
                       "spreadPoints": spreadPoints,"compareToLowerBound":compareToLowerBound,
                       "relativityScale": relativityScale}
    usedPointSets = fullPointSets + [listOfPointSets[-1]]
    usedPointSetDict = {f"point set {setNr}": usedPointSets[setNr].tolist() for setNr in range(len(usedPointSets))}
    coefsThetaListsList = [coefArray.tolist() for coefArray in coefThetaList]
    unscaledCoefsThetaListsList = [coefArray.tolist() for coefArray in unscaledCoefThetaList]
    outputDict = {"objective": bestObjVal, "coeficients disk polynomials list": coefsThetaListsList,
                  "unscaled coeficients": unscaledCoefsThetaListsList,
                  "used point sets": usedPointSetDict,
                  "input parameters": inputParameters}  # "resulting polynomials list":resultingpolyList,
    return outputDict




def onb_with_vector(v: np.ndarray) -> np.ndarray:
    """
    Return a unitary U whose first column is the given vector v (complex).
    Columns of U form an orthonormal basis of C^d including v.
    """
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = v.shape[0]
    # Normalize v just in case
    nv = np.linalg.norm(v)
    if nv == 0:
        raise ValueError("v must be nonzero")
    v = v / nv

    e1 = np.zeros(n, dtype=np.complex128)
    e1[0] = 1.0

    v1 = v[0]
    alpha = v1/abs(v1) if abs(v1) > 1e-15 else 1.0 + 0j

    u = v - alpha * e1
    nu = np.linalg.norm(u)

    if nu < 1e-15:
        # v is colinear with e1: unitary is just a diagonal phase on e1
        U = np.eye(n, dtype=np.complex128)
        U[0, 0] = alpha  # first column = alpha * e1 = v
    else:
        u /= nu
        H = np.eye(n, dtype=np.complex128) - 2.0 * np.outer(u, np.conjugate(u))
        U = alpha * H  # unitary, and U @ e1 = v

    # Sanity: columns are ON, first column equals v (up to numerical eps)
    # assert np.allclose(U @ e1, v)
    return U



def oldTestForCDCMethods():
    CDCList = [0.06248928727430177, 0.3748580783515137, 0.2810076317951289, 0, 0.08857453238889738,
               0.0033967362345047915, 0.04125126134937636, 0.004371537515641268, 0.0233650433202186,
               0.004195908603173081, 0.01934098353201613]
    coefficientSize = sum(CDCList)
    CDCList3 = [0.06248928727430177, 0.3748580783515137, 0.2810076317951289, 0, 0.08857453238889738,
                0.0033967362345047915, 0.04125126134937636, 0.004371537515641268, 0.0233650433202186,
                0.004195908603173081, 0.01934098353201613]
    CDCList2 = [i / coefficientSize for i in CDCList]
    fractionsCdc = [fractions.Fraction(i) for i in CDCList3]
    simplifiedFractions = {}
    powerCoefs = {}
    errors = {}
    for power in range(16):
        maxDenom = 2 ** (power + 1)
        scaledCdc = np.array([round(maxDenom * i) for i in fractionsCdc])
        error = np.array([scaledCdc[idx] / maxDenom - CDCList3[idx] for idx in range(len(scaledCdc))])
        simplifiedFractions[maxDenom] = np.array([fractions.Fraction(i).limit_denominator(maxDenom) for i in CDCList])
        powerCoefs[maxDenom] = scaledCdc
        errors[maxDenom] = error
    for testingPower in powerCoefs:
        testingErrors = errors[testingPower]
        testingCoefficients = powerCoefs[testingPower]
        sumCoef = np.sum(testingCoefficients)
        factorTest = 1 / testingPower
        while sumCoef < testingPower:
            ImprovementIdx = np.argmin(testingErrors)
            testingCoefficients[ImprovementIdx] += 1
            testingErrors[ImprovementIdx] += factorTest
            sumCoef += 1
        print(np.sum(np.abs(testingErrors)))

    # rng = np.random.default_rng()
    # arr2 = rng.random(12)
    # polyForTest = Polynomial(arr2,[0,1],[0,1])
    # polyForTest /= np.float64(polyForTest(1))
    # poly2TestCoef = arr2/np.sum(arr2)
    # poly2Test = poly2TestCoef[0]*createDiskPolyCosineless(4,0,0)
    # for idx in range(1,arr2.shape[0]):
    #     poly2Test += poly2TestCoef[idx] * createDiskPolyCosineless(4, idx, 0)
    # coefs = stripNonSpin(polyForTest, 4)
    # coefs2 = stripNonSpin(poly2Test, 4)
    n = 4
    outputRes = 300
    resolutionInt = 200
    radiusSpace = np.linspace(0, 1, outputRes)
    cDC = complexDoubleCap(n, outputRes, resolutionInt)
    cDC2 = SphereBasics.complexDoubleCapv2(n, outputRes, resolutionInt)
    cDC3, rAdjustedSpace = SphereBasics.complexDoubleCapv3(n, outputRes, resolutionInt)
    # sizev3Valuable = outputRes - 2
    # rAdjustedSpace = SphereBasics.alternateRadiusSpaceConstructor(outputRes, n)

    fig, ax = plt.subplots()
    v1Line, = ax.plot(radiusSpace, cDC, label="v1")
    v2Line, = ax.plot(radiusSpace, cDC2, label='v2')
    v3Line, = ax.plot(SphereBasics.padArray(rAdjustedSpace), SphereBasics.padArray(cDC3), label='v3')
    plt.show()
    weightPoly = complexWeightPolyCreator(n)
    weightArray = weightPoly(radiusSpace)
    polyEstCDC = Polynomial.fit(radiusSpace, cDC, 14)
    polyEstCDC2 = Polynomial.fit(radiusSpace, cDC2, 14)
    polyEstCDC3 = Polynomial.fit(rAdjustedSpace, cDC3, 14)
    paddedPolyEstCDCv3 = Polynomial.fit(SphereBasics.padArray(rAdjustedSpace), SphereBasics.padArray(cDC3), 14)
    intCdc = radialIntegrator(cDC, n)
    intPolyCdc = radialIntegrator(polyEstCDC(radiusSpace), n)
    intCdc2 = radialIntegrator(cDC2, n)
    intPolyCdc2 = radialIntegrator(polyEstCDC2(radiusSpace), n)
    intCdc3 = np.sum(cDC3) / outputRes
    intPolyCdc3 = np.sum(polyEstCDC3(rAdjustedSpace)) / outputRes
    intPaddedPolyCdc3 = np.sum(paddedPolyEstCDCv3(rAdjustedSpace)) / outputRes

    radiusV2 = SphereBasics.alternateRadiusSpaceConstructor(25000, n)
    coefsCDC = stripNonSpin(cDC, n, outputRes)
    coefsPolyCDC = stripNonSpin(polyEstCDC, n, 10 * outputRes)
    # polySum = np.sum(polyEstCDC)
    # truePolyEst = Polynomial(polyEstCDC)
    coefsV2CDC = stripNonSpin(cDC2, n, outputRes)
    coefsPolyCDC2 = stripNonSpin(polyEstCDC2, n, 10 * outputRes)
    coefsPolyCDC2v2 = alternateStripNonSpin(polyEstCDC2(radiusV2), n, radiusV2, np.ones(outputRes))
    coefsV3CDC = alternateStripNonSpin(cDC3, n, rAdjustedSpace, np.ones(outputRes))
    coefsPolyCDC3 = alternateStripNonSpin(polyEstCDC3(radiusV2), n, radiusV2, np.ones(outputRes))
    coefsPolyCDC3v2 = alternateStripNonSpin(paddedPolyEstCDCv3(radiusV2), n, radiusV2, np.ones(outputRes))
    strippedSum = np.sum(coefsCDC)
    strippedSumPoly = np.sum(coefsPolyCDC)
    strippedSum2 = np.sum(coefsV2CDC)
    strippedSum2Poly = np.sum(coefsPolyCDC2)
    strippedSum2Polyv2 = np.sum(coefsPolyCDC2v2)
    strippedSum3 = np.sum(coefsV3CDC)
    strippedSum3Poly = np.sum(coefsPolyCDC3)
    strippedSum3Polyv2 = np.sum(coefsPolyCDC3v2)

    weightPoly = complexWeightPolyCreator(n)
    polyLists = []
    for k1 in range(5):
        polyLists.append([])
        for gamma1 in range(5):
            polyLists[k1].append(createDiskPolyCosineless(n, k1, gamma1))
    resolution = 10000
    rSpace = np.linspace(0, 1, resolution, True)
    for k1 in range(5):
        for gamma1 in range(5):
            for k2 in range(5):
                curPoly = polyLists[k1][gamma1] * polyLists[k2][gamma1] * weightPoly
                integralCur = np.sum(curPoly(rSpace)) / resolution
                if integralCur > 0.000001:
                    print(f"Non zero integral at gamma {gamma1}, k1 {k1}, k2 {k2}. Value {integralCur}")
    rToX = np.poly1d((2, 0, -1))
    # rToXNew = Polynomial((-1,0,2),[0,1],[0,1])
    # k = random.randint(1,11)
    # alpha = random.randint(2,11)
    # beta = random.randint(2,11)
    # rPowerBeta = np.poly1d([1]+[0]*beta)
    # rPBNew = Polynomial([0]*beta+[1],[0,1],[0,1])
    # # alpha=3
    # jacobiBase = scipy.special.jacobi(k, alpha, beta)
    # rightSize = jacobiBase(1)
    # jacobiNormalized = jacobiBase/rightSize
    # newJacobi = jacobiNormalized(rToX)
    #
    # newJacobiV2 = Polynomial(jacobiNormalized.coef[::-1])(rToXNew)
    # inputR = np.linspace(0,1,100,True)
    # # inputX = rToX(inputR)
    # # print(jacobiNormalized(inputX))
    # jacobiVals = newJacobi(inputR)
    # diskNonCos = newJacobi*rPowerBeta
    # newDiskNonCos = newJacobiV2*rPBNew
    # print(diskNonCos(inputR))
    # print(newDiskNonCos(inputR))
    # print(jacobiVals)
    # fractionsCdc = [fractions.Fraction(i) for i in coefsPolyCDC]
    # simplifiedFractions = {}
    # powerCoefs = {}
    # errors = {}
    # for power in range(16):
    #     maxDenom = 2 ** (power + 1)
    #     scaledCdc = np.array([round(maxDenom * i) for i in fractionsCdc])
    #     error = np.array([scaledCdc[idx] / maxDenom - coefsPolyCDC[idx] for idx in range(len(scaledCdc))])
    #     simplifiedFractions[maxDenom] = np.array(
    #         [fractions.Fraction(i).limit_denominator(maxDenom) for i in coefsPolyCDC])
    #     powerCoefs[maxDenom] = scaledCdc
    #     errors[maxDenom] = error
    # bestError = 1
    # bestPower = 0
    # for testingPower in powerCoefs:
    #     testingErrors = errors[testingPower]
    #     testingCoefficients = powerCoefs[testingPower]
    #     sumCoef = np.sum(testingCoefficients)
    #     factorTest = 1 / testingPower
    #     while sumCoef < testingPower:
    #         ImprovementIdx = np.argmin(testingErrors)
    #         testingCoefficients[ImprovementIdx] += 1
    #         testingErrors[ImprovementIdx] += factorTest
    #         sumCoef += 1
    #     totError = np.sum(np.abs(testingErrors))
    #     print(totError)
    #     if totError < bestError:
    #         bestError = totError
    #         bestPower = testingPower
    # print(bestPower)



def facetTest(facetArray,betaArray,N):
    nrOfFacets = betaArray.shape[0]
    # boolArray = np.zeros(nrOfFacets,dtype=bool)
    charMatrix = characteristicMatrix(N)
    resultMatrix = np.zeros((facetArray.shape[0],charMatrix.shape[1]))
    for facetIdx in range(facetArray.shape[0]):
        for setIdx in range(charMatrix.shape[1]):
            resultMatrix[facetIdx,setIdx] = np.linalg.multi_dot([charMatrix[:,setIdx],facetArray[facetIdx],charMatrix[:,setIdx]])
    # resultMatrix = facetArray @ charMatrix
    boolMatrix = (resultMatrix <= betaArray[:,None])
    boolArray = np.any(boolMatrix, axis=1)
    invalidArray = np.logical_not(boolArray)
    return boolArray, invalidArray


def facetReader(filename):
    ineqList = []
    betaList = []
    NGlobal = -1
    with open(filename, 'r') as file:
        df = pd.read_csv(filename, delimiter=' ',header=None)
        for index, row in df.iterrows():
            rowValues = row.values
            N = rowValues[0]
            if index == 0:
                NGlobal = N
            else:
                if N != NGlobal:
                    print("Non uniform N")
            beta = rowValues[1]
            triangleDescription = rowValues[2:]
            A = np.zeros((N,N), dtype=np.int32)
            elementIdx = 0
            for i in range(N):
                for j in range(i,N):
                    A[j, i] = triangleDescription[elementIdx]
                    A[i,j] = triangleDescription[elementIdx]
                    elementIdx += 1
            ineqList.append(A)
            betaList.append(beta)
    ineqArray = np.array(ineqList)
    betaArray = np.array(betaList)
    validArray, invalidArray = facetTest(ineqArray, betaArray,NGlobal)

    if np.any(invalidArray):
        print(f"{np.where(invalidArray)} are invalid facets")
        return False, ineqArray, betaArray
    else:
        return True, ineqArray, betaArray

#     function
#     read_bqp_inequalities(filename)
#     ineqs = []
#     for s in eachline(filename)
#         numbers = map(x -> parse(Int, x), split(s))
#
#         N = numbers[1]
#         beta = numbers[2]
#
#         A = zeros(Int, N, N)
#         cur = 3
#         for i = 1:N
#         for j = i:N
#         A[i, j] = A[j, i] = numbers[cur]
#         cur += 1
#     end
#
#
# end
#
# if !is_bqp_ineq(A, beta)
# error("invalid BQP inequality")
# end
#
# push!(ineqs, (A, beta))
# end
#
# return ineqs
# end


def membership_indicator(X: np.ndarray,
                         v: np.ndarray,
                         criteria: list[tuple[tuple[float, float], tuple[float, float]]]
                         ) -> np.ndarray:
    """
    Determine membership in a set defined by radial and angular constraints on the inner product with v.

    Parameters
    ----------
    X : np.ndarray, shape (n, m)
        Array of n vectors of length m to test for membership.
    v : np.ndarray, shape (m,)
        The "pole" vector defining the set.
    criteria : list of ((r_min, r_max), (theta_min, theta_max))
        List of allowed (magnitude, argument) intervals:
        - r_min, r_max define the inclusive range for |<v, w>|.
        - theta_min, theta_max define the inclusive range for arg(<v, w>) (in radians, on (-pi, pi]).

    Returns
    -------
    np.ndarray, shape (n,)
        Array of 0/1 values: 1 if the corresponding row of X lies in any of the specified radial/angular zones, 0 otherwise.
    """
    # Compute complex inner products <v, w_i>
    inner = X.dot(np.conj(v).T)
    # Magnitudes and arguments
    r = np.abs(inner)
    angles = np.angle(inner)

    # Initialize mask to False for all points
    mask = np.zeros_like(r, dtype=bool)

    # Apply each (radial, angular) criterion
    for (r_min, r_max), (theta_min, theta_max) in criteria:
        rad_mask = (r >= r_min) & (r <= r_max)
        ang_mask = (angles > theta_min) & (angles < theta_max)
        mask |= (rad_mask & ang_mask)

    return mask.astype(int)


def numeric_integration(radius_resolution: int,
                        theta_resolution: int,
                        test_radius_resolution: int,
                        test_theta_resolution: int,
                        weightpoly1,
                        weightpoly2,
                        criteria: list[tuple[tuple[float, float], tuple[float, float]]]
                        ) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform the numerical integration over v and w grids.

    Returns
    -------
    vresult : np.ndarray, shape (radius_resolution, theta_resolution)
        The integrated result for each v-grid point.
    VR : np.ndarray, shape (radius_resolution, theta_resolution)
        Radial coordinate grid for v.
    VTHETA : np.ndarray, shape (radius_resolution, theta_resolution)
        Angular coordinate grid for v.
    """
    # Generate v-grid (radius and theta)
    vr = np.linspace(0, 1, radius_resolution)
    vt = np.linspace(0, 2 * np.pi, theta_resolution, endpoint=False)
    VR, VT = np.meshgrid(vr, vt, indexing='ij')
    Nv = radius_resolution * theta_resolution
    # v points in C^3
    v1 = VR.flatten() * np.exp(1j * VT.flatten())
    v2 = np.sqrt(1 - VR.flatten() ** 2)
    # Precompute conjugates
    v1_conj = np.conj(v1)
    v2_conj = np.conj(v2)



    # Build w1 grid
    wr1 = np.linspace(0, 1, test_radius_resolution + 1)[1:]  # skip zero
    th1 = np.linspace(np.pi / test_theta_resolution, 2 * np.pi + np.pi / test_theta_resolution, test_theta_resolution, endpoint=False)
    # Nw1 = wr1.size

    # Precompute which w1 pass e1-membership
    # Equivalent to: arg(w1) in criteria, and abs(w1) in radial criteria
    w1_points = np.outer(wr1, np.exp(1j * th1))
    rad1, theta1 = z2polar(w1_points)
    mask_w1 = np.zeros_like(rad1, dtype=bool)
    for (rmin, rmax), (tmin, tmax) in criteria:
        rad_ok = (rad1 >= rmin) & (rad1 <= rmax)
        ang_w1 = np.angle(w1_points)
        ang_ok = (ang_w1 > tmin) & (ang_w1 < tmax)
        mask_w1 |= (rad_ok & ang_ok)
    # Compute weight for each w1 point
    w1_weights = weightpoly1(rad1)

    # Isolate only the valid starting coordinates
    validW1Points = w1_points[mask_w1]
    validW1Weights = w1_weights[mask_w1]
    validW1Rads = rad1[mask_w1]
    validW1Thetas = theta1[mask_w1]



    # Build w2 grid (same resolution)
    wr2 = wr1.copy()
    th2 = th1.copy()


    # Precompute w2 values and weight2
    w2_vals = np.outer(wr2 , np.exp(1j * th2)).flatten()
    rads2,theta2 = z2polar(w2_vals)
    w2_weights = weightpoly2(rads2)

    # Prepare result accumulator
    vresult = np.zeros(Nv)
    Nw1 = th1.size
    Nw2 = wr2.size
    denom = Nw1 * Nw2


    # innerRes1 = np.tensordot(validW1Points, w2_vals,axes=0)
    # radsInner1 = np.outer(validW1Rads, vr)
    # thetaInner1 = np.subtract(

    # e1 for membership test
    e1 = np.array([1 + 0j, 0 + 0j, 0 + 0j])



    # it = np.nditer(mask_w1, flags=['multi_index'])
    # Loop over w1 indices
    for i in range(validW1Points.size):  # w1_val in it:    # range(Nw1):
        # i = it.multi_index
        # Pre-inner product for v: conj(v1)*w1
        w1_val = validW1Points[i]
        pre_inner = v1_conj * w1_val  # shape (Nv,)

        # weights for this w1 across all w2
        w1_weight = validW1Weights[i]
        weights_sub = w1_weight * w2_weights  # shape (Nw2,)

        # Compute inner2 and full_inner: shape (Nv, Nw2)
        inner2 = v2_conj[:, None] * w2_vals[None, :]
        full_inner = pre_inner[:, None] + inner2

        # Check criteria for each (v, w2)
        r = np.abs(full_inner)
        ang = np.angle(full_inner)
        mask = np.zeros_like(full_inner, dtype=bool)
        for (rmin, rmax), (tmin, tmax) in criteria:
            mask |= (r >= rmin) & (r <= rmax) & (ang > tmin) & (ang < tmax)

        # Accumulate weighted sum over w2 for each v
        # mask.dot(weights_sub) yields shape (Nv,)
        vresult += mask.dot(weights_sub)/denom

    # Normalize and reshape
    vresult /= denom
    return vresult.reshape(radius_resolution, theta_resolution), VR, VT

def save_test_result(testSuperDict, fileName):
    # Load existing data
    with open(fileName, "r") as f:
        data = json.load(f)

    # Add new result
    data.update(testSuperDict)

    # Save updated dictionary
    with open(fileName, "w") as f:
        json.dump(data, f, indent=2, default=complex_to_json_safe)


def complex_to_json_safe(obj):
    if isinstance(obj, complex):
        return f"{obj.real}+ j*{obj.imag}"
    raise TypeError(f"Type {type(obj)} not serializable")


def runTests(testComplexDimension, forbiddenRadius, forbiddenAngle, testAmount, argsTest, testType):
    testSaveLocation = f"TestResults_{testType}_" + datetime.datetime.now().strftime("%d_%m_%H-%M") + ".json"
    if not os.path.exists(testSaveLocation):
        with open(testSaveLocation, "w") as f:
            json.dump({}, f)
    for testNr in range(testAmount):
        testResult = finalBQPMethod(testComplexDimension,
                                    forbiddenRad=forbiddenRadius, forbiddenTheta=forbiddenAngle,
                                    **argsTest)
        testResultDict = {f"TestNr{testNr}":testResult}
        save_test_result(testResultDict, testSaveLocation)
        del testResult, testResultDict
    return


def oldTests():
    testCompDim = 2
    rIntervalsTest = np.array([[0, 1]])
    # rIntervalsTest = np.array([[1/np.sqrt(2),1]])   # np.array([[0,1]])
    thetaIntervalsTest = np.array([[0, np.pi / 2]])
    # thetaIntervalsTest = np.array([[0,np.pi/6], [np.pi*1/2,np.pi*5/6]])     # np.array([[0,np.pi]])
    halfOpenRectsToCoefs(rIntervalsTest, thetaIntervalsTest, 40, 33, testCompDim)
    # quicktest= True
    solRads = 10
    solThetas = 25
    solPoints = solRads * solThetas
    weightPolyBase = complexWeightPolyCreator(testCompDim)
    weightPolySec = complexWeightPolyCreator(testCompDim - 1)
    # criteria = [
    #     ((0.0, 1.0), (-np.pi / 2, np.pi / 2))
    # ]
    # res, VR, VTHETA = numeric_integration(
    #     radius_resolution=solRads,
    #     theta_resolution=solThetas,
    #     test_radius_resolution=15,
    #     test_theta_resolution=101,
    #     weightpoly1=weightPolyBase,
    #     weightpoly2=weightPolySec,
    #     criteria=criteria
    # )
    # resMaxFactor = 0.5 / np.max(res)
    # res *= resMaxFactor
    # import matplotlib.pyplot as plt
    # xMesh = VR * np.cos(VTHETA)
    # yMesh = VR * np.sin(VTHETA)
    # test1 = 1/4 + 1/4 * VR * np.cos(VTHETA)
    # test2 = (1-np.arccos(xMesh)/np.pi)/2
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(xMesh, yMesh,
    #                        res,
    #                        linewidth=1,
    #                        antialiased=True,
    #                        rstride=3, cstride=3
    #                        ,color=(1,0,0,0.5))
    # surf = ax.plot_surface(xMesh, yMesh,
    #                        test1,
    #                        linewidth=1,
    #                        antialiased=True,
    #                        rstride=3, cstride=3
    #                        , color=(0, 1, 0, 0.5)
    #                        )
    # surf = ax.plot_surface(xMesh, yMesh,
    #                        test2,
    #                        linewidth=1,
    #                        antialiased=True,
    #                        rstride=3, cstride=3
    #                        ,color=(0,0,1,0.5))
    # # ax.view_init(elev=0., azim=-90, roll=0)
    # # plt.show()
    # ax.view_init(elev=15., azim=-90, roll=0)
    # plt.show()
    # mse1 = np.mean(np.square(res - test1))
    # mse2 = np.mean(np.square(res - test2))
    # sumDiff1 = np.sum(weightPolyBase(np.abs(res - test1)))/solPoints
    # sumDiff2 = np.sum(weightPolyBase(np.abs(res - test2)))/solPoints
    # print("Result MSE 1:", mse1)
    # print("Result MSE 2:", mse2)
    # print("Total difference 1:", sumDiff1)
    # print("Total difference 2:", sumDiff2)
    # print(res)

    # charMatrix = characteristicMatrix(6)
    # bqpVerts = np.zeros((64, 6, 6))
    # for idx in range(64):
    #     bqpVerts[idx] = np.outer(charMatrix.T[idx], charMatrix.T[idx])
    #
    # maskMult = np.ones(64, dtype=bool)
    # maskMult[0] = False
    # for pwr in range(6):
    #     maskMult[2 ** pwr] = False
    #
    # multBqpVert = bqpVerts[maskMult]
    #
    # multBqpVertFlat = multBqpVert.reshape((57, 36))
    # orthBQP = np.zeros_like(multBqpVertFlat)
    # orthBQP[0] = multBqpVertFlat[0]
    # skipThisVert = np.zeros(57,dtype=bool)
    # for idx1 in range(1, 57):
    #     curVec = multBqpVertFlat[idx1].copy()
    #     for idx2 in range(idx1):
    #         if not skipThisVert[idx2]:
    #             compVec = orthBQP[idx2]
    #             compSize = np.dot(compVec, compVec)
    #             curSize = np.dot(compVec, curVec)
    #             curVec -= (curSize / compSize) * compVec
    #     curVecMask = (np.abs(curVec)>1e-8)
    #     if np.max(curVecMask) == True:
    #         minCurNonzero = np.min(np.abs(curVec[curVecMask]))
    #         curVecScaled = np.zeros_like(curVec)
    #         curVecScaled[curVecMask] = curVec[curVecMask]/minCurNonzero
    #         orthBQP[idx1] = curVecScaled
    #     else:
    #         skipThisVert[idx1] = True
    #
    #

    testSaveLocation = "TestResults_" + datetime.datetime.now().strftime("%d_%m_%H-%M") + ".json"
    if not os.path.exists(testSaveLocation):
        with open(testSaveLocation, "w") as f:
            json.dump({}, f)
    quicktest = True
    plotBool = False
    printBool = False
    validFacets, ineqArray, betaArray = facetReader("bqp6.dat")

    if quicktest:
        # n = 4
        #
        testMax = 0
        testResults = np.ones((testMax, 7))
        testDimension = 3
        coefsCDCV2 = c_coeffs(testDimension, 50)
        nrOfWins = np.zeros(7)
        nrOfPointsets = 9
        SizeOfPointSets = 6
        # maxDegAll = 20
        maxDegAll = 200

        resolutionBase = 201
        outputResBase = 300
        weightPolyBase = complexWeightPolyCreator(testDimension)
        weightIntBase = integratedWeightPolyCreator(testDimension) + 1
        heighestWeightPointBase = np.sqrt(1 / (2 * (testDimension - 2) + 1))
        heighestWeightBase = weightPolyBase(heighestWeightPointBase)
        scaledWeightPolyBase = weightPolyBase / heighestWeightBase
        # cDCBase = SphereBasics.complexDoubleCapv2(testDimension, outputResBase, resolutionBase)

        # radiusSpaceBase = np.linspace(0, 1, outputResBase, endpoint=True)

        # polyEstCDCBase = Polynomial.fit(radiusSpaceBase, cDCBase, 4 * (testDimension - 1))
        # coefsPolyCDCBase = stripNonSpin(polyEstCDCBase, testDimension, 10 * outputResBase)
        coefsPolyCDCBase = c_coeffs(testDimension, maxDegAll)
        polyListBase = [createDiskPolyCosineless(testDimension, kIdx, 0) for kIdx in range(maxDegAll + 1)]
        basePolyBase = sum(polyListBase[kIdx] * coefsPolyCDCBase[kIdx] for kIdx in range(len(coefsPolyCDCBase)))

        bestObjValStart = iterativeProbabilisticImprovingBPQ(testDimension,
                                                             maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                             maxSetSize=SizeOfPointSets, maxDeg=maxDegAll,
                                                             uniformCoordinateWise=True)
        bestObjValStart = iterativeProbabilisticImprovingBPQ(testDimension,
                                                             maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                             maxSetSize=SizeOfPointSets, maxDeg=maxDegAll,
                                                             basePoly=basePolyBase, uniformCoordinateWise=True)
        testResultDict = {"StartingTest": bestObjValStart}
        save_test_result(testResultDict, testSaveLocation)
        del testResultDict, bestObjValStart
        for testNr in range(testMax):
            TrueRandom = modelMultipleBQP(0, 0, testDimension, 0,
                                          maxDegAll, nrOfPoints=SizeOfPointSets, nrOfPointSets=nrOfPointsets)[3]
            bestObjValrelative = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                    maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                    maxSetSize=SizeOfPointSets, maxDeg=maxDegAll,
                                                                    basePoly=basePolyBase)
            bestObjValSpread = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                  maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                  maxSetSize=SizeOfPointSets,
                                                                  spreadPoints=True, maxDeg=maxDegAll,
                                                                  basePoly=basePolyBase)
            bestObjValInverse = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                   maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                   maxSetSize=SizeOfPointSets,
                                                                   reverseWeighted=True, maxDeg=maxDegAll,
                                                                   basePoly=basePolyBase)

            bestObjValFlat = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                maxSetSize=SizeOfPointSets, relativityScale=0,
                                                                maxDeg=maxDegAll,
                                                                basePoly=basePolyBase)
            bestObjValScaled = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                  maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                  maxSetSize=SizeOfPointSets, relativityScale=1,
                                                                  maxDeg=maxDegAll,
                                                                  basePoly=basePolyBase)
            bestObjValUniform = iterativeProbabilisticImprovingBPQ(testDimension,
                                                                   maxIter=nrOfPointsets * (SizeOfPointSets - 2),
                                                                   maxSetSize=SizeOfPointSets,
                                                                   uniformCoordinateWise=True, maxDeg=maxDegAll,
                                                                   basePoly=basePolyBase)

            testResultDict = {f"Random{testNr}": TrueRandom, f"Relative{testNr}": bestObjValrelative,
                              f"Spread{testNr}": bestObjValSpread, f"Inverse{testNr}": bestObjValInverse,
                              f"Flat{testNr}": bestObjValFlat, f"Scaled{testNr}": bestObjValScaled,
                              f"Uniform{testNr}": bestObjValUniform}
            save_test_result(testResultDict, testSaveLocation)
            del testResultDict, TrueRandom, bestObjValrelative, bestObjValSpread, \
                bestObjValInverse, bestObjValFlat, bestObjValScaled, bestObjValUniform
            # testResultArray = np.array([TrueRandom,bestObjValSpread,bestObjValInverse,bestObjValrelative,bestObjValFlat,
            #                             bestObjValScaled,bestObjValUniform])
            # objErrorMask = (testResultArray < (1/2)**(testDimension-1))
            # testResultArray[objErrorMask] = 1/testDimension
            # winner = np.argmin(testResultArray)
            # testResults[testNr]=testResultArray
            # print(testResultArray)
            # nrOfWins[winner] += 1
        # if testMax>0:
        #     overalWinner = np.argmin(testResults)
        #     print(f"Nr of wins: {nrOfWins} (true random, relative error, flat error, scaled error, uniform coordinate)")
        #     print(f"Overal winner: {overalWinner} with Objective Value: {testResults.flatten()[overalWinner]}")
    else:
        testMatrix = characteristicMatrix(6)
        modelBQP(0, 0, 3, 15, 15, nrOfPoints=14)
        makeModelv2((dimension - 3.0) / 2.0, (dimension - 3.0) / 2.0, 0)
        denomList = list(range(1, 256))
        nonSimpleSolutions = {}
        fixedMaxTime = 0
        stopAtMinTime = 0
        lp1 = LineProfiler()
        lp_wrapper1 = lp1(findMinBlock)
        lp2 = LineProfiler()
        lp_wrapper2 = lp2(findMinVar)
        printBool = False

        for testNr in range(nrOfTests):
            if printBool:
                print(f"Test nr {testNr + 1}:")
            testDenomRad = random.choice(denomList)
            # testNomRad = random.randint(0, testDenomRad)
            testNomRad = testDenomRad - 1
            testRad = fractions.Fraction(testNomRad, testDenomRad)
            testDenomAng = random.choice(denomList)
            testNomAng = random.randint(1, testDenomAng)
            testAng = fractions.Fraction(testNomAng, testDenomAng)
            if testRad == 1 and testAng == 0:
                if printBool:
                    print("\t infeasible (z==1)")
            else:
                if printBool:
                    print(f"\t Testing forbidden radius: {testRad}, theta: {testAng}")
                # fmStart = time.time()
                # thetaPartVal, thetaPartGamma, thetaPartK = findMinBlock(testRad, testAng,
                #                                                         dimension-2, max(nrOfVars, testDenomRad))
                # fixedMaxTime += time.time() - fmStart
                samStart = time.time()
                thetaPartVal, thetaPartGamma, thetaPartK = findMinVar(testRad, testAng, dimension - 2)
                stopAtMinTime += time.time() - samStart
                if printBool:
                    print(f"\t Found theta value: {-thetaPartVal / (1 - thetaPartVal)} (thetapart: {thetaPartVal})")
                    print(f"\t Value found at gamma: {thetaPartGamma}, k: {thetaPartK}")
                if (thetaPartGamma > 0 and thetaPartK > 0) or (testRad < 1 and thetaPartGamma > 5) or thetaPartK > 5:
                    nonSimpleSolutions[testNr] = [thetaPartVal, testRad, testAng, thetaPartGamma, thetaPartK]

        for testNr, testParams in nonSimpleSolutions.items():
            print(f"Test nr {testNr + 1} with params: {testParams}")
        print(f"fixed max time: {fixedMaxTime} while variable stop time: {stopAtMinTime}")
        tierListPrinter(nonSimpleSolutions, 10)
        # lp1.print_stats()
        # lp2.print_stats()
        # makeModel("yes")
        #
        # newCalcForbiddenDistancev2 = 2*(distanceForbidden**2) - 1
        # makeModelv2((dimension-3.0)/2.0, -1/2, newCalcForbiddenDistancev2)
        # # Complex RP
        #
        # makeModelv2((dimension - 2.0), 0, newCalcForbiddenDistancev2)
        # # Complex Sphere
        #
        #
        # # makeModelv2((dimension - 2.0), (dimension - 2.0), distanceForbidden)
        # makeModelCascading(dimension-2.0, distanceForbidden, newCalcForbiddenDistancev2)


def quickJumpNavigator():
    return "ok"


if __name__ == '__main__':
    allTestTypes = {0:"reverseWeighted",1:"spreadPoints",2:"sequentialEdges",3:"uniformCoordinateWise"}
    testDim = 3
    testRad = 0
    testTheta = 0
    testMatrix = characteristicMatrix(6)
    # modelBQP(0, 0, 3, 15, 15, nrOfPoints=14)
    # for dimension in range(testDim, 20):
    #     makeModelv2((dimension - 3.0) / 2.0, (dimension - 3.0) / 2.0, 0)
    nrOfTests = 1
    bqpType = allTestTypes[2]
    testArgs = {bqpType:True, "maxDeg":200,"sizeOfBQPSet":6,"setAmount":5, "setLinks":2, "improvementWithFacets":True}
    runTests(testDim,testRad,testTheta,nrOfTests,testArgs,bqpType)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# from typing import final
import re

# from sympy.stats import Gamma

import OrthonormalPolyClasses
import UnconstrainedPolyDiffOptimizer
# import time
import gurobipy as gp
from gurobipy import GRB
import math
import scipy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import fractions
# import random
import scipy.stats as st
from scipy.stats import unitary_group
from scipy.special import hyp2f1, loggamma, binom
from scipy import special
# import sympy as sym
import pandas as pd
import json
import os
import datetime
# from multiprocessing import Pool, cpu_count
from itertools import combinations
# import improvedBQPNLP
# from pympler import asizeof
# import psutil
import Scoreboard
# import gc
# from matplotlib import pyplot as plt

import SphereBasics
# from line_profiler import LineProfiler
# from numpy.polynomial import Polynomial

import minimizationUnconstrained
from SphereBasics import complexWeightPolyCreator, complexDoubleCap, radialIntegrator, \
    integratedWeightPolyCreator, createRandomPointsComplex

from OrthonormalPolyClasses import Jacobi, Disk, DiskCombi, FastRadialEstimator, FastDiskCombiEstimator


facetScoreDF = pd.DataFrame()
facetFileLoc = "facetScores.csv"
nrOfVars = 10
nrOfTests = 200
dimension = 4
distanceForbidden = 0
jacobiType = "new"
charMatrixDict = {}
_RX = re.compile(
    r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([+-])\s*j\*\s*'
    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$'
)

polar2z = lambda r, theta: r * np.exp(1j * theta)
z2polar = lambda z: (np.abs(z), np.angle(z))
# rss = lambda: psutil.Process(os.getpid()).memory_info().rss/1024**2  # MB


def stitchSets(arrayList, setLinks, keepInitials=True):
    dimArray = arrayList[0].shape[1]
    sampleUnitary = unitary_group.rvs(dimArray)
    if setLinks == 1:
        combedArrays = arrayList
    elif len(arrayList) <= setLinks:
        if keepInitials:
            combedArrays = [np.concat([ar for ar in arrayList])] #.dot(np.conjugate(unitary_group.rvs(dimArray)))
        else:
            combedArrays = [np.concat([arrayList[0][:2]] + [arrayList[setIdx][2:]
                                                                      for setIdx in range(len(arrayList))])]
    else:
        combedArrays = []
        if keepInitials:
            for comb in combinations(range(len(arrayList)), setLinks):
                combedArrays.append(np.concat([arrayList[combIdx] for combIdx in comb])) #.dot(np.conjugate(unitary_group.rvs(dimArray)))
        else:
            for comb in combinations(range(len(arrayList)), setLinks):
                combedArrays.append(np.concat([arrayList[0][:2]] +
                                                   [arrayList[combIdx][2:] for combIdx in comb]))
    return combedArrays


def characteristicMatrix(nrOfPoints):
    nrOfSets = 2**nrOfPoints
    setIndices = np.arange(nrOfSets)
    charMatrix = np.empty((nrOfPoints, nrOfSets), dtype=bool)
    for pointIdx in range(nrOfPoints):
        relativeIndices = setIndices//(2**pointIdx)
        charIndices = relativeIndices % 2
        charMatrix[pointIdx] = charIndices
    return charMatrix


def finalBQPModel(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1):
    alpha = complexDimension - 2
    gammaMax = int(gammaMax + 1)
    kMax = int(kMax + 1)
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)
    if listOfPointSets is None:
        listOfPointSets = []
        for pointsSetIdx in range(nrOfPointSets):
            listOfPointSets.append(SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints,includeInner=False)[0])
    else:
        nrOfPointSets = len(listOfPointSets)
    # charMatricesList = []
    try:
        setSizeList = []
        nrOfPointsList = []
        IPMList = []
        diskPolyValueTensorList = []
        diskPolysByGamma = [Disk(alpha, curGamma, kMax-1) for curGamma in range(gammaMax)]

        for pointSetIdx in range(nrOfPointSets):
            curPointSet = listOfPointSets[pointSetIdx]
            nrOfPoints = curPointSet.shape[0]
            # charMatrix = characteristicMatrix(nrOfPoints)
            nrOfSets = 2**nrOfPoints

            innerProductMatrix = curPointSet @ curPointSet.conj().T
            ipmRads, ipmAngles = z2polar(innerProductMatrix)
            diskPolyValueTensor = np.zeros((gammaMax, kMax, nrOfPoints, nrOfPoints))
            for gamma in range(gammaMax):
                diskPolyValueTensor[gamma] = diskPolysByGamma[gamma].calcAtRTheta(ipmRads, ipmAngles)
            # charMatricesList.append(charMatrix)
            setSizeList.append(nrOfSets)
            nrOfPointsList.append(nrOfPoints)
            IPMList.append(innerProductMatrix)
            diskPolyValueTensorList.append(diskPolyValueTensor)
        forbiddenInnerproductVals = np.zeros((gammaMax,kMax))
        for gamma in range(gammaMax):
            forbiddenInnerproductVals[gamma] = diskPolysByGamma[gamma].calcAtRTheta(forbiddenRadius, forbiddenAngle)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)


        env = gp.Env(empty=True)
        env.start()

        m = gp.Model(env=env)

        # env.start()
        # m = gp.Model(f"BQP model for forbidden innerproduct {forbiddenZ} in dimension {complexDimension}, "
        #              f"with {nrOfPoints} BQP points,"
        #              f" gamma checked {gammaMax}, k checked {kMax}", env=env)
        m.setParam("OutputFlag", 1)
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

            curDPVT = diskPolyValueTensorList[pointSetIdx]
            curNrOfPoints = nrOfPointsList[pointSetIdx]
            curCharMatrix = charMatrixDict[curNrOfPoints]

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
                diskSum += diskPolyWeights[gammaIdx,kIdx].X
                diskWeightsArray[gammaIdx,kIdx] = diskPolyWeights[gammaIdx,kIdx].X
        setSizeList = []
        setTotal = 0
        AddedPoints = 1
        for pointset in listOfPointSets:
            curSize = len(pointset)
            setSizeList.append(curSize)
            setTotal += curSize
            AddedPoints += curSize-2
        adjustedDWA = diskWeightsArray/diskSum
        if len(setSizeList) > 2:
            if min(setSizeList) == max(setSizeList):
                x=1

        return adjustedDWA, diskSum, diskWeightsArray
    finally:
        try:
            m.dispose()
        except Exception:
            pass
        try:
            env.dispose()
        except Exception:
            pass

        del diskPolysByGamma, listOfPointSets, setWeights, lbConstr, forbiddenInnerproductVals, (
            diskPolyWeights), diskPolyValueTensorList, diskPolyValueTensor, curDPVT


def finalBQPModelv2(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1,finalRun=False,compensateConcat=False):
    alpha = complexDimension - 2
    if gammaMax == 0:
        kOnly = True
    else:
        kOnly = False
    gammaMax = int(gammaMax + 1)
    kMax = int(kMax + 1)
    if compensateConcat:
        if kOnly:
            kBorder = np.array([kMax],dtype=int)
            gammaBorder = np.zeros_like(kBorder)
        else:
            kBorder = np.arange(kMax+1,dtype=int)
            gammaBorder = gammaMax*np.ones_like(kBorder)
            gammaBorder[-1] = 0
    else:
        kBorder = np.zeros(1,dtype=int)
        gammaBorder = np.zeros(1,dtype=int)
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)
    errorForbidden = superBound(forbiddenRadius,complexDimension,kBorder,gammaBorder)[0]

    # ---- build inputs (try to keep them small / see §3) ----
    if listOfPointSets is None:
        listOfPointSets = [
            SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints, baseZ=forbiddenZ)[0]
            for _ in range(nrOfPointSets)
        ]
    else:
        nrOfPointSets = len(listOfPointSets)

    diskPolysByGamma = [Disk(alpha, g, kMax-1) for g in range(gammaMax)]

    # ---- Gurobi in short-lived env ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    try:
        m = gp.Model(env=env)
        m.setParam("OutputFlag", 1)
        m.setParam(GRB.Param.FeasibilityTol, 1e-8)
        m.setParam(GRB.Param.BarConvTol, 1e-9)
        # m.setParam(GRB.Param.BarQCPConvTol, 1e-9)
        # m.setParam(GRB.Param.MIPGap, 1e-9)
        # m.setParam(GRB.Param.IntFeasTol, 1e-9)
        # m.setParam(GRB.Param.PSDTol, 1e-9)
        m.setParam(GRB.Param.OptimalityTol, 1e-8)
        m.setParam(GRB.Param.DualReductions, 0)
        # m.setParam(GRB.Param.Crossover, 2)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.Method, 1)
        m.setParam(GRB.Param.BarHomogeneous, 1)
        # m.setParam(GRB.Param.ScaleFlag,0)
        if finalRun:
            m.setParam(GRB.Param.NumericFocus, 3)
        m.setParam(GRB.Param.Presolve, 0)

        diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.SEMICONT, lb=0,ub=1, name="w")
        errorFuncWeight = m.addVar(vtype=GRB.SEMICONT, lb=0,ub=1, name="errorWeight")
        if compensateConcat:
            betaForbidden = m.addVar(lb=-1.0, ub=1.0, name="betaAtForbidden")
            m.addConstr(betaForbidden <= errorForbidden*errorFuncWeight)
            m.addConstr(-betaForbidden <= errorForbidden * errorFuncWeight)
        else:
            m.addConstr(errorFuncWeight == 0)
            betaForbidden = m.addVar(lb=0.0,ub=0.0,name="betaAtForbidden")
        # Forbidden-IP constraint
        forbiddenInner = np.zeros((gammaMax, kMax))
        for g in range(gammaMax):
            forbiddenInner[g] = diskPolysByGamma[g].calcAtRTheta(forbiddenRadius, forbiddenAngle)
        m.addConstr(gp.quicksum(forbiddenInner[g, k]*diskPolyWeights[g, k]
                                for g in range(gammaMax) for k in range(kMax))+betaForbidden == 0)

        # “SDP” constraint you coded (quadratic)
        m.addConstr(diskPolyWeights[0, 0] -
                    (diskPolyWeights.sum()+errorFuncWeight)*(diskPolyWeights.sum()+errorFuncWeight) >= 0)
        # m.addConstr(diskPolyWeights.sum() == 1)
        setWeightVars = []
        setWeightConstr = []
        # Build per–point-set pieces
        largestSet = 0
        nInit = listOfPointSets[0].shape[0]
        charM = characteristicMatrix(nInit)
        nrSets = charM.shape[1]
        for ps_idx in range(nrOfPointSets):
            pts = listOfPointSets[ps_idx]
            n = pts.shape[0]
            if charM.shape[0] != n:
                # Characteristic matrix — consider sparse (see §3)
                charM = characteristicMatrix(n)                  # shape (n, nrSets)
                nrSets = charM.shape[1]
            setW = m.addVars(nrSets, vtype=GRB.SEMICONT, lb=0,ub=1, name=f"s_{ps_idx}")

            m.addConstr(setW.sum() == 1)
            if nrSets > largestSet:
                largestSet = nrSets
            # m.addConstr(setW[0]==0)
            # IPM and disk poly values (avoid 4-D tensor; see §3)
            ipm = pts @ pts.conj().T
            r, th = z2polar(ipm)
            np.round(r, 14, out=r)
            np.round(th, 14, out=th)
            if compensateConcat:
                rAlmostOne = (r >= 1-1e-8)
                thAlmostZero = (np.abs(th) <= 1e-8)
                zAlmostOne = rAlmostOne*thAlmostZero
                betaVars = m.addVars(*r.shape,vtype=GRB.CONTINUOUS,lb=-1,ub=1,name=f"PS{ps_idx}_beta")
                errorTerms = superBound(r,complexDimension,kBorder,gammaBorder)
            # For each pair, avoid creating a full charVector array if it’s sparse:
            for i in range(n):
                for j in range(i+1):
                    # left: sum_{g,k} DP[g,k,i,j] * w[g,k]
                    # compute DP row-on-demand, no 4-D tensor:
                    lin = gp.LinExpr()
                    if compensateConcat:
                        m.addConstr(betaVars[i,j]==betaVars[j,i])
                        if zAlmostOne[i,j]:
                            m.addConstr(betaVars[i,j] == errorFuncWeight)
                        else:
                            m.addConstr(betaVars[i,j] <= errorTerms[i,j] * errorFuncWeight)
                            m.addConstr(-betaVars[i, j] <= errorTerms[i, j] * errorFuncWeight)
                        lin.addTerms(1,betaVars[i,j])
                    for g in range(gammaMax):
                        vals = diskPolysByGamma[g].calcAtRTheta(r[i, j], th[i, j])  # shape (kMax,)
                        for k in range(kMax):
                            if vals[k] != 0.0:
                                lin.addTerms(vals[k], diskPolyWeights[g, k])

                    # right: sum_{S} setW[S] * (charM[i,S] & charM[j,S])
                    # use nonzeros only
                    nz = np.flatnonzero(charM[i] * charM[j])
                    setWeightConstr.append(m.addConstr(lin == gp.quicksum(setW[s] for s in nz)))
            setWeightVars.append(setW)
        m.setObjective(diskPolyWeights.sum() + errorFuncWeight, GRB.MAXIMIZE)
        # m.setObjective(diskPolyWeights[0,0],GRB.MAXIMIZE)

        # Lower-bound loop
        lb_ok = False
        p = complexDimension - 1
        while not lb_ok:
            lb = m.addConstr(diskPolyWeights.sum() >= (0.5)**p)
            # lb = m.addConstr(diskPolyWeights[0,0] >= (0.5) ** p)
            m.update()
            m.optimize()
            m.printQuality()  # if it gets that far


            if m.Status != GRB.OPTIMAL and m.Status != GRB.SUBOPTIMAL:
                # print(m.objBound)
                m.remove(lb)
                p += 1
            else:
                # cons = m.getConstrs()
                # vio = m.getAttr("Slack", cons)
                # for i in sorted(range(len(cons)), key=lambda i: vio[i], reverse=True)[:10]:
                #     row = m.getRow(cons[i])
                #     print(f"row {i} vio={vio[i]:.2e} rhs={cons[i].RHS:.3g}")
                # print(m.objBound)
                lb_ok = True
        # print(m.objBound)
        objVal = m.objVal
        # for setW in setWeightVars:
        #     print(sum(setWCur.X for setWCur in setW.values()))
        diskSum = sum(diskPolyWeights[g, k].X for g in range(gammaMax) for k in range(kMax))
        diskWeightsArray = np.array([[diskPolyWeights[g, k].X for k in range(kMax)]
                                     for g in range(gammaMax)])
        adjustedDWA = diskWeightsArray / diskSum
        objBound = m.objBound
        if objBound-diskSum < 0 or objBound > 1.1 * diskSum:
            print(f"bound obj error with primal dual difference: {objBound-diskSum}")
        # print("b) return:", rss())
        if compensateConcat:
            return adjustedDWA, max(objBound, diskSum+errorFuncWeight.X), diskWeightsArray, errorFuncWeight.X
        else:
            return adjustedDWA, max(objBound,diskSum), diskWeightsArray

    finally:
        # Ensure native memory is released
        try:
            m.dispose()
        except Exception:
            pass
        env.dispose()
        # Drop big Python refs promptly
        del diskPolysByGamma, listOfPointSets
        # print("C) end:", rss())


def finalBQPModelv3(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1,finalRun=False):
    # print("A) before env:", rss())
    alpha = complexDimension - 2
    gammaMax = int(gammaMax + 1)
    kMax = int(kMax + 1)
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)

    # ---- build inputs (try to keep them small / see §3) ----
    if listOfPointSets is None:
        listOfPointSets = [
            SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints, baseZ=forbiddenZ)[0]
            for _ in range(nrOfPointSets)
        ]
    else:
        nrOfPointSets = len(listOfPointSets)

    diskPolysByGamma = [Disk(alpha, g, kMax-1) for g in range(gammaMax)]

    # ---- Gurobi in short-lived env ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    try:
        m = gp.Model(env=env)
        m.setParam("OutputFlag", 1)
        m.setParam(GRB.Param.FeasibilityTol, 1e-8)
        m.setParam(GRB.Param.BarConvTol, 1e-9)
        m.setParam(GRB.Param.BarQCPConvTol, 1e-9)
        # m.setParam(GRB.Param.MIPGap, 1e-9)
        m.setParam(GRB.Param.IntFeasTol, 1e-9)
        # m.setParam(GRB.Param.PSDTol, 1e-9)
        m.setParam(GRB.Param.OptimalityTol, 1e-8)
        # m.setParam(GRB.Param.DualReductions, 0)
        m.setParam(GRB.Param.Crossover, 2)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.Method, 2)
        m.setParam(GRB.Param.BarHomogeneous, 1)
        # m.setParam(GRB.Param.ScaleFlag,0)
        if finalRun:
            m.setParam(GRB.Param.NumericFocus, 3)
        # m.setParam(GRB.Param.Presolve, 0)

        diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.SEMICONT, lb=0,ub=1, name="w")

        # Forbidden-IP constraint
        forbiddenInner = np.zeros((gammaMax, kMax))
        for g in range(gammaMax):
            forbiddenInner[g] = diskPolysByGamma[g].calcAtRTheta(forbiddenRadius, forbiddenAngle)
        m.addConstr(gp.quicksum(forbiddenInner[g, k]*diskPolyWeights[g, k]
                                for g in range(gammaMax) for k in range(kMax)) == 0)

        # “SDP” constraint you coded (quadratic)
        # m.addConstr(diskPolyWeights[0, 0] - diskPolyWeights.sum()*diskPolyWeights.sum() >= 0)
        m.addConstr(diskPolyWeights.sum() == 1)
        setWeightVars = []
        setWeightConstr = []
        # Build per–point-set pieces
        largestSet = 0
        nInit = listOfPointSets[0].shape[0]
        charM = characteristicMatrix(nInit)
        nrSets = charM.shape[1]
        for ps_idx in range(nrOfPointSets):
            pts = listOfPointSets[ps_idx]
            n = pts.shape[0]
            if charM.shape[0] != n:
                # Characteristic matrix — consider sparse (see §3)
                charM = characteristicMatrix(n)                  # shape (n, nrSets)
                nrSets = charM.shape[1]
            setW = m.addVars(nrSets, vtype=GRB.SEMICONT, lb=0,ub=1, name=f"s_{ps_idx}")
            # m.addConstr(setW.sum() <= 1)
            if nrSets > largestSet:
                largestSet = nrSets
            m.addConstr(setW[0]==0)
            # IPM and disk poly values (avoid 4-D tensor; see §3)
            ipm = pts @ pts.conj().T
            r, th = z2polar(ipm)
            np.round(r, 14, out=r)
            np.round(th, 14, out=th)
            # For each pair, avoid creating a full charVector array if it’s sparse:
            for i in range(n):
                for j in range(i+1):
                    # left: sum_{g,k} DP[g,k,i,j] * w[g,k]
                    # compute DP row-on-demand, no 4-D tensor:
                    lin = gp.LinExpr()
                    for g in range(gammaMax):
                        vals = diskPolysByGamma[g].calcAtRTheta(r[i, j], th[i, j])  # shape (kMax,)
                        for k in range(kMax):
                            if vals[k] != 0.0:
                                lin.addTerms(vals[k], diskPolyWeights[g, k])

                    # right: sum_{S} setW[S] * (charM[i,S] & charM[j,S])
                    # use nonzeros only
                    nz = np.flatnonzero(charM[i] * charM[j])
                    setWeightConstr.append(m.addConstr(lin == gp.quicksum(setW[s] for s in nz)))
            setWeightVars.append(setW)
        # m.setObjective(diskPolyWeights.sum(), GRB.MAXIMIZE)
        m.setObjective(diskPolyWeights[0,0],GRB.MAXIMIZE)

        # Lower-bound loop
        lb_ok = False
        p = complexDimension - 1
        while not lb_ok:
            # lb = m.addConstr(diskPolyWeights.sum() >= (0.5)**p)
            lb = m.addConstr(diskPolyWeights[0,0] >= (0.5) ** p)
            m.update()
            m.optimize()
            print(m.objBound)
            if m.Status != GRB.OPTIMAL and m.Status != GRB.SUBOPTIMAL:
                print(m.objBound)
                m.remove(lb)
                p += 1
            else:
                lb_ok = True
        print(m.objBound)
        objVal = m.objVal
        for setW in setWeightVars:
            print(sum(setWCur.X for setWCur in setW.values()))
        diskSum = sum(diskPolyWeights[g, k].X for g in range(gammaMax) for k in range(kMax))
        diskWeightsArray = np.array([[diskPolyWeights[g, k].X for k in range(kMax)]
                                     for g in range(gammaMax)])
        adjustedDWA = diskWeightsArray * objVal
        objBound = m.objBound
        if objBound-diskSum < 0 or objBound > 1.1 * diskSum:
            print(f"bound obj error with primal dual difference: {objBound-diskSum}")
        # print("b) return:", rss())
        return adjustedDWA, max(objBound,diskSum), diskWeightsArray

    finally:
        # Ensure native memory is released
        try:
            m.dispose()
        except Exception:
            pass
        env.dispose()
        # Drop big Python refs promptly
        del diskPolysByGamma, listOfPointSets
        # print("C) end:", rss())


def paramSetFromBorder(kBorder, gammaBorder):
    arraySize = np.sum(gammaBorder) + int(gammaBorder.size)
    kArray = np.zeros(arraySize,dtype=int)
    gammaArray = np.zeros(arraySize,dtype=int)
    insertIdx = 0
    for idx in range(kBorder.size):
        gammaBorderVal = gammaBorder[idx]
        sizeOfSubArray = gammaBorderVal + 1
        kBorderVal = kBorder[idx]
        gammaVals = np.arange(sizeOfSubArray)
        kVals = kBorderVal*np.ones(sizeOfSubArray)
        kArray[insertIdx:insertIdx+sizeOfSubArray] = kVals
        gammaArray[insertIdx:insertIdx+sizeOfSubArray] = gammaVals
        insertIdx += sizeOfSubArray
    return kArray, gammaArray


def borderedPrimal(forbiddenRadius, forbiddenAngle, complexDimension, gammaBorder, kBorder, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1,finalRun=False,kOnly=False):
    alpha = complexDimension - 2
    if kOnly:
        gammaMax = 0
        kMax = int(np.max(kBorder))
        numK = kMax + 1
        numGamma = gammaMax + 1
        gammaBorder = np.zeros(1,dtype=int)
        kBorder = np.ones(1,dtype=int)*kMax
        kSet = np.arange(numK)
        gammaSet = np.zeros_like(kSet)
        includedParamsArray = np.zeros((numGamma, numK), dtype=np.bool)
        includedParamsArray[gammaSet, kSet] = True
        kMaxByGamma = np.max(np.multiply(includedParamsArray, np.arange(numK)), axis=1)
        errorForbidden = superBound(forbiddenRadius, complexDimension, np.array([numK]), np.array([0]))[0]
    else:
        kSet,gammaSet = paramSetFromBorder(kBorder, gammaBorder)
        # indicesList = [(kSet[paramIdx],gammaSet[paramIdx]) for paramIdx in range(kSet.size)]
        gammaMax = int(np.max(gammaBorder))
        kMax = int(np.max(kBorder))
        numK = kMax + 1
        numGamma = gammaMax + 1
        includedParamsArray = np.zeros((numGamma,numK),dtype=np.bool)
        includedParamsArray[gammaSet, kSet] = True
        kMaxByGamma = np.max(np.multiply(includedParamsArray, np.arange(numK)), axis=1)
        errorForbidden = superBound(forbiddenRadius, complexDimension, kBorder, gammaBorder)[0]
    # notIncludedParams = np.logical_not(includedParamsArray)
    # numParams = kSet.size
    # if kOnly:
    #     kBorder = np.array([kMax],dtype=int)
    #     gammaBorder = np.zeros_like(kBorder)
    # else:
    #     kBorder = np.arange(kMax+1,dtype=int)
    #     gammaBorder = gammaMax*np.ones_like(kBorder)
    #     gammaBorder[-1] = 0

    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)

    # ---- build inputs (try to keep them small / see §3) ----
    if listOfPointSets is None:
        listOfPointSets = [
            SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints, baseZ=forbiddenZ)[0]
            for _ in range(nrOfPointSets)
        ]
    else:
        nrOfPointSets = len(listOfPointSets)

    listOfBoundArrays = {}
    ipmSetList = {}
    ipmRadList = {}
    ipmThetaList = {}

    for psIdx in range(nrOfPointSets):
        curPS = listOfPointSets[psIdx]
        ipm = curPS @ curPS.conj().T
        ipmSetList[psIdx] = ipm
        ipmRad, ipmTheta = z2polar(ipm)
        ipmRadList[psIdx] = ipmRad
        ipmThetaList[psIdx] = ipmTheta
        if kOnly:
            foundBoundsCur = superBound(ipmRad, complexDimension, np.array([numK]), np.array([0]))
        else:
            foundBoundsCur = superBound(ipmRad, complexDimension, kBorder, gammaBorder)
        np.fill_diagonal(foundBoundsCur, 1.0)
        listOfBoundArrays[psIdx] = foundBoundsCur
        # identityMatrix = np.zeros_like(foundBoundsCur)
        # np.fill_diagonal(identityMatrix, 1.0)
        # listOfIdentityMatrices[psIdx] = identityMatrix

    # kMaxByGamma = np.zeros(gammaMax+1)
    # for curGamma in range(gammaMax+1):
    #     kMaxByGamma[curGamma] = np.max(includedParamsArray,axis=0)
    #     kMaxByGamma[curGamma] = np.max(kSet[np.where(gammaSet == curGamma)])
    diskPolysByGamma = [Disk(alpha, g, kMaxByGamma[g]) for g in range(numGamma)]

    # ---- Gurobi in short-lived env ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    try:
        m = gp.Model(env=env)
        m.setParam("OutputFlag", 1)
        m.setParam(GRB.Param.FeasibilityTol, 1e-8)
        m.setParam(GRB.Param.BarConvTol, 1e-9)
        # m.setParam(GRB.Param.BarQCPConvTol, 1e-9)
        # m.setParam(GRB.Param.MIPGap, 1e-9)
        # m.setParam(GRB.Param.IntFeasTol, 1e-9)
        # m.setParam(GRB.Param.PSDTol, 1e-9)
        m.setParam(GRB.Param.OptimalityTol, 1e-8)
        m.setParam(GRB.Param.DualReductions, 0)
        # m.setParam(GRB.Param.Crossover, 2)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.Method, 1)
        m.setParam(GRB.Param.BarHomogeneous, 1)
        # m.setParam(GRB.Param.ScaleFlag,0)
        if finalRun:
            m.setParam(GRB.Param.NumericFocus, 3)
        m.setParam(GRB.Param.Presolve, 0)

        diskPolyWeights = m.addVars(numGamma,numK, vtype=GRB.SEMICONT, lb=0,ub=1, name="w")
        m.addConstrs(diskPolyWeights[curGamma,curK] == 0
                     for curGamma in range(numGamma) for curK in range(numK) if not includedParamsArray[curGamma,curK])

        errorFuncWeight = m.addVar(vtype=GRB.SEMICONT, lb=0,ub=1, name="errorWeight")
        betaForbidden = m.addVar(lb=-1.0, ub=1.0, name="betaAtForbidden")
        m.addConstr(betaForbidden <= errorForbidden*errorFuncWeight)
        m.addConstr(-betaForbidden <= errorForbidden * errorFuncWeight)

        # Forbidden-IP constraint
        forbiddenInner = np.zeros((numGamma, numK))
        for g in range(numGamma):
            forbiddenInner[g,:kMaxByGamma[g]+1] = diskPolysByGamma[g].calcAtRTheta(forbiddenRadius,
                                                                                   forbiddenAngle)[:kMaxByGamma[g]+1]
        m.addConstr(gp.quicksum(forbiddenInner[curGamma, curK]*diskPolyWeights[curGamma, curK]
                                for curGamma in range(numGamma) for curK in range(numK)
                                if includedParamsArray[curGamma,curK])+betaForbidden == 0)

        # “SDP” constraint you coded (quadratic)
        valAtOne = gp.quicksum(diskPolyWeights[g,k] for g in range(numGamma)
                                    for k in range(numK) if includedParamsArray[g,k]) + errorFuncWeight
        m.addConstr(diskPolyWeights[0, 0] -
                    valAtOne * valAtOne >= 0)
        # m.addConstr(diskPolyWeights.sum() == 1)
        setWeightVars = []
        setWeightConstr = []
        # Build per–point-set pieces
        largestSet = 0
        nInit = listOfPointSets[0].shape[0]
        charM = characteristicMatrix(nInit)
        nrSets = charM.shape[1]
        for ps_idx in range(nrOfPointSets):
            pts = listOfPointSets[ps_idx]
            n = pts.shape[0]
            if charM.shape[0] != n:
                # Characteristic matrix — consider sparse (see §3)
                charM = characteristicMatrix(n)                  # shape (n, nrSets)
                nrSets = charM.shape[1]
            setW = m.addVars(nrSets, vtype=GRB.SEMICONT, lb=0,ub=1, name=f"s_{ps_idx}")

            m.addConstr(setW.sum() == 1)
            if nrSets > largestSet:
                largestSet = nrSets
            # m.addConstr(setW[0]==0)
            # IPM and disk poly values (avoid 4-D tensor; see §3)
            ipm = pts @ pts.conj().T
            r, th = z2polar(ipm)
            np.round(r, 14, out=r)
            np.round(th, 14, out=th)
            rAlmostOne = (r >= 1-1e-8)
            thAlmostZero = (np.abs(th) <= 1e-8)
            zAlmostOne = rAlmostOne*thAlmostZero
            betaVars = m.addVars(*r.shape,vtype=GRB.CONTINUOUS,lb=-1,ub=1,name=f"PS{ps_idx}_beta")
            errorTerms = superBound(r,complexDimension,kBorder,gammaBorder)
            # For each pair, avoid creating a full charVector array if it’s sparse:
            for i in range(n):
                for j in range(i+1):
                    # left: sum_{g,k} DP[g,k,i,j] * w[g,k]
                    # compute DP row-on-demand, no 4-D tensor:
                    lin = gp.LinExpr()
                    m.addConstr(betaVars[i,j] == betaVars[j,i])
                    if zAlmostOne[i,j]:
                        m.addConstr(betaVars[i,j] == errorFuncWeight)
                    else:
                        m.addConstr(betaVars[i,j] <= errorTerms[i,j] * errorFuncWeight)
                        m.addConstr(-betaVars[i, j] <= errorTerms[i, j] * errorFuncWeight)
                    lin.addTerms(1,betaVars[i,j])
                    for g in range(numGamma):
                        vals = diskPolysByGamma[g].calcAtRTheta(r[i, j], th[i, j])  # shape (kMax,)
                        for k in range(kMax):
                            if includedParamsArray[g,k]:
                                if vals[k] != 0.0:
                                    lin.addTerms(vals[k], diskPolyWeights[g, k])

                    # right: sum_{S} setW[S] * (charM[i,S] & charM[j,S])
                    # use nonzeros only
                    nz = np.flatnonzero(charM[i] * charM[j])
                    setWeightConstr.append(m.addConstr(lin == gp.quicksum(setW[s] for s in nz)))
            setWeightVars.append(setW)
        m.setObjective(valAtOne, GRB.MAXIMIZE)
        # m.setObjective(diskPolyWeights[0,0],GRB.MAXIMIZE)

        # Lower-bound loop
        lb_ok = False
        p = complexDimension - 1
        while not lb_ok:
            lb = m.addConstr(valAtOne >= (0.5)**p)
            # lb = m.addConstr(diskPolyWeights[0,0] >= (0.5) ** p)
            m.update()
            m.optimize()
            m.printQuality()  # if it gets that far


            if m.Status != GRB.OPTIMAL and m.Status != GRB.SUBOPTIMAL:
                # print(m.objBound)
                m.remove(lb)
                p += 1
            else:
                # cons = m.getConstrs()
                # vio = m.getAttr("Slack", cons)
                # for i in sorted(range(len(cons)), key=lambda i: vio[i], reverse=True)[:10]:
                #     row = m.getRow(cons[i])
                #     print(f"row {i} vio={vio[i]:.2e} rhs={cons[i].RHS:.3g}")
                # print(m.objBound)
                lb_ok = True
        # print(m.objBound)
        objVal = m.objVal
        # for setW in setWeightVars:
        #     print(sum(setWCur.X for setWCur in setW.values()))
        diskSum = sum(diskPolyWeights[g, k].X for g in range(numGamma) for k in range(numK) if includedParamsArray[g,k])
        diskWeightsArray = np.array([[diskPolyWeights[g, k].X for k in range(numK)]
                                     for g in range(numGamma)])
        adjustedDWA = diskWeightsArray / diskSum
        objBound = m.objBound
        if objBound-diskSum < 0 or objBound > 1.1 * diskSum:
            print(f"bound obj error with primal dual difference: {objBound-diskSum}")
        # print("b) return:", rss())
        return adjustedDWA, max(objBound, diskSum+errorFuncWeight.X), diskWeightsArray, errorFuncWeight.X


    finally:
        # Ensure native memory is released
        try:
            m.dispose()
        except Exception:
            pass
        env.dispose()
        # Drop big Python refs promptly
        del diskPolysByGamma, listOfPointSets
        # print("C) end:", rss())


def concatDual(forbiddenRadius, forbiddenAngle, complexDimension, kBorder, gammaBorder, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1,finalRun=False,kOnly=False):
    # print("A) before env:", rss())
    forbiddenRadius = float(forbiddenRadius)
    forbiddenAngle = float(forbiddenAngle)
    alpha = complexDimension - 2
    if kOnly:
        gammaMax = 0
        kMax = np.max(kBorder)
        gammaBorder = np.zeros(1)
        kBorder = np.ones(1)*kMax
        kSet = np.arange(kMax+1)
        gammaSet = np.zeros_like(kSet)
        includedParamsArray = np.zeros((gammaMax + 1, kMax + 1), dtype=np.bool)
        includedParamsArray[gammaSet, kSet] = True
        kMaxByGamma = np.max(np.multiply(includedParamsArray, np.arange(kMax + 1)), axis=1)
        boundsForbidden = superBound(forbiddenRadius, complexDimension, np.array([kMax+1]), np.array([0]))[0]
    else:
        kSet,gammaSet = paramSetFromBorder(kBorder, gammaBorder)
        gammaMax = np.max(gammaBorder)
        kMax = np.max(kBorder)
        includedParamsArray = np.zeros((gammaMax+1,kMax+1),dtype=np.bool)
        includedParamsArray[gammaSet, kSet] = True
        kMaxByGamma = np.max(np.multiply(includedParamsArray, np.arange(kMax + 1)), axis=1)
        boundsForbidden = superBound(forbiddenRadius, complexDimension, kBorder, gammaBorder)[0]
    forbiddenZ = polar2z(forbiddenRadius, forbiddenAngle)



    # ---- build inputs (try to keep them small / see §3) ----
    if listOfPointSets is None:
        listOfPointSets = [
            SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints, baseZ=forbiddenZ)[0]
            for _ in range(nrOfPointSets)
        ]
    else:
        nrOfPointSets = len(listOfPointSets)

    listOfBoundArrays = {}
    listOfIdentityMatrices = {}
    ipmSetList = {}
    ipmRadList = {}
    ipmThetaList = {}

    for psIdx in range(nrOfPointSets):
        curPS = listOfPointSets[psIdx]
        ipm = curPS @ curPS.conj().T
        ipmSetList[psIdx] = ipm
        ipmRad, ipmTheta = z2polar(ipm)
        ipmRadList[psIdx] = ipmRad
        ipmThetaList[psIdx] = ipmTheta
        if kOnly:
            foundBoundsCur = superBound(ipmRad, complexDimension, np.array([kMax+1]), np.array([0]))
        else:
            foundBoundsCur = superBound(ipmRad, complexDimension, kBorder, gammaBorder)
        np.fill_diagonal(foundBoundsCur,0.0)
        listOfBoundArrays[psIdx] = foundBoundsCur
        identityMatrix = np.zeros_like(foundBoundsCur)
        np.fill_diagonal(identityMatrix, 1.0)
        listOfIdentityMatrices[psIdx] = identityMatrix

    # kMaxByGamma = np.zeros(gammaMax+1)
    # for curGamma in range(gammaMax+1):
    #     kMaxByGamma[curGamma] = np.max(includedParamsArray,axis=0)
    #     kMaxByGamma[curGamma] = np.max(kSet[np.where(gammaSet == curGamma)])
    diskPolysByGamma = [Disk(alpha, g, kMaxByGamma[g]) for g in range(gammaMax+1)]

    # ---- Gurobi in short-lived env ----
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    try:
        m = gp.Model(env=env)
        m.setParam("OutputFlag", 1)
        # m.setParam(GRB.Param.FeasibilityTol, 1e-8)
        # m.setParam(GRB.Param.BarConvTol, 1e-9)
        # m.setParam(GRB.Param.BarQCPConvTol, 1e-9)
        # m.setParam(GRB.Param.MIPGap, 1e-9)
        # m.setParam(GRB.Param.IntFeasTol, 1e-9)
        # m.setParam(GRB.Param.PSDTol, 1e-9)
        # m.setParam(GRB.Param.OptimalityTol, 1e-8)
        # m.setParam(GRB.Param.DualReductions, 0)
        # m.setParam(GRB.Param.Crossover, 2)
        m.setParam(GRB.Param.NonConvex, 0)
        # m.setParam(GRB.Param.Method, 2)
        # m.setParam(GRB.Param.BarHomogeneous, 1)
        # m.setParam(GRB.Param.ScaleFlag,0)
        if finalRun:
            m.setParam(GRB.Param.NumericFocus, 3)
        m.setParam(GRB.Param.Presolve, 0)


        z1Var = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z1")
        z2Var = m.addVar(vtype=GRB.CONTINUOUS,lb=-float('inf'),  name="z2")
        z3Var = m.addVar(vtype=GRB.CONTINUOUS,lb=-float('inf'), ub=0, name="z3")


        # forbiddenMuPlus = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="forbiddenMuPlus")
        # forbiddenMuMin = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="forbiddenMuMin")
        forbiddenMuAbs = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="forbiddenMuAbs")
        forbiddenMuVal = m.addVar(vtype=GRB.CONTINUOUS,lb=-float('inf'), name="forbiddenMuVal")
        m.addConstr(forbiddenMuAbs >= -forbiddenMuVal, name=f"AbsMuForbidden")
        m.addConstr(forbiddenMuAbs >= forbiddenMuVal, name=f"AbsMuForbidden")

        # m.addConstr(forbiddenMuVal == forbiddenMuPlus - forbiddenMuMin, name=f"ValMuForbidden")
        # muPlusDictOfVars = {}
        # muMinDictOfVars = {}
        muAbsDictOfVars = {}
        muValDictOfVars = {}
        psSizes = {}
        nuVarDict = {}
        for psIdx, ipm in ipmSetList.items():
            nuVar = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"nu{psIdx}")
            n = ipm.shape[0]
            psSizes[psIdx] = n
            # muPlus = m.addVars(n, n, lb=0.0, name=f"MuPlusForSet{psIdx}")
            # muMin = m.addVars(n, n, lb=0.0, name=f"MuMinForSet{psIdx}")

            # Optional: signed value mu = muPlus - muMin
            muVal = m.addVars(n, n, lb=-float('inf'), name=f"MuValForSet{psIdx}")
            # m.addConstrs((muVal[idx1,idx2] == muPlus[idx1,idx2] - muMin[idx1,idx2]
            #               for idx1 in range(n) for idx2 in range(n)), name=f"MuSigned_{psIdx}")
            charMat = characteristicMatrix(n)
            nrOfSubsets = charMat.shape[1]
            for curSubsetIdx in range(nrOfSubsets):
                subsetPoints = np.nonzero(charMat[:,curSubsetIdx])[0]
                m.addConstr(nuVar-gp.quicksum(muVal[idx1,idx2] for idx1 in subsetPoints for idx2 in subsetPoints)>=0)

            # Absolute-with-diagonal-exception:
            # off-diag: muAbs = muPlus + muMin
            # diag:     muAbs = muPlus - muMin
            muAbs = m.addVars(n, n, lb=0, name=f"MuAbsForSet{psIdx}")
            M = np.ones((n, n))
            np.fill_diagonal(M, -1.0)  # +1 off-diag, -1 on diag
            # The trick is that this makes offdiagonals of |mu_uv| and diagonals of -mu_vv such that the
            # constraint for compensating the concatenation is z_2-<EPS(V,V),MU_abs>=z_2-eps_uv|mu_uv| +eps_vv mu_vv
            m.addConstrs((muAbs[idx1,idx2] >= muVal[idx1,idx2]
                          for idx1 in range(n) for idx2 in range(n)), name=f"AbsMix_{psIdx}")
            m.addConstrs((muAbs[idx1, idx2] >= -muVal[idx1, idx2]
                          for idx1 in range(n) for idx2 in range(n)), name=f"AbsMix_{psIdx}")

            muAbsDictOfVars[psIdx] = muAbs
            muValDictOfVars[psIdx] = muVal
            nuVarDict[psIdx] = nuVar



        # Forbidden-IP constraint
        forbiddenInner = np.zeros((gammaMax+1, kMax+1))
        for g in range(gammaMax+1):
            forbiddenInner[g,:kMaxByGamma[g]+1] = diskPolysByGamma[g].calcAtRTheta(forbiddenRadius, forbiddenAngle)


        # “SDP” constraint you coded (quadratic)
        # m.addQConstr(4*z1Var*(-z3Var) >= z2Var*z2Var, name="SDP constraint")
        m.addQConstr(z2Var * z2Var <= 4.0 * z1Var * (-z3Var),  # same constraint, convex sense
                     name="PSD_2x2")
        # k=gamma=0 constraint
        m.addConstr(z2Var+z3Var + forbiddenMuVal + gp.quicksum((muValDictOfVars[i]).sum()
                                   for i in range(nrOfPointSets))>=1)
        # Loop over gammas
        for curGamma in range(gammaMax+1):
            relevantKVals = includedParamsArray[curGamma]
            diskPolyVals = {}
            for psIdx in range(nrOfPointSets):
                diskPolyVals[psIdx] = diskPolysByGamma[curGamma].calcAtRTheta(ipmRadList[psIdx],ipmThetaList[psIdx])
            forbiddenInnerCur = forbiddenInner[curGamma]
            for curK in np.nonzero(relevantKVals)[0]:
                if curGamma == 0 and curK == 0:
                    continue
                else:
                    m.addConstr(z2Var + forbiddenInnerCur[curK]*forbiddenMuVal +
                                gp.quicksum(gp.quicksum(diskPolyVals[i][curK,idx1,idx2]*muValDictOfVars[i][idx1,idx2]
                                             for idx1 in range(psSizes[i]) for idx2 in range(psSizes[i]))
                                                                             for i in range(nrOfPointSets)) >= 1)


        # # concat constraint
        m.addConstr(z2Var - boundsForbidden * forbiddenMuAbs +
                    gp.quicksum(gp.quicksum(-listOfBoundArrays[i][idx1,idx2] * muAbsDictOfVars[i][idx1,idx2]
                                 for idx1 in range(psSizes[i])
                                 for idx2 in range(psSizes[i]) if idx1 != idx2) for i in range(nrOfPointSets))
                                + gp.quicksum(gp.quicksum(listOfIdentityMatrices[i][idx1,idx1] * muValDictOfVars[i][idx1,idx1]
                                               for idx1 in range(psSizes[i]))
                                                for i in range(nrOfPointSets)) >= 1)
        # objExpression = z1Var + gp.quicksum((muValDictOfVars[i]).sum()
        #                            for i in range(nrOfPointSets))
        objExpression = z1Var + sum(curNuVar for curNuVar in nuVarDict.values())
        m.setObjective(objExpression, GRB.MINIMIZE)

        # Lower-bound loop
        lb_ok = False
        ubVal = 0.5**(complexDimension-1)
        looseningFactor = 1.1
        tryNumber = 1
        while not lb_ok:
            # lb = m.addConstr(diskPolyWeights.sum() >= (0.5)**p)
            # ub = m.addConstr(objExpression <= ubVal*(looseningFactor**tryNumber))
            m.update()
            m.optimize()
            # m.computeIIS()
            # for c in m.getConstrs():
            #     if c.IISConstr: print(c.ConstrName)
            # for qc in m.getQConstrs():
            #     if qc.IISQConstr: print(qc.QCName)
            # for v in m.getVars():
            #     if v.IISLB or v.IISUB: print("BOUND", v.VarName, v.LB, v.UB)
            if m.Status != GRB.OPTIMAL and m.Status != GRB.SUBOPTIMAL:
                # m.remove(ub)
                tryNumber += 1
            else:
                lb_ok = True
        print(m.objBound)
        objVal = m.objVal
        return objVal

    finally:
        # Ensure native memory is released
        try:
            m.dispose()
        except Exception:
            pass
        env.dispose()
        # Drop big Python refs promptly
        del diskPolysByGamma, listOfPointSets, listOfBoundArrays, ipmSetList, ipmRadList, ipmThetaList
        # print("C) end:", rss())




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


def polyDiffBQPSetFinder(dim, lovaszCoef, indepCoef, nrOfPoints, rRes=1000,verificationBool=False):
    normLovaszCoef = lovaszCoef/np.sum(lovaszCoef)
    normIndepCoef = indepCoef/np.sum(indepCoef)
    eCoef = normLovaszCoef-normIndepCoef
    eCoefNormalized = eCoef/np.sum(np.abs(eCoef))
    significantCoefs = np.argwhere(np.abs(eCoefNormalized) > 1e-4)
    idxOfLastSignificantCoef = np.max(significantCoefs, axis=0)
    startingPolynomial = FastDiskCombiEstimator(dim - 2,
                                                eCoef[:idxOfLastSignificantCoef[0] + 1,
                                                        :idxOfLastSignificantCoef[1] + 1]
                                                , resolution=1000)

    U0 = np.random.default_rng(0).standard_normal((2 * dim, nrOfPoints)).ravel()
    rGrid = np.linspace(0,1, rRes, endpoint=True)
    factorArray = np.ones(idxOfLastSignificantCoef[0]+1)/2
    factorArray[0] = 1
    eGrid = np.sum(np.array([startingPolynomial.fastRadialEstimatorList[freIdx](rGrid)*factorArray[freIdx]
                      for freIdx in range(idxOfLastSignificantCoef[0]+1)]),axis=0)
    eSquareGrid = np.square(eGrid)
    h, lam = UnconstrainedPolyDiffOptimizer.pick_h_lambda_from_E(eSquareGrid,rGrid,gamma=5)
    res = scipy.optimize.minimize(
        UnconstrainedPolyDiffOptimizer.objective_diversity_normalize,
        U0,
        args=(dim, nrOfPoints, startingPolynomial, lam, h),  # lam=0.2, h=0.05
        method="L-BFGS-B",
        jac=None,  # let SciPy FD the gradient
        options=dict(maxiter=1000, ftol=1e-8, maxls=50)
    )

    resultingPoints = UnconstrainedPolyDiffOptimizer.Z_from_result(res.x, dim, nrOfPoints).T
    if verificationBool:
        ipmArray = resultingPoints @ np.conj(resultingPoints.T)
        radIpms, thetaIpms = z2polar(ipmArray)

        mask = np.triu(np.ones((nrOfPoints, nrOfPoints), dtype=bool), 1)
        print(f"uniqueRads:")
        print(radIpms[mask])
        radsIpFlat = radIpms[mask]
        indepDisk = OrthonormalPolyClasses.DiskCombi(dim - 2, normIndepCoef)
        lovaszDisk = OrthonormalPolyClasses.DiskCombi(dim-2,normLovaszCoef)
        indepVals = indepDisk(rGrid,0)[0]
        lovaszVals = lovaszDisk(rGrid, 0)[0]
        eInters = startingPolynomial(radsIpFlat,0)[0]
        iInters = indepDisk(radsIpFlat, 0)[0]
        lInters = lovaszDisk(radsIpFlat, 0)[0]
        fig, ax = plt.subplots()
        ax.plot(rGrid, eGrid, color=(1,0,0))
        ax.plot(rGrid, indepVals, color=(0,1,0))
        ax.plot(rGrid, lovaszVals, color=(0,0,1))
        ax.plot(radsIpFlat, eInters, 'o', color=(1,0,0))
        ax.plot(radsIpFlat, iInters, 'o', color=(0,1,0))
        ax.plot(radsIpFlat, lInters, 'o', color=(0,0,1))
        plt.show()
        x=1


    return resultingPoints



def facetInequalityBQPGammaless(dim,startingPointset, startingCoefficients,
                                startingFacet=0, maxDryRun=-1, stopEarly=True, useRadialEstimator=True,combineBool=True,
                                errorFuncCoef=0.0,scoreFacets=False):
    scoreCategories = ["Times Tested", "Bad Scores", "Times Won (delivered)", "Times Won (random)",
                       "Goals (delivered)", "Goals (random)", "Scores (delivered)", "Scores (random)", "Scores (tie)"]
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
    validIneqs, facetIneqs, betasv1 = facetReader("bqp6.dat")
    if errorFuncCoef != 0.0:
        # errorFuncComp = errorFuncCoef*np.eye(pointSetSize, dtype=np.float64)
        facetIneqTraces = np.trace(facetIneqs, axis1=-2,axis2=-1)
        betas = betasv1-errorFuncCoef*facetIneqTraces
    else:
        betas = betasv1
    recomplexify = lambda S: ((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))
    polyVals = lambda S:startingPolynomial(*(z2polar((S.reshape(shapeStartingGuess)[:dim]-
                                            1j*S.reshape(shapeStartingGuess)[dim:]).T @
                                                                    (S.reshape(shapeStartingGuess)[:dim]+
                                                                     1j*S.reshape(shapeStartingGuess)[dim:]))))

    bestFacet = 0
    bestType = "None"
    sol = startingGuess
    lastImprovement = 0
    facetIdx = startingFacet + 1
    bestFacetIdx = "0"
    scoreboard = {"delivered":0,"random":0,"tie":0, "bad":0}
    goalboard = {"delivered":0,"random":0}
    facetNr = 0
    goals = 0
    nrOfFacets = betas.shape[0]
    if validIneqs:
        facetIndices = np.arange(nrOfFacets)
        np.random.shuffle(facetIndices)
        if maxDryRun == -1:
            maxDryRun = nrOfFacets
        # relativeSizes = np.max(np.max(np.abs(facetIneqs),axis=1), axis=1)
        for facetNr in range(nrOfFacets):
            startingLocations = createRandomPointsComplex(dim,6, includeInner=False)[0]
            startingGuess = np.concat([startingLocations.T.real, startingLocations.T.imag])
            flattenedStartingGuess = startingGuess.ravel()
            # facetIdx = -nrOfFacets + facetNr + startingFacet
            facetIdx = facetIndices[facetNr]
            idxStr = str(facetIdx)
            if scoreFacets:

                facetScoreDF.at[scoreCategories[0], idxStr] += 1
            objective = lambda S: betas[facetIdx] - np.sum(
                np.multiply(facetIneqs[facetIdx],polyVals(S)))
            constraint = lambda S: np.sum(np.square(S.reshape(shapeStartingGuess)),axis=0)-1

            # relativeSizeV1 = np.sqrt(np.sum(np.abs(facetIneqs[facetIdx])))
            relativeSizeV1 = 1
            currentCutOff = -0.3 * (-1/np.log2(nrOfFacets+2)+1/np.log2(facetNr+2))

            succesBase = False
            succesRand = False
            baseObj = 100
            randObj = 100
            baseSol = startingPointset
            randSol = startingLocations
            # objv2 = improvedBQPNLP.make_objective(facetIneqs[facetIdx], shapeStartingGuess, startingPolynomial)
            # print(objective(flattenedStartingGuess)-objv2(flattenedStartingGuess)-betas[facetIdx])
            # t0 = time.perf_counter()
            # randompoints as startingpoints
            try:
                res, tau = minimizationUnconstrained.solve_facets_softmin(flattenedStartingGuess, shapeStartingGuess,
                                                                          startingPolynomial, facetIneqs, betas)
                # res = minimizationUnconstrained.run_from_flattened_start(
                #     flattenedStartingGuess, shapeStartingGuess, startingPolynomial, facetIneqs[facetIdx],
                #     unconstrained_mode="normalize",  # or "stereo"
                #     unconstrained_method="L-BFGS-B",  # or "trust-constr"
                #     workers_trust_constr=10,  # if you later set >1, keep BLAS threads = 1
                #     finite_diff_rel_step=1e-6,
                #     gtol=1e-6, xtol=1e-12, maxiter=2000, verbose=0
                # )
                # res = improvedBQPNLP.solve_problem(flattenedStartingGuess, shapeStartingGuess, startingPolynomial,
                #                                        facetIneqs, facetIdx)
                # res = scipy.optimize.minimize(
                #     objective, flattenedStartingGuess,
                #     method='trust-constr',
                #     constraints={'type': 'eq', 'fun': constraint},
                #     tol=1e-8,
                #     options={"maxiter":500}
                # )
            except Exception as e:
                print(e)
                succesRand = False
            else:
                relativeObjective = (betas[facetIdx]+res.fun) / relativeSizeV1  # / relativeSizes[facetIdx]
                # print(relativeObjective)
                randObj = relativeObjective
                if relativeObjective < bestFacet: # and res.constr_violation < 1e-8:
                    randSol = minimizationUnconstrained.Z_from_unconstrained_normalize(res.x,n=dim,m=6)

                    succesRand = True
                else:
                    succesRand = False
            # t1 = time.perf_counter()
            #
            # resImpr = minimizationUnconstrained.run_from_flattened_start(
            #     flattenedStartingGuess, shapeStartingGuess, startingPolynomial, facetIneqs[facetIdx],
            #     unconstrained_mode="normalize",       # or "stereo"
            #     unconstrained_method="L-BFGS-B",      # or "trust-constr"
            #     workers_trust_constr=10,               # if you later set >1, keep BLAS threads = 1
            #     finite_diff_rel_step=1e-6,
            #     gtol=1e-6, xtol=1e-12, maxiter=1000, verbose=0
            # )
            # t2 = time.perf_counter()
            # print(f"Old: {t1-t0} \n New:{t2-t1} \n Objectives: {res.fun+betas[facetIdx]} vs {resImpr.fun+betas[facetIdx]}")
            # if 1.3*(t1-t0)<t2-t1:
            #     x=1
            # input points as startingpoints
            # try:
            #     # res = minimizationUnconstrained.run_from_flattened_start(
            #     #     baseFlattenedStartingGuess, shapeStartingGuess, startingPolynomial, facetIneqs[facetIdx],
            #     #     unconstrained_mode="normalize",  # or "stereo"
            #     #     unconstrained_method="L-BFGS-B",  # or "trust-constr"
            #     #     workers_trust_constr=10,  # if you later set >1, keep BLAS threads = 1
            #     #     finite_diff_rel_step=1e-6,
            #     #     gtol=1e-6, xtol=1e-12, maxiter=2000, verbose=0
            #     # )
            #     res, tau = minimizationUnconstrained.solve_facets_softmin(baseFlattenedStartingGuess, shapeStartingGuess,
            #                                                               startingPolynomial, facetIneqs, betas)
            #     # res = improvedBQPNLP.solve_problem(baseFlattenedStartingGuess, shapeStartingGuess, startingPolynomial,
            #     #                                        facetIneqs, facetIdx)
            #     # res = scipy.optimize.minimize(
            #     #     objective, baseFlattenedStartingGuess,
            #     #     method='trust-constr',
            #     #     constraints={'type': 'eq', 'fun': constraint},
            #     #     tol=1e-8,
            #     #     options={"maxiter":500}
            #     # )
            # except:
            #     succesBase = False
            # else:
            #     relativeObjective = (res.fun+betas[facetIdx]) / relativeSizeV1 # / relativeSizes[facetIdx]
            #     # print(f"suc{res.x},obj{relativeObjective}")
            #     baseObj = relativeObjective
            #     if relativeObjective < bestFacet: # and res.constr_violation < 1e-8:
            #         # baseSol = res.x.reshape(shapeStartingGuess)
            #         baseSol = minimizationUnconstrained.Z_from_unconstrained_normalize(res.x,n=dim,m=6)
            #         baseObj = relativeObjective
            #         succesBase = True
            #     else:
            #         succesBase = False

            # Find best sol out of starting options
            resObj = randObj
            resSol = randSol
            succesRes = succesRand
            resType = "random"
            # resObj = baseObj
            # resSol = baseSol
            # succesRes = succesBase
            # resType = "delivered"

            if succesRand:
                if randObj < baseObj:
                    resObj = randObj
                    resSol = randSol
                    succesRes = succesRand
                    resType = "random"
            if scoreFacets:
                if baseObj > 0 and randObj > 0:
                    scoreboard["bad"] += 1
                    facetScoreDF.at[scoreCategories[1], idxStr] += 1
                elif baseObj < randObj:
                    scoreboard["delivered"] += 1
                    facetScoreDF.at[scoreCategories[6], idxStr] += 1
                elif baseObj > randObj:
                    scoreboard["random"] += 1
                    facetScoreDF.at[scoreCategories[7], idxStr] += 1
                else:
                    scoreboard["tie"] += 1
                    facetScoreDF.at[scoreCategories[8], idxStr] += 1
            # print(f"suc{succesRes},obj{resObj}")
            # Update best sol
            if succesRes:
                print(f"found {resObj} at {idxStr} (nr of facets tested: {facetNr+1})")
                lastImprovement = 0
                bestFacet = resObj
                sol = resSol
                bestType = resType
                if scoreFacets:
                    goalboard[bestType] += 1
                    if bestType == "delivered":
                        facetScoreDF.at[scoreCategories[4], idxStr] += 1
                    else:
                        facetScoreDF.at[scoreCategories[5], idxStr] += 1
                    goals += 1
                bestFacetIdx = idxStr
            else:
                lastImprovement += 1

            # Stop if criteria are met
            if bestFacet < 0 and stopEarly:
                if bestFacet < currentCutOff or lastImprovement >= maxDryRun:
                    break
    if scoreFacets:
        Scoreboard.show_dashboard(scoreboard,goalboard,facetNr+validIneqs,goals)
        print(f"Final winner: {bestType}")
        # facetScoreDF.to_csv(facetFileLoc)
        if bestFacet < 0:
            if bestType == "delivered":
                facetScoreDF.at[scoreCategories[2],bestFacetIdx] += 1
            else:
                facetScoreDF.at[scoreCategories[3], bestFacetIdx] += 1
            # facetScoreDF.to_csv(facetFileLoc)
            return True, sol.T, (facetIdx+1) % nrOfFacets
    elif bestFacet < 0:
        return True, sol.T, (facetIdx + 1) % nrOfFacets
    else:
        # facetScoreDF.to_csv(facetFileLoc)
        return False, startingGuess.T, (facetIdx+1) % nrOfFacets




def finalBQPMethod(dim, maxDeg=0, maxGamma=0, forbiddenRad=0, forbiddenTheta=0,
                    sizeOfBQPSet=10, setAmount=10, setLinks=1,
                    sequentialEdges=False, uniformCoordinateWise=False, reverseWeighted=False, spreadPoints=False,
                    compareToLowerBound=False, trulyUniformOnSphere=False, improvementWithFacets = False,
                    relativityScale=-1, basePolyCoefs=None,
                    printBool=False,inbetweenSetRun=False):
    # Make sure all correct information is available
    if basePolyCoefs is None and compareToLowerBound:
        print("Using double cap as base poly")
        basePolyCoefs = np.zeros((maxGamma+1,maxDeg+1))
        basePolyCoefs[0] = c_coeffs(dim,maxDeg)
    if maxDeg == 0:
        maxDeg = 4 * (dim - 1)
    if reverseWeighted:
        if relativityScale == -1:
            relativityScale = (maxDeg - 1) / maxDeg
    if spreadPoints:
        innerProductList = np.array([0, 1])
    else:
        innerProductList = []
    implementedBool = False

    # Create radial measures and their densities for each dimension
    weightPolyList = [complexWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    weightIntsList = [integratedWeightPolyCreator(dim - depth) for depth in range(dim - 1)]



    # Start making point sets for simplest set with inner product t
    pointSet1 = np.zeros((2, dim), dtype=np.complex128)
    pointSet1[0, 0] = 1
    pointSet1[1, 1] = polar2z(forbiddenRad, -forbiddenTheta)
    pointSet1[1, 1] = np.sqrt(1 - forbiddenRad ** 2)

    # Run once to get a baseline
    coefsThetav0, bestObjVal, unscaledcoefsThetav0 = finalBQPModelv2(forbiddenRad,
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
        elif compareToLowerBound and implementedBool:
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

        elif compareToLowerBound and not implementedBool:
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

        # if setLinks == 1:
        #     pointSetsForModel = inputPointSets
        # elif len(fullPointSets) < setLinks:
        #     if improvementWithFacets:
        #         pointSetsForModel = [np.concat(inputPointSets)]
        #     else:
        #         pointSetsForModel = [np.concat([inputPointSets[0][:2]]+[inputPointSets[setIdx][2:]
        #                                                                 for setIdx in range(len(inputPointSets))])]
        # else:
        #     pointSetsForModel = []
        #     if improvementWithFacets:
        #         for comb in combinations(range(len(inputPointSets)), setLinks):
        #             pointSetsForModel.append(np.concat([inputPointSets[combIdx] for combIdx in comb]))
        #     else:
        #         for comb in combinations(range(len(inputPointSets)), setLinks):
        #             pointSetsForModel.append(np.concat([inputPointSets[0][:2]]+
        #                                                [inputPointSets[combIdx][2:] for combIdx in comb]))
        # Run model with new BQP sets if set is full or you want to run with non full sets
        if inbetweenSetRun or listOfPointSets[iterationNr + 1].shape[0] == sizeOfBQPSet:
            if compareToLowerBound and listOfPointSets[iterationNr + 1].shape[0] == sizeOfBQPSet:
                newPointset = polyDiffBQPSetFinder(dim,coefThetaList[-1],basePolyCoefs, sizeOfBQPSet)
                listOfPointSets[iterationNr + 1] = newPointset
            inputPointSets = [listOfPointSets[iterationNr + 1]] + fullPointSets
            pointSetsForModel = stitchSets(inputPointSets, setLinks, improvementWithFacets)
            (coefsCurTheta, curObjVal,
             unscaledCoefsCurTheta) = finalBQPModelv2(forbiddenRad, forbiddenTheta, dim,
                                                    maxGamma, maxDeg,
                                                    listOfPointSets=pointSetsForModel)
        # Else copy the last vals
        else:
            (coefsCurTheta, curObjVal,
             unscaledCoefsCurTheta) = (coefThetaList[-1],bestObjVal,unscaledCoefThetaList[-1])
        # If the set is full and we want to improve with NLP, then improve the pointset and run it again to verify
        # the program found an improvement
        if improvementWithFacets:
            if listOfPointSets[iterationNr + 1].shape[0] == 6:
                print(f"Before face optimization obj {curObjVal}")
                (foundBool, ineqPointset,
                 facetStart) = facetInequalityBQPGammaless(dim, listOfPointSets[iterationNr + 1],
                                                           unscaledCoefsCurTheta,
                                                           startingFacet=facetStart, stopEarly=True)

                if foundBool:
                    inputPointSets = [ineqPointset] + fullPointSets
                    pointSetsForModel = stitchSets(inputPointSets, setLinks)
                    # if setLinks == 1:
                    #     pointSetsForModel = inputPointSets
                    # elif len(fullPointSets) < setLinks:
                    #     pointSetsForModel = [np.concat(inputPointSets)]
                    # else:
                    #     pointSetsForModel = []
                    #     for comb in combinations(range(len(inputPointSets)), setLinks):
                    #         pointSetsForModel.append(np.concat([inputPointSets[combIdx] for combIdx in comb]))
                    (coefsFacetTheta, facetObjVal,
                     unscaledCoefsFacetTheta) = finalBQPModelv2(forbiddenRad, forbiddenTheta,
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


def pointSetFromThetaSolAndDC(dim,coefsCurTheta,sizeOfBQPSet):
    thetaDisk = DiskCombi(dim - 2, coefsCurTheta)
    nrOfCoefs = coefsCurTheta.shape[1]
    doubleCapCoefs = c_coeffs(dim, nrOfCoefs - 1).reshape((1, nrOfCoefs))
    dcDisk = DiskCombi(dim - 2, doubleCapCoefs)

    radiusResolution = 10000
    radsArray = np.linspace(0, 1, radiusResolution, endpoint=True)
    thetasArray = np.zeros_like(radsArray)
    thetaVals = thetaDisk(radsArray, thetasArray)
    absTheta = np.abs(thetaVals)
    dcVals = dcDisk(radsArray, thetasArray)
    absDC = np.abs(dcVals)

    thetaLarger = np.greater(absTheta, absDC)
    DCLarger = np.greater(absDC, absTheta)

    diffVals = np.zeros_like(radsArray)
    np.divide((absDC - absTheta), absTheta, out=diffVals, where=thetaLarger)
    np.divide((absTheta - absDC), absDC, out=diffVals, where=DCLarger)

    intedDiffs = np.cumsum(diffVals)
    normalizedIntedDiffs = intedDiffs / intedDiffs[-1]
    initPS = np.zeros((dim, sizeOfBQPSet), dtype=np.float64)
    initPS[0, 0] = 1.0
    radRealizations = np.random.random((dim - 1, sizeOfBQPSet))
    thetaRealizations = np.random.random((dim, sizeOfBQPSet))
    for pointIdx in range(1, sizeOfBQPSet):
        finalCoordinateIdx = min(pointIdx, dim - 1)
        remainingRad = 1.0
        squaredRemainingRad = 1.0
        # at every step, select the cheapest (wrt radius) theta to achieve the inner product
        for curCoordinate in range(finalCoordinateIdx):
            if remainingRad <= 0:
                break
            ipmBase = np.vdot(initPS[curCoordinate], initPS[pointIdx])
            baseRad, baseTheta = z2polar(ipmBase)
            curRadRealization = radRealizations[pointIdx, curCoordinate]

            relevantCoordinate = initPS[curCoordinate, curCoordinate]
            relevantRad, relevantTheta = z2polar(relevantCoordinate)
            if relevantRad == 0:
                foundRad = curRadRealization
                foundTheta = thetaRealizations[pointIdx, curCoordinate]
            else:
                maxInfluence = remainingRad * relevantRad
                minimalResultingRad = max(0.0, baseRad - maxInfluence)
                leftBoundIdx = np.min(np.argwhere(radsArray >= minimalResultingRad))
                leftBoundVal = normalizedIntedDiffs[leftBoundIdx]
                maximalResultingRad = min(1.0, baseRad + maxInfluence)
                rightBoundIdx = np.max(np.argwhere(radsArray <= maximalResultingRad))
                rightBoundVal = normalizedIntedDiffs[rightBoundIdx]

                intFactor = rightBoundVal - leftBoundVal
                desiredInt = intFactor * curRadRealization + leftBoundVal
                desiredRadIdxL = np.max(np.argwhere(normalizedIntedDiffs <= desiredInt), initial=0)
                if desiredRadIdxL < radiusResolution - 1:
                    lRadVal = normalizedIntedDiffs[desiredRadIdxL]
                    rRadVal = normalizedIntedDiffs[desiredRadIdxL + 1]
                    leftFactor = (desiredInt - lRadVal) / (rRadVal - lRadVal)
                    desiredIPMRad = leftFactor * (radsArray[desiredRadIdxL]) + (1 - leftFactor)
                else:
                    desiredIPMRad = 1.0
                desiredRad = (desiredIPMRad - baseRad) / relevantRad
                if -1 <= desiredRad < 0:
                    foundRad = -desiredRad
                    foundTheta = baseTheta + relevantTheta + np.pi
                elif 0 <= desiredRad <= 1:
                    foundRad = desiredRad
                    foundTheta = baseTheta + relevantTheta
                elif desiredRad < -1:
                    foundRad = 1
                    foundTheta = baseTheta + relevantTheta + np.pi
                    print("wanted to dedicate to much negative radius, aka error, assuming radFactor = -1")
                else:
                    foundRad = 1
                    foundTheta = baseTheta + relevantTheta
                    print("wanted to dedicate to much radius, aka error, assuming radFactor = 1")

            # foundTheta = 0.0
            initPS[pointIdx, finalCoordinateIdx] = polar2z(foundRad, foundTheta)
            squaredRemainingRad -= np.square(foundRad)
            remainingRad = np.sqrt(squaredRemainingRad)
    startingPS = initPS.dot(unitary_group.rvs(dim))
    return startingPS


def primalStartDualFinish(dim, forbiddenRad=0, forbiddenTheta=0, eps=0.001,
                            sizeOfBQPSet=6, setAmount=5, setLinks=1,
                            uniformCoordinateWise=False,
                            compareToLowerBound=False, trulyUniformOnSphere=False, improvementWithFacets=False):
    # Make sure all correct information is available
    if forbiddenRad == 0.0:
        allKOnly = True
    else:
        allKOnly = False
        if compareToLowerBound:
            raise Exception("Compare to lower bound only currently implemented for t=0")
    # Create radial measures and their densities for each dimension
    # weightPolyList = [complexWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    # weightIntsList = [integratedWeightPolyCreator(dim - depth) for depth in range(dim - 1)]
    forbiddenZ = polar2z(forbiddenRad, -forbiddenTheta)


    # Start making point sets for simplest set with inner product t
    pointSet1 = np.zeros((2, dim), dtype=np.complex128)
    pointSet1[0, 0] = 1
    pointSet1[1, 1] = forbiddenZ
    pointSet1[1, 1] = np.sqrt(1 - forbiddenRad ** 2)
    pointSetsForModel = [pointSet1]

    # Run once to get a baseline
    if forbiddenRad < 1:
        curKBorder,curGammaBorder = borderBasedOnEpsilon(eps,dim,np.array([forbiddenZ]),allKOnly)
    else:
        curKBorder,curGammaBorder = borderBasedOnEpsilon(eps, dim, np.array([0.99]), allKOnly)
    (coefsThetav0, bestObjVal,
     unscaledcoefsThetav0,curErrorTerm) = borderedPrimal(forbiddenRad, forbiddenTheta, dim, curGammaBorder, curKBorder,
                                                                    listOfPointSets=pointSetsForModel,kOnly=allKOnly)
    # Disk poly from baseline
    # diskPolyBQPLess = DiskCombi(dim-2,unscaledcoefsThetav0)

    # Initialize the lists for BQP sets, disk polynomials and their coefficients in both normalizations
    fullPointSets = []
    coefThetaList = [coefsThetav0]
    unscaledCoefThetaList = [unscaledcoefsThetav0]

    # Set counters to start
    facetStart = 0
    setNr = 0
    curObjVal = bestObjVal
    unscaledCoefsCurTheta = unscaledcoefsThetav0
    coefsCurTheta = coefsThetav0
    for setIdx in range(setAmount):
        if uniformCoordinateWise:
            radRealizations = np.random.random((dim-1,sizeOfBQPSet))
            thetaRealizations = np.random.random((dim,sizeOfBQPSet))*2*np.pi
            remainingRadiusArray = np.ones(sizeOfBQPSet, dtype=np.float64)
            squaredRemainingRadiusArray = np.ones_like(remainingRadiusArray)
            startingPSRads = np.zeros((dim,sizeOfBQPSet), dtype=np.float64)
            for coordinateIdx in range(dim-1):
                curRadRealizations = radRealizations[coordinateIdx]
                curRad = np.multiply(curRadRealizations,remainingRadiusArray)
                startingPSRads[coordinateIdx] = curRad
                squaredRemainingRadiusArray -= np.square(curRad)
                remainingRadiusArray = np.sqrt(squaredRemainingRadiusArray)
            startingPSRads[dim-1] = remainingRadiusArray
            startingPS = polar2z(startingPSRads,thetaRealizations)
        elif compareToLowerBound:
            startingPS = pointSetFromThetaSolAndDC(dim,coefsCurTheta, sizeOfBQPSet)
        elif trulyUniformOnSphere:
            initPS = SphereBasics.createRandomPointsComplex(dim,sizeOfBQPSet,includeInner=False)[0]
            startingPS = initPS.dot(unitary_group.rvs(dim))
        else:
            # if none were true, also generate points uniformly random according to surface measure
            initPS = SphereBasics.createRandomPointsComplex(dim, sizeOfBQPSet, includeInner=False)[0]
            startingPS = initPS.dot(unitary_group.rvs(dim))
        if improvementWithFacets:
            (foundBool, ineqPointset,
             facetStart) = facetInequalityBQPGammaless(dim, startingPS,
                                                       unscaledCoefsCurTheta, errorFuncCoef=curErrorTerm,
                                                       startingFacet=facetStart, stopEarly=True)
            if foundBool:
                psForModel = ineqPointset
            else:
                psForModel = startingPS
        else:
            psForModel = startingPS
        fullPointSets.append(psForModel)
        pointSetsForModel = stitchSets(fullPointSets, setLinks)
        ipmsForModel = [curPS.dot(curPS.conj().T) for curPS in pointSetsForModel]
        uniqueValueArray = np.concat(ipmsForModel, axis=None)
        curKBorder,curGammaBorder  = borderBasedOnEpsilon(eps, dim, uniqueValueArray, allKOnly)
        (coefsCurTheta, curObjVal,
         unscaledCoefsCurTheta, curErrorTerm) = borderedPrimal(forbiddenRad, forbiddenTheta, dim,
                                                               curGammaBorder, curKBorder,
                                                                listOfPointSets=pointSetsForModel, kOnly=allKOnly)
        if curObjVal > bestObjVal:
            print("Error, problem got worse with more sets")
        else:
            bestObjVal = curObjVal
        coefThetaList.append(coefsCurTheta)
        unscaledCoefThetaList.append(unscaledCoefsCurTheta)
    print("Primal found all pointsets, running Dual for final value")
    finalObj = concatDual(forbiddenRad,forbiddenTheta,dim,
                          curKBorder,curGammaBorder,listOfPointSets=pointSetsForModel,kOnly=allKOnly)


    inputParameters = {"dim": dim, "eps": eps,
                       "forbiddenRad":forbiddenRad,"forbiddenTheta":forbiddenTheta,
                       "sizeOfBQPSet":sizeOfBQPSet, "setAmount":setAmount, "setLinks":setLinks,
                       "improvementWithFacets":improvementWithFacets,
                       "uniformCoordinateWise": uniformCoordinateWise,
                       "compareToLowerBound":compareToLowerBound}
    # usedPointSets = fullPointSets + [listOfPointSets[-1]]
    usedPointSetDict = {f"point set {setNr}": fullPointSets[setNr].tolist() for setNr in range(len(fullPointSets))}
    # coefsThetaListsList = [coefArray.tolist() for coefArray in coefThetaList]
    # unscaledCoefsThetaListsList = [coefArray.tolist() for coefArray in unscaledCoefThetaList]
    outputDict = {"objective": finalObj,
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




def facetTest(facetArray,betaArray,N,scaleBool=False):
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
    if scaleBool:
        residualMatrix = betaArray[:,None]-resultMatrix
        maxResiduals = np.max(residualMatrix, axis=1)
        if np.any(maxResiduals<=0):
            print("No slack found")
            facetsScaled = facetArray
            betasScaled = betaArray
        else:
            facetsScaled = facetArray/maxResiduals[:,None,None]
            betasScaled = betaArray/maxResiduals
        return boolArray, invalidArray, facetsScaled, betasScaled
    else:
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
    validArray, invalidArray,scaledIneqs, scaledBetas = facetTest(ineqArray, betaArray,NGlobal,True)

    if np.any(invalidArray):
        print(f"{np.where(invalidArray)} are invalid facets")
        return False, scaledIneqs, scaledBetas
    else:
        return True, scaledIneqs, scaledBetas

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


def strToComp(s):
    m = _RX.match(s)
    if not m:
        return s  # leave non-complex strings untouched
    real = float(m.group(1))
    sign = 1.0 if m.group(2) == '+' else -1.0
    imag = float(m.group(3)) * sign
    return complex(real, imag)

def listToCompArray(x):
    if isinstance(x, dict):
        return {curKey:listToCompArray(curVal) for curKey, curVal in x.items()}
    if isinstance(x, list):
        return np.array([listToCompArray(v) for v in x], dtype=complex)
    if isinstance(x, str):
        return strToComp(s=x)
    return x
#
# def json_to_complex_ndarray(json_text):
#     data = json.loads(json_text)
#     nested = _walk_convert(data)
#     return np.array(nested, dtype=complex)


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


def runTestsv2(testComplexDimension, forbiddenRadius, forbiddenAngle, testAmount, argsTest, testType):
    testSaveLocation = f"TestResults_{testType}_" + datetime.datetime.now().strftime("%d_%m_%H-%M") + ".json"

    with open(testSaveLocation, "w") as f:
        f.write('{\n')  # open JSON object
        for testNr in range(testAmount):
            testResult = finalBQPMethod(
                testComplexDimension,
                forbiddenRad=forbiddenRadius,
                forbiddenTheta=forbiddenAngle,
                **argsTest
            )
            if testNr:
                f.write(',\n')
            f.write(f'  "TestNr{testNr}": ')
            json.dump(testResult, f, default=complex_to_json_safe)  # write just the value
            f.flush()  # keep buffers small
            del testResult
        f.write('\n}\n')  # close JSON object
    return


def runTestsPD(testComplexDimension, forbiddenRadius, forbiddenAngle, testAmount, argsTest, testType):
    testSaveLocation = f"TestResults_{testType}_" + datetime.datetime.now().strftime("%d_%m_%H-%M") + ".json"
    if not os.path.exists(testSaveLocation):
        with open(testSaveLocation, "w") as f:
            json.dump({}, f)
    for testNr in range(testAmount):
        testResult = primalStartDualFinish(testComplexDimension,
                                    forbiddenRad=forbiddenRadius, forbiddenTheta=forbiddenAngle,
                                    **argsTest)
        testResultDict = {f"TestNr{testNr}": testResult}
        save_test_result(testResultDict, testSaveLocation)
    return


def readPointsetAndRunModelWith(fileLocation,testKey, overridingSetting=None):
    if overridingSetting is None:
        overridingSetting = {}
    file = open(fileLocation)
    dataDict = json.load(file)[testKey]
    del dataDict["coeficients disk polynomials list"], dataDict["unscaled coeficients"]
    pointsets = listToCompArray(dataDict["used point sets"])
    inputParams = dataDict["input parameters"]
    inputParams.update(overridingSetting)
    listOfStichedSets = stitchSets(list(pointsets.values()), inputParams["setLinks"])
    # inputParams = dataDict["input parameters"]
    # inputParams.update(overridingSetting)
    finalBQPModelv2(inputParams["forbiddenRad"],
                    inputParams["forbiddenTheta"],
                    inputParams["dim"],
                    inputParams["maxGamma"],
                    inputParams["maxDeg"],
                    listOfStichedSets,
                    compensateConcat=True)
    yeah = "yeah"


def readPointsetAndRunDualWith(fileLocation,testKey, eps=0.01,overridingSetting=None):
    if overridingSetting is None:
        overridingSetting = {}
    file = open(fileLocation)
    dataDict = json.load(file)[testKey]
    del dataDict["coeficients disk polynomials list"], dataDict["unscaled coeficients"]
    pointsets = listToCompArray(dataDict["used point sets"])
    inputParams = dataDict["input parameters"]
    pointsetsList = [curPs for curPs in pointsets.values()][:3]
    pointsets = {curPsIdx: pointsetsList[curPsIdx] for curPsIdx in range(3)}
    maxGammaRead = inputParams["maxGamma"]
    inputRad = inputParams["forbiddenRad"]
    inputTheta = inputParams["forbiddenTheta"]
    inputZ = polar2z(inputRad,inputTheta)
    inputDim = inputParams["dim"]
    if maxGammaRead == 0:
        kOnlyInput = True
    else:
        kOnlyInput = False
    inputIpms = [curPs @ np.conj(curPs).T for curPs in pointsets.values()] + [np.array(inputZ)]
    allInputIpms = np.concatenate(inputIpms, axis=None)
    inputKBorder, inputGammaBorder = borderBasedOnEpsilon(eps, inputDim,allInputIpms, kOnly=kOnlyInput)
    inputParams.update(overridingSetting)
    listOfStichedSets = stitchSets(list(pointsets.values()), inputParams["setLinks"])
    # inputParams = dataDict["input parameters"]
    # inputParams.update(overridingSetting)
    concatDual(forbiddenRadius=inputRad,
                    forbiddenAngle=inputTheta,
                    complexDimension=inputDim,
                    kBorder=inputKBorder,
                    gammaBorder=inputGammaBorder,
                    listOfPointSets=listOfStichedSets,
                    finalRun=False,
                    kOnly=kOnlyInput)
    yeah = "yeah"



def readOrInitFacetScores():
    path = Path(facetFileLoc)
    global facetScoreDF
    if not path.exists():
        facetIndices = np.arange(428)
        scoreCategories = ["Times Tested","Bad Scores","Times Won (delivered)","Times Won (random)",
                           "Goals (delivered)","Goals (random)","Scores (delivered)","Scores (random)", "Scores (tie)"]
        vals = np.zeros((len(scoreCategories), 428))
        facetScoreDF = pd.DataFrame(vals, columns=facetIndices, index=scoreCategories)
        facetScoreDF.to_csv(facetFileLoc)
    else:
        facetScoreDF = pd.read_csv(path, header=0, index_col=0)

def borderMaker(lambdaFunc, bigN, verificationFunc=None):
    gammaStop = 0
    searchingBool = True
    while searchingBool:
        if lambdaFunc(0,gammaStop) <= bigN:
            gammaStop += 1
        else:
            searchingBool = False
            gammaStop += 1
    gammaArray = np.arange(gammaStop, dtype=int)
    kStop = 0
    searchingBool = True
    while searchingBool:
        if lambdaFunc(kStop, 0) <= bigN:
            kStop += 1
        else:
            searchingBool = False
            kStop += 1
    kArray = np.arange(kStop, dtype=int)

    kMesh, gammaMesh = np.meshgrid(kArray,gammaArray)
    funcVals = lambdaFunc(kMesh, gammaMesh)
    included = (funcVals <= bigN)
    print(np.sum(included))
    # maxStop = min(kStop, gammaStop)
    if kStop <= gammaStop:
        kBorder = np.arange(kStop, dtype=int)
        # gammaBorder = np.zeros(kStop, dtype=int)
        gammaIdxs = gammaMesh.copy()
        gammaIdxs[included] = gammaStop
        gammaBorder = np.min(gammaIdxs,axis=0)
    else:
        gammaBorder = np.arange(gammaStop, dtype=int)
        # gammaBorder = np.zeros(kStop, dtype=int)
        kIdxs = kMesh.copy()
        kIdxs[included] = kStop
        kBorder = np.min(kIdxs,axis=1)

    if verificationFunc is not None:
        kLinSpace = np.linspace(0,kStop,5001,endpoint=True)
        gammaVals = verificationFunc(kLinSpace)
        fig, ax = plt.subplots()
        ax.plot(kLinSpace, gammaVals, color=(1,0,0))
        ax.plot(kBorder, gammaBorder, 'o', color=(0,0,1))
        plt.show()
    x=1
    return kBorder, gammaBorder


def bigBounds(radiusArray, comDim, kArrayShaped, gammaArrayShaped,perParamaterBool=False):
    radiusArray = np.atleast_1d(radiusArray)
    radiusShape = radiusArray.shape
    flatRad = radiusArray.ravel()
    epsArray = np.zeros_like(flatRad)
    kArrayShaped = np.atleast_1d(kArrayShaped)
    gammaArrayShaped = np.atleast_1d(gammaArrayShaped)
    paramShape = kArrayShaped.shape
    kArray = kArrayShaped.ravel()
    gammaArray = gammaArrayShaped.ravel()
    onesMask = (flatRad == 1)
    zerosMask = (flatRad == 0)
    # Handle ones
    epsArray[onesMask] = 1
    # Handle non extrema
    boundMask = (1 > flatRad > 0)
    boundRadii = flatRad[boundMask]
    boundSqrRem = (1-np.square(boundRadii))
    radiusDenom = np.power(boundSqrRem,1-comDim/2)
    strongRadiusDenom = np.power(4*np.square(boundRadii)*boundSqrRem,-1/4) * radiusDenom
    lg = (loggamma(comDim - 1 + kArray) + loggamma(comDim - 1 + kArray + gammaArray) - 2*loggamma(comDim - 1)
          - loggamma(kArray + 1) - loggamma(kArray + gammaArray + 1))/2
    poch_ratio = np.exp(lg)
    convBoundFac = 12/np.power(2*kArray + gammaArray + comDim - 1, 1/4)
    startBoundFac = np.power(((kArray+1)*(kArray+gammaArray+comDim-1))/((kArray+comDim-1)*(kArray+gammaArray+1)),1/4)

    # convBoundRadless = convBoundFac/(poch_ratio)  # * radiusDenom * strongRadiusDenom)
    # convBound = np.outer(convBoundRadless, strongRadiusDenom)
    # startBoundRadless = startBoundFac/(poch_ratio) # * radiusDenom)
    # startBound = np.outer(startBoundRadless, radiusDenom)

    convBoundRadless = convBoundFac / (poch_ratio)  # * radiusDenom * strongRadiusDenom)
    convBound = np.clip(np.outer(strongRadiusDenom, convBoundRadless),0,1)
    startBoundRadless = startBoundFac / (poch_ratio)  # * radiusDenom)
    startBound = np.clip(np.outer(radiusDenom, startBoundRadless),0,1)
    bestBound = np.minimum(convBound, startBound)
    superBound = np.max(bestBound, axis=1)
    epsArray[boundMask] = superBound
    # Handle zeros
    zeroGammas = (gammaArray == 0)
    boundZeroArray = zeroGammas/poch_ratio
    boundZero = np.max(boundZeroArray)
    epsArray[zerosMask] = boundZero
    epsMatrix = epsArray.reshape(radiusShape)
    if perParamaterBool:
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None, :]
        epsOutput[boundMask] = bestBound
        epsOutputRes = np.reshape(epsOutput, radiusShape + paramShape)
        return epsOutputRes
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None,:]
        epsOutput[boundMask] = bestBound
        epsOutputRes = np.reshape(epsOutput,radiusShape+paramaterSize)
        return epsOutputRes
    else:
        return epsMatrix


def stupidBigN(k_arr, r_arr):

    k = np.asarray(k_arr, dtype=int).ravel()
    r = np.asarray(r_arr, dtype=float).ravel()
    n, m = r.size, k.size

    # Harmonic numbers H_k
    j = np.arange(1, k.max()+1, dtype=float)
    H = np.zeros_like(k, dtype=float)
    for idx, kk in enumerate(k):
        H[idx] = np.sum(1.0/j[:kk]) if kk > 0 else 0.0

    # Broadcast masks
    kMesh,radMesh = np.meshgrid(k,r)
    hMesh,radMesh = np.meshgrid(H,r)
    # R = r[:, None]          # (n,1)
    # K = k[None, :]          # (1,m)
    # Hk = H[None, :]         # (1,m)

    # Existence mask
    exist = (kMesh >= 1) & (radMesh >= np.exp(-hMesh)) & (radMesh < 1.0)

    # Safe upper bound gamma'
    gamma_up = np.zeros((n, m), dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_up[exist] = (radMesh[exist] * (2*kMesh[exist] + 1.0) - 1.0) / (2.0 * (1.0 - radMesh[exist]))
    gamma_up = np.clip(gamma_up, 0.0, np.inf)

    ceil_upper = np.where(exist, np.ceil(gamma_up).astype(int), 0)

    return ceil_upper


def stupidBound(radiusArray,comDim,kArray,gammaArray,perParamaterBool=False):
    radiusArray = np.atleast_1d(radiusArray)
    radiusShape = radiusArray.shape
    flatRad = radiusArray.ravel()
    epsArray = np.zeros_like(flatRad)
    kArray = np.atleast_1d(kArray)
    gammaArray = np.atleast_1d(gammaArray)

    onesMask = (flatRad == 1)
    zerosMask = (flatRad == 0)
    # Handle ones
    epsArray[onesMask] = 1
    # Handle zeros
    zeroGammas = (gammaArray == 0)
    lg = (-loggamma(comDim - 1 + kArray) + loggamma(kArray + 1)
          + loggamma(comDim - 1))
    valAt0 = np.exp(lg)
    boundZeroArray = zeroGammas * valAt0
    boundZero = np.max(boundZeroArray)
    epsArray[zerosMask] = boundZero
    # Handle non extrema
    boundMask = (1 > flatRad > 0)
    boundRadii = flatRad[boundMask]
    gammaMesh, boundRadMesh = np.meshgrid(gammaArray,boundRadii)
    kMesh, boundRadMesh = np.meshgrid(kArray,boundRadii)
    # find M such that Jacobi<= (k+M choose k)/(k+n-2 choose k)
    bigNgamma = stupidBigN(kArray,boundRadii)
    comDimMesh = (comDim-2)*np.ones_like(bigNgamma)
    gammaEvalMesh = np.maximum(gammaMesh,bigNgamma)
    gammaPochMesh = np.maximum(gammaEvalMesh, comDimMesh)

    # calc r^{M}* max(P_k^{n-2,M}(-1), P_k^{n-2,M}(1))
    lg = (-loggamma(comDim - 1 + kMesh) + loggamma(kMesh + gammaPochMesh + 1) + loggamma(comDim - 1)
          - loggamma(gammaPochMesh + 1))
    poch_ratio = np.exp(lg)
    radPowered = np.power(boundRadMesh,gammaEvalMesh)
    foundBounds = poch_ratio*radPowered
    epsArray[boundMask] = np.max(foundBounds,axis=0)

    # boundSqrRem = (1 - np.square(boundRadii))
    # radiusDenom = np.power(boundSqrRem, comDim / 2 - 1)
    epsMatrix = epsArray.reshape(radiusShape)
    if perParamaterBool:
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None,:]
        epsOutput[boundMask] = foundBounds
        epsOutputRes = np.reshape(epsOutput, radiusShape + paramaterSize)
        return epsOutputRes
    else:
        return epsMatrix


def kOneBound(radiusArray,comDim,kArray,gammaArray, perParamaterBool=False):
    radiusArray = np.atleast_1d(radiusArray)
    radiusShape = radiusArray.shape
    flatRad = radiusArray.ravel()
    epsArray = np.zeros_like(flatRad)
    kArray = np.atleast_1d(kArray)
    gammaArray = np.atleast_1d(gammaArray)
    ParamMask = (kArray==1)
    IgnoreMask = (kArray!=1)

    onesMask = (flatRad == 1)
    zerosMask = (flatRad == 0)
    # Handle ones
    epsArray[onesMask] = 1
    # Handle zeros
    zeroGammas = (gammaArray == 0)
    lg = (-loggamma(comDim - 1 + kArray) + loggamma(kArray + 1)
          + loggamma(comDim - 1))
    valAt0 = np.exp(lg)
    boundZeroArray = zeroGammas * valAt0
    boundZero = np.max(boundZeroArray)
    epsArray[zerosMask] = boundZero
    # Handle non extrema
    boundMask = (1 > flatRad > 0)
    boundRadii = flatRad[boundMask]
    squaredRadii = np.square(boundRadii)
    gammaMesh, boundRadMesh = np.meshgrid(gammaArray, boundRadii)
    kMesh, boundSquaredRadMesh = np.meshgrid(kArray, squaredRadii)
    # find M such that Jacobi<= (k+M choose k)/(k+n-2 choose k)
    bigNgamma = -(1/np.log(boundSquaredRadMesh)+1+((comDim-1)*boundSquaredRadMesh)/(1-boundSquaredRadMesh))
    # comDimMesh = (comDim - 2) * np.ones_like(bigNgamma)
    gammaEvalMesh = np.maximum(gammaMesh, bigNgamma)
    # gammaPochMesh = np.maximum(gammaEvalMesh, comDimMesh)

    # calc r^{M}* ( ((M+1)/(n-1)-1)) (1-r^2) +1)
    jacobiBound = boundSquaredRadMesh + (1-boundSquaredRadMesh)*((gammaEvalMesh+1)/(comDim-1))
    radPowered = np.power(boundRadMesh, gammaEvalMesh)
    foundBounds = np.ones_like(boundRadMesh)
    foundBounds[:,ParamMask] = (jacobiBound * radPowered)[:,ParamMask]
    epsArray[boundMask] = np.max(foundBounds, axis=0)

    # boundSqrRem = (1 - np.square(boundRadii))
    # radiusDenom = np.power(boundSqrRem, comDim / 2 - 1)
    epsMatrix = epsArray.reshape(radiusShape)
    if perParamaterBool:
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None, :]
        epsOutput[boundMask] = foundBounds
        epsOutputRes = np.reshape(epsOutput, radiusShape + paramaterSize)
        return epsOutputRes
    else:
        return epsMatrix


def kTwoBound(radiusArray,comDim,kArray,gammaArray, perParamaterBool=False):
    radiusArray = np.atleast_1d(radiusArray)
    radiusShape = radiusArray.shape
    flatRad = radiusArray.ravel()
    epsArray = np.zeros_like(flatRad)
    kArray = np.atleast_1d(kArray)
    gammaArray = np.atleast_1d(gammaArray)
    ParamMask = (kArray==2)
    IgnoreMask = (kArray!=2)

    onesMask = (flatRad == 1)
    zerosMask = (flatRad == 0)
    # Handle ones
    epsArray[onesMask] = 1
    # Handle zeros
    zeroGammas = (gammaArray == 0)
    lg = (-loggamma(comDim - 1 + kArray) + loggamma(kArray + 1)
          + loggamma(comDim - 1))
    valAt0 = np.exp(lg)
    boundZeroArray = zeroGammas * valAt0
    boundZero = np.max(boundZeroArray)
    epsArray[zerosMask] = boundZero
    # Handle non extrema
    boundMask = (1 > flatRad > 0)
    boundRadii = flatRad[boundMask]
    squaredRadii = np.square(boundRadii)
    gammaMesh, boundRadMesh = np.meshgrid(gammaArray, boundRadii)
    kMesh, boundSquaredRadMesh = np.meshgrid(kArray, squaredRadii)
    # find M such that Jacobi<= (k+M choose k)/(k+n-2 choose k)

    evalledDMesh = (2*comDim+1)**2 + 4/(np.square(np.log(boundRadMesh))) - (4*(comDim**2-comDim)/(1-boundSquaredRadMesh))
    validD = (evalledDMesh>=0)
    adjustedDMesh = evalledDMesh*validD
    bigNgammaFalse = -((3+2/np.log(boundRadMesh)+np.sqrt(adjustedDMesh))/2)
    bigNgamma = bigNgammaFalse*validD
    # comDimMesh = (comDim - 2) * np.ones_like(bigNgamma)
    gammaEvalMesh = np.maximum(gammaMesh, bigNgamma)
    # gammaPochMesh = np.maximum(gammaEvalMesh, comDimMesh)

    # calc r^{M}* ( ((M+1)/(n-1)-1)) (1-r^2) +1)
    jacobiBound = 1 + np.square(1-boundSquaredRadMesh)*(-1+(((gammaEvalMesh+1)*(gammaEvalMesh+2))/((comDim)*(comDim-1))))
    radPowered = np.power(boundRadMesh, gammaEvalMesh)
    foundBounds = np.ones_like(boundRadMesh)
    foundBounds[:,ParamMask] = (jacobiBound * radPowered)[:,ParamMask]
    epsArray[boundMask] = np.max(foundBounds, axis=0)

    # boundSqrRem = (1 - np.square(boundRadii))
    # radiusDenom = np.power(boundSqrRem, comDim / 2 - 1)
    epsMatrix = epsArray.reshape(radiusShape)
    if perParamaterBool:
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None, :]
        epsOutput[boundMask] = foundBounds
        epsOutputRes = np.reshape(epsOutput, radiusShape + paramaterSize)
        return epsOutputRes
    else:
        return epsMatrix


def superBound(radiusArray,comDim,kArrayShaped,gammaArrayShaped, perParamaterBool=False, compCustoms=False):
    radiusArray = np.atleast_1d(radiusArray)
    radiusShape = radiusArray.shape
    flatRad = radiusArray.ravel()
    epsArray = np.zeros_like(flatRad)
    kArrayShaped = np.atleast_1d(kArrayShaped)
    gammaArrayShaped = np.atleast_1d(gammaArrayShaped)
    paramShape = kArrayShaped.shape
    kArray = kArrayShaped.ravel()
    gammaArray = gammaArrayShaped.ravel()
    # visitedKArray = np.unique(kArray)


    onesMask = (flatRad >= 1)
    zerosMask = (flatRad <= 0)
    # The trick here is (1 > flatRad > 0) iff not 1 and not 0
    boundMask = np.equal(onesMask,zerosMask)
    # Handle ones
    epsArray[onesMask] = 1
    # Handle zeros
    zeroGammas = (gammaArray == 0)
    lgAtZero = (-loggamma(comDim - 1 + kArray) + loggamma(kArray + 1)
                + loggamma(comDim - 1))
    valAt0 = np.exp(lgAtZero)
    boundZeroArray = zeroGammas * valAt0
    boundZero = np.max(boundZeroArray)
    epsArray[zerosMask] = boundZero
    # Handle non extrema
    # boundMask = (1 > flatRad > 0)
    # create every type of radius needed
    boundRadii = flatRad[boundMask]

    boundSqrRem = (1 - np.square(boundRadii))
    radiusDenom = np.power(boundSqrRem, 1- comDim / 2)
    strongRadiusDenom = np.power(4 * np.square(boundRadii) * boundSqrRem, -1 / 4) * radiusDenom
    squaredRadii = np.square(boundRadii)
    gammaMesh, boundRadMesh = np.meshgrid(gammaArray, boundRadii)
    kMesh, boundSquaredRadMesh = np.meshgrid(kArray, squaredRadii)
    if compCustoms:
        customRadii = np.sqrt(boundRadii)
        customSquared = boundRadii
        gammaMesh, customBoundRadMesh = np.meshgrid(gammaArray, customRadii)
        kMesh, customBoundSquaredRadMesh = np.meshgrid(kArray, customSquared)
    else:
        customRadii = boundRadii
        customBoundRadMesh = boundRadMesh
        customBoundSquaredRadMesh = boundSquaredRadMesh


    # k=2
    # find M such that Jacobi<= r^M ( (|P_2^{n-2,M}(-1)|-1)(1-r^2)^2 + 1)

    evalledDMesh = (2 * comDim + 1) ** 2 + 4 / (np.square(np.log(boundRadMesh))) - (
                4 * (comDim ** 2 - comDim) / (1 - boundSquaredRadMesh))
    validD = (evalledDMesh >= 0)
    adjustedDMesh = evalledDMesh * validD
    bigNgammaFalse = -((3 + 2 / np.log(boundRadMesh) + np.sqrt(adjustedDMesh)) / 2)
    bigNgammaKTwo = bigNgammaFalse * validD

    gammaEvalMeshKTwo = np.maximum(gammaMesh, bigNgammaKTwo)


    # calc r^{M}* ( ((M+1)/(n-1)-1)) (1-r^2) +1)
    jacobiBoundKTwo = 1 + np.square(1 - customBoundSquaredRadMesh) * (
                -1 + (((gammaEvalMeshKTwo + 1) * (gammaEvalMeshKTwo + 2)) / ((comDim) * (comDim - 1))))
    radPoweredKTwo = np.power(customBoundRadMesh, gammaEvalMeshKTwo)
    foundBoundsKTwo = np.ones_like(customBoundRadMesh)
    ParamMaskKTwo = (kArray == 2)
    foundBoundsKTwo[:, ParamMaskKTwo] = (jacobiBoundKTwo * radPoweredKTwo)[:, ParamMaskKTwo]
    # epsArrayKTwo[boundMask] = np.max(foundBounds, axis=0)





    # k=1
    # find M such that Jacobi<= <= r^M ( (|P_1^{n-2,M}(-1)|-1)(1-r^2) + 1)
    bigNgammaKOne = -(
                1 / np.log(customBoundSquaredRadMesh) + 1 +
                ((comDim - 1) * customBoundSquaredRadMesh) / (1 - customBoundSquaredRadMesh))
    gammaEvalMeshKOne = np.maximum(gammaMesh, bigNgammaKOne)

    # calc r^{M}* ( ((M+1)/(n-1)-1)) (1-r^2) +1)
    jacobiBoundKOne = customBoundSquaredRadMesh + (1 - customBoundSquaredRadMesh) * ((gammaEvalMeshKOne + 1) / (comDim - 1))
    radPoweredKOne = np.power(customBoundRadMesh, gammaEvalMeshKOne)
    foundBoundsKOne = np.ones_like(customBoundRadMesh)
    ParamMaskKOne = (kArray == 1)
    foundBoundsKOne[:, ParamMaskKOne] = (jacobiBoundKOne * radPoweredKOne)[:, ParamMaskKOne]


    # Stupid bound
    # bound by r^{\gamma}|P_k^{n-2,\gamma}(x)| x=-1 if gamma>=n-2, x=1 if gamma<n-2
    # Handle non extrema
    # boundMask = (1 > flatRad > 0)
    # boundRadii = flatRad[boundMask]
    # gammaMesh, boundRadMesh = np.meshgrid(gammaArray, boundRadii)
    # kMesh, boundRadMesh = np.meshgrid(kArray, boundRadii)
    # find M such that r^{M} outgrows jacobi bound
    bigNgammaStupid = stupidBigN(kArray, customRadii)
    comDimMesh = (comDim - 2) * np.ones_like(bigNgammaStupid)
    gammaEvalMeshStupid = np.maximum(gammaMesh, bigNgammaStupid)
    gammaPochMeshStupid = np.maximum(gammaEvalMeshStupid, comDimMesh)

    # calc r^{M}* max(P_k^{n-2,M}(-1), P_k^{n-2,M}(1))
    lgStupid = (gammaEvalMeshStupid*np.log(customBoundRadMesh)-loggamma(comDim - 1 + kMesh) +
          loggamma(kMesh + gammaPochMeshStupid + 1) + loggamma(comDim - 1)
          - loggamma(gammaPochMeshStupid + 1))
    # poch_ratio = np.exp(lg)
    # radPowered = np.power(boundRadMesh, gammaEvalMesh)
    # foundBounds = poch_ratio * radPowered
    foundBoundsStupid = np.exp(lgStupid)



    # the two bounds that prove convergence
    lgConv = (loggamma(comDim - 1 + kArray) + loggamma(comDim - 1 + kArray + gammaArray) - 2 * loggamma(comDim - 1)
          - loggamma(kArray + 1) - loggamma(kArray + gammaArray + 1)) / 2
    poch_ratioConv = np.exp(lgConv)
    convBoundFacConv = 12 / np.power(2 * kArray + gammaArray + comDim - 1, 1 / 4)
    startBoundFacConv = np.power(
        ((kArray + 1) * (kArray + gammaArray + comDim - 1)) / ((kArray + comDim - 1) * (kArray + gammaArray + 1)),
        1 / 4)
    convBoundRadlessConv = convBoundFacConv / (poch_ratioConv)  # * radiusDenom * strongRadiusDenom)
    convBoundConv = np.outer(strongRadiusDenom, convBoundRadlessConv)
    startBoundRadlessConv = startBoundFacConv / (poch_ratioConv)  # * radiusDenom)
    startBoundConv = np.outer(radiusDenom, startBoundRadlessConv)

    # Handle zeros
    # zeroGammas = (gammaArray == 0)
    # boundZeroArray = zeroGammas / poch_ratio

    # boundSqrRem = (1 - np.square(boundRadii))
    # radiusDenom = np.power(boundSqrRem, comDim / 2 - 1)
    allBounds = np.stack([convBoundConv, startBoundConv, foundBoundsKOne,foundBoundsKTwo,foundBoundsStupid])
    bestBound = np.min(allBounds, axis=0)
    superBoundArray = np.max(bestBound, axis=1)
    epsArray[boundMask] = superBoundArray
    epsMatrix = epsArray.reshape(radiusShape)
    if perParamaterBool:
        flatRadSize = flatRad.shape
        paramaterSize = kArray.shape
        epsOutput = np.ones(flatRadSize + paramaterSize)
        epsOutput[zerosMask] = boundZeroArray[None, :]
        epsOutput[boundMask] = bestBound
        epsOutputRes = np.reshape(epsOutput, radiusShape + paramShape)

        # sufficientPlaces = (epsOutputRes <= 0.001)[0]
        # has_any = sufficientPlaces.any(axis=0)
        # # sufficientK = checkKMesh[sufficientPlaces]
        # # sufficientGamma = checkGammaMesh[sufficientPlaces]
        # foundK = kArrayShaped[0,has_any]  # k values that have any hit
        # min_gamma_cols = np.where(sufficientPlaces, gammaArrayShaped, np.inf).min(axis=0)
        # minimumGamma = min_gamma_cols[has_any]  # min γ for each found k
        # foundGammaBorder[foundK] = minimumGamma
        return epsOutputRes
    else:
        return epsMatrix


def borderBasedOnEpsilon(eps, comDim, ipmSet,kOnly=False):
    radiusSet, thetaSet = z2polar(ipmSet)
    nonZero = (radiusSet > 0+eps)
    nonOne = (radiusSet < 1-eps)
    interestingIdx = nonZero*nonOne
    interestingVals = radiusSet[interestingIdx]
    if comDim == 2:
        if len(interestingVals) == 0:
            if np.any(nonOne):
                mainRad = 0
            else:
                mainRad = 1-eps
        else:
            mainRad = interestingVals[np.argmax(np.abs(interestingVals-0.5))]
        maxK = np.ceil((((12/eps)**4)/(4*np.square(mainRad)*(1-np.square(mainRad)))-1)/2)
    else:
        # mainRad = np.max(interestingVals)
        if len(interestingVals) == 0:
            if np.any(nonOne):
                mainRad = 0
            else:
                mainRad = 1 - eps
        else:
            mainRad = np.sqrt(-(np.power(np.average(np.power(1-np.square(interestingVals),1-comDim/2)),1/(1-comDim/2))-1))
        upperK = np.ceil(np.power(special.gamma(comDim-1)/eps, 1/(comDim-2)))/(np.sqrt(1-mainRad**2))
        interestedK = np.arange(upperK+1)
        interestedGamma = np.zeros_like(interestedK)
        calcedEps = bigBounds(mainRad, comDim, interestedK,interestedGamma,perParamaterBool=True)[0]
        sufficientK = (calcedEps <= eps)
        maxK = np.min(np.argwhere(sufficientK))+1
    if kOnly:
        foundKBorder = np.array([maxK],dtype=int)
        foundGammaBorder = np.zeros_like(foundKBorder)
        return foundKBorder, foundGammaBorder
    foundKBorder = np.arange(maxK+1,dtype=int)
    foundGammaBorder = np.zeros_like(foundKBorder)
    kLeftToFind = np.ones_like(foundKBorder,dtype=np.bool)
    kLeftToFind[-1] = False
    # curGammaIdx = 0
    gammaStepSize = math.ceil(100000.0/maxK)
    checkingGamma = np.arange(gammaStepSize)
    while np.any(kLeftToFind):
        checkingK = foundKBorder[kLeftToFind]
        checkKMesh,checkGammaMesh = np.meshgrid(checkingK,checkingGamma)
        boundValMatrix = superBound(mainRad, comDim, checkKMesh, checkGammaMesh, perParamaterBool=True, compCustoms=True)[0]
        sufficientPlaces = (boundValMatrix <= eps)
        has_any = sufficientPlaces.any(axis=0)
        # sufficientK = checkKMesh[sufficientPlaces]
        # sufficientGamma = checkGammaMesh[sufficientPlaces]
        foundK = checkingK[has_any]  # k values that have any hit
        min_gamma_cols = np.where(sufficientPlaces, checkGammaMesh, np.inf).min(axis=0)
        minimumGamma = min_gamma_cols[has_any]  # min γ for each found k
        foundGammaBorder[foundK] = minimumGamma
        kLeftToFind[foundK] = False

        checkingGamma += gammaStepSize
    return foundKBorder, foundGammaBorder



def quickJumpNavigator():
    return "ok"




if __name__ == '__main__':
    # testPath = "TestResults_sequentialEdges_07_09_14-32.json"
    # testKey = "TestNr0"
    # readPointsetAndRunModelWith(testPath, testKey,
    #                            overridingSetting={"setLinks": 2})
    # readPointsetAndRunDualWith(testPath, testKey,
    #                             overridingSetting={"setLinks":2})
    # gc.disable()
    allTestTypes = {0:"reverseWeighted",1:"spreadPoints",2:"sequentialEdges",
                    3:"uniformCoordinateWise",4:"compareToLowerBound"}
    allTestTypesPD = {0:"trulyUniformOnSphere",1: "uniformCoordinateWise", 2: "compareToLowerBound"}
    testDim = 4
    testRad = 0.0
    testTheta = 0.0
    nrOfTests = 3
    bqpType = allTestTypesPD[0]
    testArgsPD = {bqpType: True, "eps": 0.001, "sizeOfBQPSet": 6, "setAmount": 2, "setLinks": 1,
                    "improvementWithFacets": True}
    runTestsPD(testDim, testRad, testTheta, nrOfTests, testArgsPD, bqpType)
    # for setSize in range(2,testArgs["setLinks"]*testArgs["sizeOfBQPSet"]+1):
    #     charMatrixDict[setSize] = characteristicMatrix(setSize)
    # readOrInitFacetScores()
    # testArgs = {bqpType: True, "maxDeg": 1500, "sizeOfBQPSet": 6, "setAmount": 5, "setLinks": 2,
    #             "improvementWithFacets": True}
    # runTests(testDim, testRad, testTheta, nrOfTests, testArgs, bqpType)
    # testRadArray = np.linspace(0,1,1000)# np.random.random(1) #(5,10)
    # pointSet,_,_ = SphereBasics.createRandomPointsComplex(4,6,includeInner=False)
    # ipmTestSet = pointSet @ np.conj(pointSet).T
    # aMaxK = np.arange(11)
    # aMaxGamma = np.arange(21)
    # radiusSet1, thetaSet1 = z2polar(ipmTestSet)
    # nonZero1 = (radiusSet1 > 0 + 0.001)
    # nonOne1 = (radiusSet1 < 1 - 0.001)
    # interestingIdx1 = nonZero1 * nonOne1
    # interestingVals1 = radiusSet1[interestingIdx1]
    # interestingThetas1 = thetaSet1[interestingIdx1]
    # interestingZs1 = ipmTestSet[interestingIdx1]
    # pointSetSize = interestingVals1.size
    #
    #
    #
    # coefArrayAmax = np.ones((21,11))/(21*11)
    # aMaxTester = DiskCombi(testDim-2,coefArrayAmax)
    # # interestingAvals = aMaxTester.allValues(interestingVals1,
    # #                                         interestingThetas1).reshape((21*11,))[1:,:]
    # # betaMax = np.max(np.abs(interestingAvals), axis=0)
    # # epsWeights = np.square(betaMax)
    # # #
    # Aarray = aMaxTester.allValues(z2polar(ipmTestSet)[0],z2polar(ipmTestSet)[1])
    # AVals = np.sum(np.abs(Aarray), axis=(2, 3)).ravel()[1:] - 6
    # AMax = np.max(AVals)
    #
    #
    # testRadArray = np.random.random((6,6))
    # testThetaArray = np.zeros_like(testRadArray)
    # testZArray = polar2z(testRadArray,testThetaArray)
    # # finds a border such that concatenating that way makes the problem
    # # have an optimal value of around (1+objDiffMax)*theta (likely lower though)
    # objDiffMax = 0.0001
    # testKBorder,testGammaBorder = borderBasedOnEpsilon(objDiffMax/AMax,testDim,ipmTestSet)
    # testKBorder[0]=5000
    #
    # testkArray = np.arange(3)
    # testgammaArray = np.arange(500)
    #
    #
    # testkMesh,testgammaMesh = np.meshgrid(testkArray,testgammaArray)
    # foundBounds = superBound(radiusSet1,testDim,testKBorder,testGammaBorder,True)
    # # allZeroGammaBorder = np.ones_like(testGammaBorder,dtype=int)
    # # allZeroGammaBorder[-1] = 0
    # concatDual(0,0,testDim,testKBorder,testGammaBorder, listOfPointSets=[pointSet],kOnly=True)
    # # kMax = 500
    # # testkArray = np.arange(kMax)
    # # testgammaArray = np.clip(np.ceil(kMax ** 2 / np.clip(testkArray, 1, np.inf) - testkArray), 0, kMax ** 2)
    # # foundBounds = superBound(testRadArray, testDim, testkArray, testgammaArray, True)
    # testRad = 0
    # testTheta = 0
    # testMatrix = characteristicMatrix(6)
    # cutOffFunc = lambda k, gamma: (((((k+1)**(testDim-2)) * ((k+gamma+1)**(testDim-2)))**(1/2))
    #                                 * ((((k+testDim-1)*(k+gamma+1))/((k+1)*(k+gamma+testDim-1)))**(1/4)))
    # cutOffFunc = lambda k, gamma: (((k +1) ** (testDim - 2)) * ((k + gamma + 1) ** (testDim - 2))) ** (1 / 2)
    # cutOffVal = 100
    # verificationFunc = lambda k: ((cutOffVal**(2))/((k+1)**(testDim-2)))**(1/(testDim-2))-k
    # verificationFunc = lambda k: ((cutOffVal ** (2)) / ((k + 1) ** (testDim - 2))) ** (1 / (testDim - 2)) - k
    # kBorder,gammaBorder = borderMaker(cutOffFunc,cutOffVal,verificationFunc)
    # randomRadii = np.linspace(0,1,1000)
    # bigBounds(randomRadii, testDim, kBorder,gammaBorder)
    # modelBQP(0, 0, 3, 15, 15, nrOfPoints=14)
    # for dimension in range(testDim, 20):
    #     makeModelv2((dimension - 3.0) / 2.0, (dimension - 3.0) / 2.0, 0)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

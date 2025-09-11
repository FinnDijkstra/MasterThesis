# from typing import final
import re
import time
import gurobipy as gp
from gurobipy import GRB
import math
import scipy
import numpy as np
from pathlib import Path
# import fractions
# import random
import scipy.stats as st
from scipy.special import hyp2f1, loggamma, binom
# import sympy as sym
import pandas as pd
import json
import os
import datetime
# from multiprocessing import Pool, cpu_count
from itertools import combinations
# import improvedBQPNLP
# from pympler import asizeof
import psutil
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
rss = lambda: psutil.Process(os.getpid()).memory_info().rss/1024**2  # MB


def stitchSets(arrayList, setLinks, keepInitials=True):
    if setLinks == 1:
        combedArrays = arrayList
    elif len(arrayList) <= setLinks:
        if keepInitials:
            combedArrays = [np.concat(arrayList)]
        else:
            combedArrays = [np.concat([arrayList[0][:2]] + [arrayList[setIdx][2:]
                                                                      for setIdx in range(len(arrayList))])]
    else:
        combedArrays = []
        if keepInitials:
            for comb in combinations(range(len(arrayList)), setLinks):
                combedArrays.append(np.concat([arrayList[combIdx] for combIdx in comb]))
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
            listOfPointSets.append(SphereBasics.createRandomPointsComplex(complexDimension, nrOfPoints,
                                                                            baseZ=forbiddenZ)[0])
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
            pointSet = listOfPointSets[pointSetIdx]
            nrOfPoints = pointSet.shape[0]
            # charMatrix = characteristicMatrix(nrOfPoints)
            nrOfSets = 2**nrOfPoints

            innerProductMatrix = pointSet @ pointSet.conj().T
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

        print("A) before env:", rss())
        env = gp.Env(empty=True)
        env.start()
        print("B) after env :", rss())
        m = gp.Model(env=env)
        print("C) after model:", rss())
        # env.start()
        # m = gp.Model(f"BQP model for forbidden innerproduct {forbiddenZ} in dimension {complexDimension}, "
        #              f"with {nrOfPoints} BQP points,"
        #              f" gamma checked {gammaMax}, k checked {kMax}", env=env)
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
        print("D) after build:", rss())

        print("F) after delete:", rss())
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
        print("E) after dispose:", rss())
        del diskPolysByGamma, listOfPointSets, setWeights, lbConstr, forbiddenInnerproductVals, (
            diskPolyWeights), diskPolyValueTensorList, diskPolyValueTensor, curDPVT


def finalBQPModelv2(forbiddenRadius, forbiddenAngle, complexDimension, gammaMax, kMax, listOfPointSets=None,
                     nrOfPoints=2, nrOfPointSets=1):
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
        m.setParam("OutputFlag", 0)
        m.setParam(GRB.Param.FeasibilityTol, 1e-9)
        # m.setParam(GRB.Param.BarConvTol, 1e-12)
        # m.setParam(GRB.Param.BarQCPConvTol, 1e-12)
        # m.setParam(GRB.Param.MIPGap, 1e-12)
        # m.setParam(GRB.Param.IntFeasTol, 1e-9)
        # m.setParam(GRB.Param.PSDTol, 1e-12)
        m.setParam(GRB.Param.OptimalityTol, 1e-9)
        m.setParam(GRB.Param.DualReductions, 0)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.Method, 1)
        m.setParam(GRB.Param.NumericFocus, 2)
        # m.setParam(GRB.Param.Presolve, 0)

        diskPolyWeights = m.addVars(gammaMax, kMax, vtype=GRB.CONTINUOUS, lb=0, name="w")

        # Forbidden-IP constraint
        forbiddenInner = np.zeros((gammaMax, kMax))
        for g in range(gammaMax):
            forbiddenInner[g] = diskPolysByGamma[g].calcAtRTheta(forbiddenRadius, forbiddenAngle)
        m.addConstr(gp.quicksum(forbiddenInner[g, k]*diskPolyWeights[g, k]
                                for g in range(gammaMax) for k in range(kMax)) == 0)

        # “SDP” constraint you coded (quadratic)
        m.addConstr(diskPolyWeights[0, 0] - diskPolyWeights.sum()*diskPolyWeights.sum() >= 0)
        setWeightVars = []
        # Build per–point-set pieces
        largestSet = 0
        for ps_idx in range(nrOfPointSets):
            pts = listOfPointSets[ps_idx]
            n = pts.shape[0]

            # Characteristic matrix — consider sparse (see §3)
            charM = characteristicMatrix(n)                  # shape (n, nrSets)
            nrSets = charM.shape[1]
            setW = m.addVars(nrSets, vtype=GRB.CONTINUOUS, lb=0, name=f"s_{ps_idx}")
            m.addConstr(setW.sum() == 1)
            if nrSets > largestSet:
                largestSet = nrSets

            # IPM and disk poly values (avoid 4-D tensor; see §3)
            ipm = pts @ pts.conj().T
            r, th = z2polar(ipm)

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
                    m.addConstr(lin == gp.quicksum(setW[s] for s in nz))
            setWeightVars.append(setW)
        m.setObjective(diskPolyWeights.sum(), GRB.MAXIMIZE)

        # Lower-bound loop
        lb_ok = False
        p = complexDimension - 1
        while not lb_ok:
            lb = m.addConstr(diskPolyWeights.sum() >= (0.5)**p)
            m.update()
            m.optimize()
            if m.Status != GRB.OPTIMAL and m.Status != GRB.SUBOPTIMAL:
                m.remove(lb)
                p += 1
            else:
                lb_ok = True

        diskSum = sum(diskPolyWeights[g, k].X for g in range(gammaMax) for k in range(kMax))
        diskWeightsArray = np.array([[diskPolyWeights[g, k].X for k in range(kMax)]
                                     for g in range(gammaMax)])
        adjustedDWA = diskWeightsArray / diskSum
        objBound = m.objBound
        if objBound-diskSum < 0:
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





def facetInequalityBQPGammaless(dim,startingPointset, startingCoefficients,
                                startingFacet=0, maxDryRun=-1, stopEarly=True, useRadialEstimator=True):
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
    bestType = "None"
    sol = startingGuess
    lastImprovement = 0
    facetIdx = startingFacet + 1
    bestFacetIdx = "0"
    scoreboard = {"delivered":0,"random":0,"tie":0, "bad":0}
    goalboard = {"delivered":0,"random":0}
    facetNr = 0
    goals = 0
    if validIneqs:
        nrOfFacets = betas.shape[0]
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
            facetScoreDF.at[scoreCategories[0], idxStr] += 1
            objective = lambda S: betas[facetIdx] - np.sum(
                np.multiply(facetIneqs[facetIdx],polyVals(S)))
            constraint = lambda S: np.sum(np.square(S.reshape(shapeStartingGuess)),axis=0)-1

            # relativeSizeV1 = np.sqrt(np.sum(np.abs(facetIneqs[facetIdx])))
            relativeSizeV1 = 1
            currentCutOff = -1.5 * (-1/np.log2(nrOfFacets+2)+1/np.log2(facetNr+2))

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
                res = minimizationUnconstrained.run_from_flattened_start(
                    flattenedStartingGuess, shapeStartingGuess, startingPolynomial, facetIneqs[facetIdx],
                    unconstrained_mode="normalize",  # or "stereo"
                    unconstrained_method="L-BFGS-B",  # or "trust-constr"
                    workers_trust_constr=10,  # if you later set >1, keep BLAS threads = 1
                    finite_diff_rel_step=1e-6,
                    gtol=1e-6, xtol=1e-12, maxiter=2000, verbose=0
                )
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
            try:
                res = minimizationUnconstrained.run_from_flattened_start(
                    baseFlattenedStartingGuess, shapeStartingGuess, startingPolynomial, facetIneqs[facetIdx],
                    unconstrained_mode="normalize",  # or "stereo"
                    unconstrained_method="L-BFGS-B",  # or "trust-constr"
                    workers_trust_constr=10,  # if you later set >1, keep BLAS threads = 1
                    finite_diff_rel_step=1e-6,
                    gtol=1e-6, xtol=1e-12, maxiter=2000, verbose=0
                )
                # res = improvedBQPNLP.solve_problem(baseFlattenedStartingGuess, shapeStartingGuess, startingPolynomial,
                #                                        facetIneqs, facetIdx)
                # res = scipy.optimize.minimize(
                #     objective, baseFlattenedStartingGuess,
                #     method='trust-constr',
                #     constraints={'type': 'eq', 'fun': constraint},
                #     tol=1e-8,
                #     options={"maxiter":500}
                # )
            except:
                succesBase = False
            else:
                relativeObjective = (res.fun+betas[facetIdx]) / relativeSizeV1 # / relativeSizes[facetIdx]
                # print(f"suc{res.x},obj{relativeObjective}")
                baseObj = relativeObjective
                if relativeObjective < bestFacet: # and res.constr_violation < 1e-8:
                    # baseSol = res.x.reshape(shapeStartingGuess)
                    baseSol = minimizationUnconstrained.Z_from_unconstrained_normalize(res.x,n=dim,m=6)
                    baseObj = relativeObjective
                    succesBase = True
                else:
                    succesBase = False

            # Find best sol out of starting options
            resObj = baseObj
            resSol = baseSol
            succesRes = succesBase
            resType = "delivered"

            if succesRand:
                if randObj < baseObj:
                    resObj = randObj
                    resSol = randSol
                    succesRes = succesRand
                    resType = "random"
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
                print(f"found {resObj} at {idxStr} (nr of facets tested: {facetNr})")
                lastImprovement = 0
                bestFacet = resObj
                sol = resSol
                bestType = resType
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
    Scoreboard.show_dashboard(scoreboard,goalboard,facetNr+validIneqs,goals)
    print(f"Final winner: {bestType}")
    # facetScoreDF.to_csv(facetFileLoc)
    if bestFacet < 0:
        if bestType == "delivered":
            facetScoreDF.at[scoreCategories[2],bestFacetIdx] += 1
        else:
            facetScoreDF.at[scoreCategories[3], bestFacetIdx] += 1
        facetScoreDF.to_csv(facetFileLoc)
        return True, sol.T, (facetIdx+1) % nrOfFacets
    else:
        facetScoreDF.to_csv(facetFileLoc)
        return False, startingGuess.T, (facetIdx+1) % nrOfFacets




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
        pointSetsForModel = stitchSets(inputPointSets, setLinks, improvementWithFacets)
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
        # Run model with new BQP sets
        (coefsCurTheta, curObjVal,
         unscaledCoefsCurTheta) = finalBQPModelv2(forbiddenRad, forbiddenTheta, dim,
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


def readPointsetAndRunModelWith(fileLocation,testKey, overridingSetting=None):
    if overridingSetting is None:
        overridingSetting = {}
    file = open(fileLocation)
    dataDict = json.load(file)[testKey]
    del dataDict["coeficients disk polynomials list"], dataDict["unscaled coeficients"]
    pointsets = listToCompArray(dataDict["used point sets"])
    listOfStichedSets = stitchSets(list(pointsets.values()), dataDict["input parameters"]["setLinks"])
    inputParams = dataDict["input parameters"]
    inputParams.update(overridingSetting)
    finalBQPModelv2(inputParams["forbiddenRad"],
                    inputParams["forbiddenTheta"],
                    inputParams["dim"],
                    inputParams["maxGamma"],
                    inputParams["maxDeg"],
                    listOfStichedSets)
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


def quickJumpNavigator():
    return "ok"




if __name__ == '__main__':
    # readPointsetAndRunModelWith("TestResults_sequentialEdges_07_09_14-32.json", "TestNr0",
    #                             overridingSetting={"maxDeg":1200, "setLinks":5})
    # gc.disable()
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
    testArgs = {bqpType:True, "maxDeg":1500,"sizeOfBQPSet":6,"setAmount":5, "setLinks":2, "improvementWithFacets":True}
    # for setSize in range(2,testArgs["setLinks"]*testArgs["sizeOfBQPSet"]+1):
    #     charMatrixDict[setSize] = characteristicMatrix(setSize)
    readOrInitFacetScores()
    runTests(testDim,testRad,testTheta,nrOfTests,testArgs,bqpType)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from typing import final
import time
import gurobipy as gp
from gurobipy import GRB
import math
import scipy
import fractions
import random
from line_profiler import LineProfiler

nrOfVars = 10
nrOfTests = 200
dimension = 12
distanceForbidden = 0
jacobiType = "new"

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

    # for slIdx in range(nrOfSls):
    #     slindices, slvalues = getRow(aST, slIdx)
    #     rowSize = len(slindices)
    #     m.addConstr((gp.quicksum((tourWeight[slindices[clusterIdx]] * slvalues[clusterIdx])
    #                              for clusterIdx in range(rowSize)) + absoluteErrors[slIdx]
    #                  >= cl[slIdx]), name=f"positive errors Screenline {slIdx}")
    #     m.addConstr((gp.quicksum((tourWeight[slindices[clusterIdx]] * slvalues[clusterIdx])
    #                              for clusterIdx in range(rowSize)) - absoluteErrors[slIdx]
    #                  <= cl[slIdx]), name=f"negative errors Screenline {slIdx}")
    #
    # m.addConstrs(
    #     ((tourWeight[clusterIdx] - tourDeviation[clusterIdx] <= tbw[clusterIdx]) for clusterIdx in range(nrOfClusters)),
    #     name="Positive deviation of cluster")
    # m.addConstrs(
    #     ((tourWeight[clusterIdx] + tourDeviation[clusterIdx] >= tbw[clusterIdx]) for clusterIdx in range(nrOfClusters)),
    #     name="Negative deviation of cluster")
    # # m.addConstrs(((tourWeight[ODpair] - 1 <= largeErrors[ODpair]) for ODpair in toursDF.index.tolist()),
    # #              name="Large errors in tour")
    # # m.addConstr(totalError == gp.quicksum(tourDeviation[idx]*tourDeviation[idx]*tComp[idx]/tbw[idx] for idx in range(nrOfClusters)), name=f"total error")
    # m.addConstr(totalError == gp.quicksum(
    #     tourDeviation[idx] * singleTComp for idx in range(nrOfClusters)),
    #             name=f"total error")
    #
    # objective = (gp.quicksum(absoluteErrors[slIdx] for slIdx in range(nrOfSls))
    #              + (
    #                  totalError))  # + gp.quicksum(largeErrors[ODpair] for ODpair in toursDF.index.tolist())
    # m.setObjective(objective, GRB.MINIMIZE)
    # m.update()


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


if __name__ == '__main__':
    makeModelv2((dimension-3.0)/2.0, (dimension-3.0)/2.0, 0)
    denomList = list(range(1,256))
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
                print(f"\t Found theta value: {-thetaPartVal/(1-thetaPartVal)} (thetapart: {thetaPartVal})")
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


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

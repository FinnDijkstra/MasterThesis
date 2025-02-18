from typing import final

import gurobipy as gp
from gurobipy import GRB
import math
import scipy

nrOfVars = 10
dimension = 6
distanceForbidden = 0
jacobiType = "new"

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
    functDegree = degree
    if jacobiType == "old":
        return jacobiValue(functDegree, alpha, beta, location)/jacobiValue(functDegree, alpha, beta, 1)
    return scipy.special.eval_jacobi(functDegree, alpha, beta, location)/scipy.special.eval_jacobi(functDegree, alpha, beta, 1)

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

    m.addConstr((gp.quicksum(fList[fIter]
                             for fIter in range(nrOfVars)) == 1),
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

    m.addConstr((gp.quicksum(fList[fIter]
                             for fIter in range(nrOfVars)) == 1),
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

    m.addConstr((gp.quicksum(fList[fIter]
                             for fIter in range(nrOfVars)) == 1),
                name=f"Measure Constraint")

    # objective = (sphereMeasure**2 * fList[0])
    m.setObjective(fList[0], GRB.MAXIMIZE)
    m.update()
    m.optimize()
    print(m.ObjVal)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    makeModel("yes")

    newCalcForbiddenDistancev2 = 2*(distanceForbidden) - 1
    makeModelv2((dimension-3.0)/2.0, -1/2, newCalcForbiddenDistancev2)
    # Complex RP

    makeModelv2((dimension - 2.0), 0, newCalcForbiddenDistancev2)
    # Complex Sphere


    makeModelv2((dimension - 2.0), (dimension - 2.0), distanceForbidden)
    makeModelCascading(dimension-2.0, distanceForbidden, newCalcForbiddenDistancev2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

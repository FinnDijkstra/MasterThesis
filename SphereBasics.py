import scipy as sp
import scipy.stats as st
from scipy.interpolate import splrep, PPoly, UnivariateSpline
import math
import cmath
from numpy.polynomial import Polynomial
import numpy as np
# from scipy.interpolate import UnivariateSpline
from line_profiler import LineProfiler


polar2z = lambda r, theta: r * np.exp(1j * theta)
z2polar = lambda z: (np.abs(z), np.angle(z))

# Makes an autocorrelation function of double caps with radius c
def cap_fraction_grid(t_array, n, u1_points=200, u2_points=200, c=1.0 / np.sqrt(2.0)):
    t_array = np.atleast_1d(t_array)

    # Create uniform grids
    # make a grid for the poles (cos(thet), sin(thet), 0, 0,...)
    perspectivePoleInnerGrid = np.linspace(-1, 1, u1_points)
    sinPPIG = np.sin(np.arccos(perspectivePoleInnerGrid))
    # the weight of the group with (cos(thet),.....)
    pPIGWeight = np.abs(np.sin(np.arccos(perspectivePoleInnerGrid))) ** (n - 3)

    # Make a grid for the cos of the remainder of w with innerproduct t to v wrt (0,1,0,0,0,...) and its weight
    doubleProjectiveGridTheta = np.linspace(0, 2 * math.pi, u2_points)
    cosDPGT = np.cos(doubleProjectiveGridTheta)
    sinDPGT = np.abs(np.sin(doubleProjectiveGridTheta))
    dPGTWeight = sinDPGT ** (n - 3)
    dPGTTotWeight = np.sum(dPGTWeight)

    # Create mask if cap wrt pole is on v
    maskCap = (np.abs(perspectivePoleInnerGrid) > c)
    pPIGTotWeight = np.sum(pPIGWeight[maskCap]) / 2
    # Create mask if cap wrt pole has atleast 1 point that has innerproduct t
    term1 = np.abs(t_array[:, None] * perspectivePoleInnerGrid[None, :])
    term2 = np.sqrt(1 - t_array[:, None] ** 2) * np.sqrt(1 - perspectivePoleInnerGrid[None, :] ** 2)
    maskTCap = ((term1 + term2) > c)
    maskTot = maskTCap * maskCap

    # For each t, for each cap, test if each cosine of a point (with innerproduct t) wrt e_2 is in the cap and add its
    # weight if that is the case
    outputTVol = np.zeros_like(t_array)
    for tIDX in range(t_array.shape[0]):
        t = t_array[tIDX]
        tOrth = math.sqrt(1 - t ** 2)
        curTDensity = np.zeros(u1_points)
        for pole in range(u1_points):
            if maskCap[pole] & maskTCap[tIDX, pole]:
                poleInner = perspectivePoleInnerGrid[pole]
                poleOrth = sinPPIG[pole]
                innerFactor = t * poleInner
                OrthFactor = tOrth * poleOrth
                OrthComp = OrthFactor * cosDPGT
                innerPT = OrthComp + innerFactor
                capTAngMask = (innerPT > c)
                curTDensity[pole] = np.sum(dPGTWeight[capTAngMask]) / dPGTTotWeight
        outputTVol[tIDX] = np.inner(curTDensity, pPIGWeight) / pPIGTotWeight
    return outputTVol


def complexInnerproduct(radiiMatrix, anglesMatrix):
    nrOfColumns = radiiMatrix.shape[0]
    realPartMatrix = np.zeros((nrOfColumns, nrOfColumns))
    imagPartMatrix = np.zeros((nrOfColumns, nrOfColumns))
    for vec1Idx in range(nrOfColumns):
        vec1radii = radiiMatrix[vec1Idx]
        vec1angles = anglesMatrix[vec1Idx]
        for vec2Idx in range(nrOfColumns):
            vec2radii = radiiMatrix[vec2Idx]
            vec2angles = anglesMatrix[vec2Idx]
            newAngles = vec1angles - vec2angles
            newRadii = vec1radii * vec2radii
            realPart = np.dot(newRadii, np.cos(newAngles))
            imagPart = np.dot(newRadii, np.sin(newAngles))
            realPartMatrix[vec1Idx][vec2Idx] = realPart
            imagPartMatrix[vec1Idx][vec2Idx] = imagPart
    return realPartMatrix, imagPartMatrix


def decompose(realParts, imaginaryParts):
    columnNr = realParts.shape[0]
    radiiMatrix = np.power(np.power(realParts, 2) + np.power(imaginaryParts, 2), 1 / 2)
    nonZeroMask = (radiiMatrix > 0)
    angleMatrix = np.zeros((columnNr, columnNr))
    angleMatrix[nonZeroMask] = np.arccos(realParts[nonZeroMask] / radiiMatrix[nonZeroMask])
    overPiMask = (imaginaryParts < 0)
    angleMatrix[overPiMask] = - angleMatrix[overPiMask]
    # for vec1Idx in range(columnNr):
    #     for vec2Idx in range(columnNr):
    #         realPart = realParts[vec1Idx][vec2Idx]
    #         imagPart = imaginaryParts[vec1Idx][vec2Idx]
    #         radiiMatrix[vec1Idx][vec2Idx] = realPart**2 + imagPart**2
    #         if realPart == 0
    return radiiMatrix, angleMatrix


# Takes dimension and desired size and gives the weight density for the real Sphere
def realSphereDensity(sphereDim, arraySize):
    innerproductArray = np.linspace(-1, 1, arraySize)
    constFactor = sp.special.gamma(sphereDim / 2) / (sp.special.gamma(sphereDim / 2) * math.sqrt(math.pi) * arraySize)
    return constFactor * np.power((1 - np.power(innerproductArray, 2)), (sphereDim - 3) / 2)


# Takes dimension and desired size and gives the weight density of radii for the complex Sphere
def complexSphereDensity(sphereDim, arraySize):
    innerproductArray = np.linspace(0, 1, arraySize)
    constFactor = 2 * (sphereDim - 1) / arraySize
    return constFactor * np.power((1 - np.power(innerproductArray, 2)), (sphereDim - 2)) * innerproductArray


class real_sphere_pdf(st.rv_continuous):
    shapes = "c"

    def _argcheck(self, c):
        # only allow positive c, for instance
        return c > 0

    def _pdf(self, x, c):
        constFactor = sp.special.gamma(c / 2) / (
                sp.special.gamma(c / 2) * math.sqrt(math.pi))
        return constFactor * (1 - x ** 2) ** ((c - 3) / 2)


class complex_sphere_pdf(st.rv_continuous):
    shapes = "c"

    def _argcheck(self, c):
        # only allow positive c, for instance
        return c > 0

    def _pdf(self, x, c):
        return 2 * (c - 1) * x * (1 - x ** 2) ** (c - 2)


class piecewisePolyPdf(st.rv_continuous):
    shapes = "poly"

    def _argcheck(self, poly):
        # only allow positive c, for instance
        return isinstance(poly, PPoly)

    def _pdf(self, x, poly):
        return poly(x)          # __call__


class polyPdf(st.rv_continuous):
    # shapes = "poly"

    def __init__(self, poly, a, b, **kwargs):
        self.poly = poly
        super().__init__(a=a, b=b, **kwargs)

    # def _argcheck(self, poly):
    #     # only allow positive c, for instance
    #     return isinstance(poly, Polynomial)

    def _pdf(self, x):
        return self.poly(x)          # __call__


def createRandomPointsReal(baseDim, nrPoints, checkForMistakesBool=False):
    nrPoints = max(2, nrPoints)
    randomPointMatrix = np.zeros((nrPoints, baseDim))
    randomPointMatrix[0][0] = 1
    randomPointMatrix[1][1] = 1
    rsDistr = real_sphere_pdf(a=-1, b=1,
                              name=f'Equal distrubution on real sphere in R^{baseDim}')
    pdfList = [rsDistr(c=baseDim - layerIdx)
               for layerIdx in range(baseDim - 2)]
    variatesArray = np.array([pdfList[idx].rvs(size=nrPoints) for idx in range(baseDim - 2)])
    angles = st.uniform.rvs(scale=math.pi, size=nrPoints)
    signs = np.random.choice([-1, 1], size=nrPoints)
    for pointIdx in range(2, nrPoints):
        rSquared = 1
        degOfVecFreedom = min(pointIdx, baseDim - 1)
        pointVec = np.zeros(baseDim)
        for coordIdx in range(degOfVecFreedom):
            r = math.sqrt(rSquared)
            # catch two dimensional case (due to infinite densities as |r|=1)
            if coordIdx == baseDim - 2:
                angle = angles[pointIdx]
                pointVec[baseDim - 2] = r * math.cos(angle)
                rSquared -= (r * math.cos(angle)) ** 2
            # find v[coordIdx]=<e_{coordIdx},v> with prior knowledge of v[i<coordIdx] results in n-coordIdx sphere
            # with radius r = sqrt(1-v_0^2 - v_1^2 -...- v_{coordIdx-1}^2)
            else:
                coordValue = r * variatesArray[coordIdx][pointIdx]
                rSquared -= coordValue ** 2
                pointVec[coordIdx] = coordValue
            # Should not happen tbh, only happens if calcs are wrong or if a variate is exactly 1)
            if rSquared <= 0:
                print("Situation with prob<=0")
                break
        pointVec[degOfVecFreedom] = math.sqrt(rSquared) * signs[pointIdx]
        randomPointMatrix[pointIdx] = pointVec
    if checkForMistakesBool:
        squaredCoordinates = np.power(randomPointMatrix, 2)
        vectorSizes = np.sum(squaredCoordinates, axis=1)
        if np.any((vectorSizes - 1) > 0.00001):
            print("mistake found")
    return randomPointMatrix


def createRandomPointsComplex(baseDim, nrPoints, baseZ=0+0j, checkForMistakesBool=False, includeInner=True):
    nrPoints = max(2, nrPoints)

    randomPointMatrixRadii = np.zeros((nrPoints, baseDim))
    randomPointMatrixRadii[0][0] = 1
    if includeInner:
        baseRad, baseAngle = z2polar(baseZ)
    else:
        randomVals = np.random.random(2)
        baseRad = randomVals[0]
        baseAngle = 2*np.pi * randomVals[1]
    randomPointMatrixRadii[1][0] = baseRad
    randomPointMatrixRadii[1][1] = 1-baseRad

    csDistr = complex_sphere_pdf(a=0, b=1,
                                 name=f'Equal distrubution on complex sphere in R^{baseDim}')
    pdfList = [csDistr(c=baseDim - layerIdx)
               for layerIdx in range(baseDim - 1)]
    variatesArray = np.array([pdfList[idx].rvs(size=nrPoints) for idx in range(baseDim - 1)])
    coordinateAngles = st.uniform.rvs(scale=math.pi, size=(nrPoints, baseDim))
    coordinateAngles[1][0] = baseAngle
    unangledCoordinates = np.zeros((nrPoints, baseDim), dtype=bool)
    smallestCoordinate = min(baseDim, nrPoints)
    for i in range(smallestCoordinate):
        for j in range(i, smallestCoordinate):
            unangledCoordinates[i][j] = True
    coordinateAngles[unangledCoordinates] = 0
    for pointIdx in range(2, nrPoints):
        rSquared = 1
        degOfVecFreedom = min(pointIdx, baseDim - 1)
        pointVec = np.zeros(baseDim)
        for coordIdx in range(degOfVecFreedom):
            r = math.sqrt(rSquared)
            # find v[coordIdx]=<e_{coordIdx},v> with prior knowledge of v[i<coordIdx] results in n-coordIdx sphere
            # with radius r = sqrt(1-v_0^2 - v_1^2 -...- v_{coordIdx-1}^2)
            coordValue = r * variatesArray[coordIdx][pointIdx]
            rSquared -= coordValue ** 2
            pointVec[coordIdx] = coordValue
            # Should not happen tbh, only happens if calcs are wrong or if a variate is exactly 1)
            if rSquared <= 0:
                print("Situation with prob<=0")
                break
        pointVec[degOfVecFreedom] = math.sqrt(rSquared)
        randomPointMatrixRadii[pointIdx] = pointVec
    if checkForMistakesBool:
        squaredCoordinates = np.power(randomPointMatrixRadii, 2)
        vectorSizes = np.sum(squaredCoordinates, axis=1)
        if np.any(vectorSizes != 1):
            print("mistake found")
    complexMatrix = polar2z(randomPointMatrixRadii, coordinateAngles)
    return complexMatrix, randomPointMatrixRadii, coordinateAngles


def complexTester():
    randomComplexSpherePointRadii, randomAngles = createRandomPointsComplex(4, 6, checkForMistakesBool=True)[1:]
    realPartInnerProducts, imaginaryPartInnerProducts = complexInnerproduct(randomComplexSpherePointRadii, randomAngles)
    radii, angles = decompose(realPartInnerProducts, imaginaryPartInnerProducts)
    radii2, angles2 = z2polar(realPartInnerProducts + 1j * imaginaryPartInnerProducts)

    complexPointNotation = polar2z(randomComplexSpherePointRadii, randomAngles)
    complexInnerProductVals = (complexPointNotation @ np.conjugate(complexPointNotation).T)
    radii3, angles3 = z2polar(complexInnerProductVals)
    return radii, angles, radii2, angles2, radii3, angles3


def complexAngleIntegrator(inputDatapoints, angleAxis=1, normalize=False):
    integratedPoints = np.sum(inputDatapoints, axis=angleAxis)/inputDatapoints.shape[angleAxis]
    if normalize:
        integratedPoints = integratedPoints / np.sum(integratedPoints)
    return integratedPoints


def complexWeightApply(inputDatapoints, complexDimension, radiusArray=None, radiusAxis=0):
    inputDataSize = inputDatapoints.shape[radiusAxis]
    if radiusArray is None:
        radiusArray = np.linspace(0,1, inputDataSize)
    weightArray = 2*(complexDimension - 1)*radiusArray*np.power((1-np.square(radiusArray)), complexDimension-2)/inputDataSize
    shape = [1] * inputDatapoints.ndim
    shape[radiusAxis] = weightArray.shape[0]
    weightNDimArray = weightArray.reshape(shape)
    return inputDatapoints*weightNDimArray


def completeIntegrator(inputDatapoints, complexDimension, radiusArray=None, applyWeightBool=True,
                       radiusAxis=0, angleAxis=1):
    if applyWeightBool:
        weightedInputDatapoints = complexWeightApply(inputDatapoints, complexDimension, radiusArray, radiusAxis)
    else:
        weightedInputDatapoints = inputDatapoints
    radialIntegrals = complexAngleIntegrator(weightedInputDatapoints, angleAxis)
    return np.sum(radialIntegrals)


def radialIntegrator(inputDatapoints, complexDimension, radiusArray=None, applyWeightBool=True):
    if applyWeightBool:
        weightedInputDatapoints = complexWeightApply(inputDatapoints, complexDimension, radiusArray, 0)
    else:
        weightedInputDatapoints = inputDatapoints
    return np.sum(weightedInputDatapoints)


def weightedDifference(largerData, smallerData, complexDimension, radiusArray=None, radiusAxis=0, normalizedBool=True):
    if normalizedBool:
        diffMatrix = complexWeightApply(np.abs(largerData-smallerData), complexDimension, radiusArray, radiusAxis)
        integral = completeIntegrator(diffMatrix, complexDimension, radiusArray, radiusAxis=radiusAxis)
        if integral > 0:
            return diffMatrix/integral
        else:
            return diffMatrix
    else:
        return complexWeightApply(np.abs(largerData-smallerData), complexDimension, radiusArray, radiusAxis)


def fourierCoefficientsNormalized(inputData, maxDeg, radiusAxis, angleAxis=1):
    M = inputData.shape[angleAxis]
    dTheta = 2 * np.pi / M
    thetaArray = np.linspace(0, 2 * np.pi, M, endpoint=False)
    A = inputData.shape[radiusAxis]
    hat_a = np.zeros((A, maxDeg + 1))
    hat_b = np.zeros((A, maxDeg + 1))

    # k=0 term is just the mean:
    hat_a[:, 0] = np.sum(inputData, axis=1) * dTheta

    for k in range(1, maxDeg + 1):
        cos_ktheta = np.cos(k * thetaArray)
        sin_ktheta = np.sin(k * thetaArray)

        # For each radius a, take dot product with cos/sin
        hat_a[:, k] = np.tensordot(inputData, cos_ktheta, axes=([1], [0])) * dTheta
        hat_b[:, k] = np.tensordot(inputData, sin_ktheta, axes=([1], [0])) * dTheta

    return hat_a, hat_b

def normalizeRadials(inputDatapoints, radiusAxis=0, angleAxis=1):
    fr = complexAngleIntegrator(inputDatapoints, angleAxis)
    fMask = np.logical_not(fr > 0)
    fInt = radialIntegrator(fr, 0, applyWeightBool=False)
    # fr /= fInt
    fr[fMask] = 1
    shape = [1] * inputDatapoints.ndim
    shape[radiusAxis] = fr.shape[0]
    frNDimArray = fr.reshape(shape)
    normalizedDatapoints = inputDatapoints / frNDimArray
    oNDimArray = fMask.reshape(shape)
    normalizedDatapoints += oNDimArray
    return normalizedDatapoints


def complexDoubleCapv3(dim, outputArraySize, poleResolution=0):
    if poleResolution == 0:
        poleResolution = outputArraySize
    # inbetweenSize = outputArraySize-2
    tArray = alternateRadiusSpaceConstructor(outputArraySize, dim, 0, 1)
    r1Array = alternateRadiusSpaceConstructor(poleResolution, dim, np.sqrt(1/2), 1)
    r2Array = alternateRadiusSpaceConstructor(poleResolution, dim-1, 0, 1)
    tMesh, r1Mesh, r2Mesh = np.meshgrid(tArray, r1Array, r2Array,indexing="ij")
    tSquaredMesh = np.square(tMesh)
    tRemSquared = (1-tSquaredMesh)
    tRemainder = np.sqrt(tRemSquared)
    r1SquaredMesh = np.square(r1Mesh)
    r1RemSquared = (1-r1SquaredMesh)
    r1Remainder = np.sqrt(r1RemSquared)
    r2SquaredMesh = np.square(r2Mesh)
    r2RemSquared = (1-r2SquaredMesh)
    r2Remainder = np.sqrt(r2RemSquared)
    # r1WeightPoly = 2**(dim-1)*complexWeightPolyCreator(dim)*(1 - np.sqrt(1/2))
    # r2WeightPoly = complexWeightPolyCreator(dim-1)
    # r1WeightMesh = r1WeightPoly(r1Mesh)
    # r2WeightMesh = r2WeightPoly(r2Mesh)
    r2True = r2Mesh*r1Remainder
    numeratorMesh = - tSquaredMesh*r1SquaredMesh - tRemSquared*r1RemSquared*r2SquaredMesh + 1/2
    denominatorMesh = 2 *tMesh*tRemainder*r1Mesh*r1Remainder*r2Mesh
    bestMesh = -denominatorMesh+numeratorMesh
    possibleMesh = (bestMesh < 0)
    worstMesh = denominatorMesh+numeratorMesh
    avoidableMesh = (worstMesh > 0)
    cosinableMesh = np.logical_and(possibleMesh, avoidableMesh)
    impossibleMesh = np.logical_not(possibleMesh)
    unavoidableMesh = np.logical_not(avoidableMesh)
    cosineMesh = np.ones_like(tMesh)
    cosineMesh[cosinableMesh] = numeratorMesh[cosinableMesh]/denominatorMesh[cosinableMesh]
    cosineMesh[unavoidableMesh] = -1
    cosineMesh[impossibleMesh] = 1
    arccosMesh = np.arccos(cosineMesh)
    psiValueMesh = arccosMesh/np.pi
    # weightedMesh = psiValueMesh*r1WeightMesh*r2WeightMesh
    r2integrated = np.sum(psiValueMesh, axis=2)/poleResolution
    r1integrated = np.sum(r2integrated, axis=1)/poleResolution
    # outputArray = np.zeros(outputArraySize)
    # outputArray[-1] = 1
    # outputArray[1:-1] = r1integrated
    # fullRadiusArray = np.zeros(outputArraySize)
    # fullRadiusArray[1:-1] = tArray
    # fullRadiusArray[-1] = 1
    return r1integrated, tArray


def padArray(array,leftVal=0,rightVal=1):
    paddedArray = np.zeros(array.shape[0]+2)
    paddedArray[1:-1] = array
    paddedArray[0] = leftVal
    paddedArray[-1] = rightVal
    return paddedArray


def complexDoubleCapv2(dim, outputArraySize, poleResolution=0):
    if poleResolution == 0:
        poleResolution = outputArraySize
    tArray = np.linspace(0, 1, outputArraySize, endpoint=True)
    r1Array = np.linspace(np.sqrt(1/2),1,poleResolution, endpoint=True)
    r2Array = np.linspace(0,1,poleResolution, endpoint=True)
    tMesh, r1Mesh, r2Mesh = np.meshgrid(tArray, r1Array, r2Array,indexing="ij")
    tSquaredMesh = np.square(tMesh)
    tRemSquared = (1-tSquaredMesh)
    tRemainder = np.sqrt(tRemSquared)
    r1SquaredMesh = np.square(r1Mesh)
    r1RemSquared = (1-r1SquaredMesh)
    r1Remainder = np.sqrt(r1RemSquared)
    r2SquaredMesh = np.square(r2Mesh)
    r2RemSquared = (1-r2SquaredMesh)
    r2Remainder = np.sqrt(r2RemSquared)
    r1WeightPoly = 2**(dim-1)*complexWeightPolyCreator(dim)*(1 - np.sqrt(1/2))
    r2WeightPoly = complexWeightPolyCreator(dim-1)
    r1WeightMesh = r1WeightPoly(r1Mesh)
    r2WeightMesh = r2WeightPoly(r2Mesh)
    r2True = r2Mesh*r1Remainder
    numeratorMesh = - tSquaredMesh*r1SquaredMesh - tRemSquared*r1RemSquared*r2SquaredMesh + 1/2
    denominatorMesh = -2 *tMesh*tRemainder*r1Mesh*r1Remainder*r2Mesh
    bestMesh = denominatorMesh-numeratorMesh
    possibleMesh = (bestMesh < 0)
    worstMesh = -denominatorMesh-numeratorMesh
    avoidableMesh = (worstMesh > 0)
    cosinableMesh = np.logical_and(possibleMesh, avoidableMesh)
    impossibleMesh = np.logical_not(possibleMesh)
    unavoidableMesh = np.logical_not(avoidableMesh)
    cosineMesh = np.ones_like(tMesh)
    cosineMesh[cosinableMesh] = -numeratorMesh[cosinableMesh]/denominatorMesh[cosinableMesh]
    cosineMesh[unavoidableMesh] = 1
    cosineMesh[impossibleMesh] = -1
    arccosMesh = np.arccos(cosineMesh)
    psiValueMesh = arccosMesh/np.pi
    weightedMesh = psiValueMesh*r1WeightMesh*r2WeightMesh
    r2integrated = np.sum(weightedMesh, axis=2)/poleResolution
    r1integrated = np.sum(r2integrated, axis=1)/poleResolution
    return r1integrated


def complexDoubleCap(dim, outputArraySize,poleArraySize=0):
    # Find f(r) numerically that describes the correlation function for the double cap in dim
    if poleArraySize == 0:
        poleArraySize = outputArraySize
    # arrays of forbidden innerproducts (only r=t, not z=te^{i\theta} as it has the same value for each \theta)
    # in order: t, t^2, sqrt(1-t^2), (1-t^2), f(t)
    outputRadiusArray = np.linspace(0, 1, outputArraySize, endpoint=True,dtype=np.longdouble)
    sqrdORA = np.square(outputRadiusArray)
    remainderORA = np.sqrt(1-sqrdORA)
    sqrdremainderORA = 1 - sqrdORA
    outputArray = np.zeros(outputArraySize)
    # Arrays for r_1, in order: r_1, r_1^2, sqrt(1-r_1^2), (1-r_1^2)
    poleRadiusArray = np.linspace(math.sqrt(1/2), 1, poleArraySize, endpoint=True,dtype=np.longdouble)
    sqrdPRA = np.square(poleRadiusArray)
    remainderPRA = np.sqrt(1-sqrdPRA)
    sqrdremainderPRA = 1 - sqrdPRA
    # Arrays for r_2, in order: r_2, r_2^2
    poleSecondRadius = np.linspace(0, 1, poleArraySize, endpoint=True,dtype=np.longdouble)
    sqrdSRA = np.square(poleSecondRadius)
    # weights for r_1 (the first coordinate where the cap is centered on)
    wpoleRadius = (((2**dim)*poleRadiusArray*(dim-1)*(np.power((1-np.square(poleRadiusArray)),(dim-2))))
                   / poleArraySize * (1 - math.sqrt(1/2)))

    # weights for r_2 (the second coordinate where the cap is centered on)
    wspoleSecondRadius = (2*poleSecondRadius*(dim-2)*((1-np.square(poleSecondRadius))**(dim-3)))/poleArraySize

    # loop over r_1 in [sqrt(1/2),1], r_2 in [0,1]
    for oneIdx, (radOne, sqrdRadOne, remainderRadOne,sqrdRemainderRadOne,weightRadOne) in enumerate(
            zip(poleRadiusArray,sqrdPRA,remainderPRA,sqrdremainderPRA,wpoleRadius)):
        intermediateOutput = np.zeros(outputArraySize)
        for twoIdx, (radTwo, sqrdRadTwo, weightRadTwo) in enumerate(zip(poleSecondRadius,sqrdSRA,wspoleSecondRadius)):
            # see latex for numerator and denominator of (2)
            arccosNumeratorArray = sqrdORA*sqrdRadOne + sqrdremainderORA*sqrdRemainderRadOne*sqrdRadTwo - 1/2
            arccosDenominatorArray = -2*outputRadiusArray*remainderORA*radOne*remainderRadOne*radTwo

            # Checks if the most disadvantageous/advantageous phase is too large/small respectively
            maskMustBeInIt = (-arccosNumeratorArray-arccosDenominatorArray<=0)
            maskCantBeInIt = (-arccosNumeratorArray + arccosDenominatorArray >= 0)

            # Fraction is b in [-1,1], where (1/2>=r^2_1 t^2 + c * r_1 t r_2\sqrt{1-r_1^2} \sqrt{1-t^2}
            # +r^2_2 (1-r_1^2) (1-t^2)) if and only if c in [-1,b]
            arccosFraction = np.clip(arccosNumeratorArray/arccosDenominatorArray,-1,1)

            # sets the correct fractions for the guaranteed values as a safety measure
            arccosFraction[maskCantBeInIt] = 1
            arccosFraction[maskMustBeInIt] = -1
            # find what fraction of cosines are in [b,1]
            arcCosArray = np.arccos(arccosFraction)/np.pi

            # Apply weight to this value for r_2 and add it to the total of r_1
            totalIterdy = weightRadTwo*arcCosArray
            np.add(intermediateOutput, totalIterdy, out=intermediateOutput)
        # Apply weight to the value for r_1 and add it to the total (hopefully mitigating some rounding errors)
        np.add(outputArray,weightRadOne*intermediateOutput,out=outputArray)

    return outputArray


def complexWeightPolyCreator(dim):
    if dim == 1:
        return 0
    else:
        coefficient = 2 * (dim - 1)
        rFactor = Polynomial([0,1],[0,1],[0,1])
        remaindersqrd = Polynomial([1,0,-1],[0,1],[0,1])
        return coefficient*rFactor*(remaindersqrd**(dim-2))

def integratedWeightPolyCreator(dim):
    if dim == 1:
        return 0
    else:
        remaindersqrd = Polynomial([1, 0, -1], [0, 1], [0, 1])
        return -Polynomial([1,0,-1],[0,1],[0,1])**(dim-1)


def alternateRadiusSpaceConstructor(size, dim, lb=0, ub=1):
    weightPoly = complexWeightPolyCreator(dim)
    integratedPoly = integratedWeightPolyCreator(dim)
    domainSize = integratedPoly(ub)-integratedPoly(lb)
    lbVal = integratedPoly(lb)
    dx = 1/(size)
    measureMiddleLb = dx/2
    measureMiddleUb = 1-dx/2
    integralPPFs = np.linspace(measureMiddleLb,measureMiddleUb,size,endpoint=True)
    scaledPPFs = integralPPFs*domainSize
    desiredRightVals = -(scaledPPFs+lbVal)
    remainderSquaredVals = np.power(desiredRightVals,1/(dim-1))
    rSquaredVals = 1-remainderSquaredVals
    rVals = np.sqrt(rSquaredVals)
    return rVals





if __name__ == '__main__':
    rVals2 = alternateRadiusSpaceConstructor(200,4,np.sqrt(1/2),1)
    complexDoubleCapv2(4,50,250)
    fz = complexWeightPolyCreator(4)
    cDim = 5
    outputArraySize = 513
    resolution = 1025
    invariantDoubleComplexCap = complexDoubleCap(cDim,outputArraySize,resolution)
    radialIntegral = radialIntegrator(invariantDoubleComplexCap, complexDimension=cDim)
    DCC = np.zeros(outputArraySize)
    rads = np.linspace(0,1,outputArraySize,endpoint=True)
    radsMask = (rads>=math.sqrt(1/2))
    DCC[radsMask] = 1
    DCCIntegral = radialIntegrator(DCC,complexDimension=cDim)
    randomSpherePoints = createRandomPointsReal(4, 7, checkForMistakesBool=True)
    complexPointsRadii = complexSphereDensity(3,200)
    complexPoints = np.ones((200,250))
    shape = [1] * 2
    shape[0] = 200
    complexPointsRadii = complexPointsRadii.reshape(shape)
    complexPoints = complexPoints * complexPointsRadii
    normalizedDatapoints = normalizeRadials(complexPoints)
    innerProductsMatrix = randomSpherePoints @ randomSpherePoints.T

    #
    # lp = LineProfiler()
    # lp_wrapper = lp(complexTester)
    # lp_wrapper()
    # lp.print_stats()

    print("done")

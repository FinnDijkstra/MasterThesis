import math

import main
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import quad
from matplotlib import cm
labelDict = {"RadAng":["Radius","Angle", "Result (log of -thetapart)"], "RadGamma":["Radius","Gamma", "Result"],
             "JacobiSol":["Innerproduct","Height", "Result"]}
testType = "JacobiSol"
dimension = 4

minGamma = 5
maxGamma = 15
maxDegree = 10


def compute_denominator(n, c=1 / np.sqrt(2)):
    """Compute the denominator D analytically."""

    def denom_integrand(x):
        return (1 - x ** 2) ** ((n - 3) / 2)

    D, _ = quad(denom_integrand, c, 1.0)
    return 2.0 * D

def bestCaseInnerproduct(x,y):
    return


def cap_fraction_grid(t_array, n, u1_points=200, u2_points=200):
    """
    Approximate cap_fraction for multiple t values using a fixed 2D grid.

    Parameters
    ----------
    t_array : array-like of floats
        Inner products at which to evaluate.
    n : int
        Dimension ≥ 2.
    u1_points : int
        Number of grid points for u1 in [-1,1].
    u2_points : int
        Number of grid points for u2 in [-1,1].

    Returns
    -------
    numpy.ndarray
        Approximate fractions for each t in t_array.
    """
    t_array = np.atleast_1d(t_array)
    c = 1.0 / np.sqrt(2.0)

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
    dPGTWeight = sinDPGT**(n-3)
    # dPGTWeight = ((doubleProjectiveGridTheta / math.pi) ** (((dimension - 3)/ 2) ) *
    #               (1 - doubleProjectiveGridTheta / math.pi) ** (((dimension - 3)/ 2) ))
    dPGTTotWeight = np.sum(dPGTWeight)


    # Create mask if cap wrt pole is on v
    maskCap = (np.abs(perspectivePoleInnerGrid) > math.sqrt(1/2))
    pPIGTotWeight = np.sum(pPIGWeight[maskCap])/2
    # Create mask if cap wrt pole has atleast 1 point that has innerproduct t
    term1 = np.abs(t_array[:, None] * perspectivePoleInnerGrid[None, :])
    term2 = np.sqrt(1 - t_array[:, None] ** 2) * np.sqrt(1 - perspectivePoleInnerGrid[None, :] ** 2)
    maskTCap = ((term1 + term2) > math.sqrt(1/2))
    maskTot = maskTCap * maskCap

    # For each t, for each cap, test if each cosine of a point (with innerproduct t) wrt e_2 is in the cap and add its
    # weight if that is the case
    outputTVol = np.zeros_like(t_array)
    for tIDX in range(t_array.shape[0]):
        t = t_array[tIDX]
        tOrth = math.sqrt(1-t**2)
        curTDensity = np.zeros(u1_points)
        for pole in range(u1_points):
            if maskCap[pole] & maskTCap[tIDX,pole]:
                poleInner = perspectivePoleInnerGrid[pole]
                poleOrth = sinPPIG[pole]
                innerFactor = t*poleInner
                OrthFactor = tOrth * poleOrth
                OrthComp = OrthFactor * cosDPGT
                innerPT = OrthComp + innerFactor
                capTAngMask = (innerPT > math.sqrt(1/2))
                curTDensity[pole] = np.sum(dPGTWeight[capTAngMask])/dPGTTotWeight
        outputTVol[tIDX] = np.inner(curTDensity, pPIGWeight)/pPIGTotWeight


    # u1 = np.linspace(-1, 1, u1_points)
    # u2 = np.linspace(-1, 1, u2_points)
    # du1 = u1[1] - u1[0]
    # du2 = u2[1] - u2[0]
    #
    # U1, U2 = np.meshgrid(u1, u2, indexing='ij')
    # mask_sphere_cap = (U1 ** 2 + U2 ** 2 <= 1.0) & (np.abs(U1) >= c)
    # mask_sphere_over = (U1 ** 2 + U2 ** 2 > 1.0) & (np.abs(U1) >= c)
    # mask_sphere = np.logical_or(mask_sphere_cap,mask_sphere_over)
    # # Precompute integrand on grid
    # integrand = np.zeros_like(U1)
    # integrand[mask_sphere_cap] = (1 - U1[mask_sphere_cap] ** 2 - U2[mask_sphere_cap] ** 2) ** ((n - 3) / 2)
    # integrand[mask_sphere_over] = (-1 + U1[mask_sphere_over] ** 2 + U2[mask_sphere_over] ** 2) ** ((n - 3) / 2)
    # # Compute denominator
    # D = compute_denominator(n, c)
    # # Calculate fractions for each t
    # fractions = []
    # for t in t_array:
    #     if abs(abs(t) - 1.0) < 1e-8:
    #         fractions.append(1.0)
    #         continue
    #     sqrt1mt2 = np.sqrt(1 - t ** 2)
    #     # Condition for inner product with second pole
    #     mask2 = np.abs(t * U1 + sqrt1mt2 * U2) >= c
    #     N = np.sum(integrand * (mask2 & mask_sphere)) * du1 * du2
    #     fractions.append(N / D)

    return outputTVol



def leftCalculator(k,beta,alpha):
    return math.comb(k+beta, k)/math.comb(k+alpha, k)


def interpolateDegree(location, left, right, degree, insideBrackets):
    lrDiff = left - right
    percLoc = (location+1)/2
    if insideBrackets:
        lPart = 1 - (percLoc**degree)
    else:
        lPart = (1-percLoc)**degree
    return 1 + lrDiff*lPart

denseDots = (0,(1,1))
dotDash = (0,(3,3,1,3))
dashes = (0,(3,3))
thetaVFunc = np.vectorize(main.thetaBasedOnThetapart)
interVFunc = np.vectorize(interpolateDegree)
if testType == "RadGamma":
    xMin = -1
    xMax = 1
    xSteps = 500
    xStep = (xMax - xMin) / xSteps

    X = np.linspace(xMin, xMax, xSteps)

    vfunc = np.vectorize(main.normedJacobiValue)
else:
    xMin = -1
    xMax = 1
    xSteps = 500
    xStep = (xMax-xMin)/xSteps
    # xStep = 0.00125

    X = np.linspace(xMin, xMax, xSteps)

    vfunc = np.vectorize(main.normedJacobiValue)
if "Rad" in testType:
    for gamma in range(minGamma, maxGamma):

        fig, ax = plt.subplots()
        for deg in range(maxDegree+1):
            even = (deg % 2 == 0)
            if even:
                signCur = 1
            else:
                signCur = -1
            sideNr = deg // 2
            colourCode = f"C{sideNr}"
            lineWidth = even*0.5 + 0.5
            Y = vfunc(degree=deg,alpha=dimension-2,beta=gamma,location=X)
            leftVal = leftCalculator(deg, gamma, dimension-2)
            linY = signCur*interVFunc(location=X, left=leftVal, right=1, degree=1, insideBrackets=False)
            outY = signCur*interVFunc(location=X, left=leftVal, right=1, degree=deg, insideBrackets=False)
            inY = signCur*interVFunc(location=X, left=leftVal, right=1, degree=deg, insideBrackets=True)

            ax.plot(X, Y, color=colourCode, linewidth=lineWidth, alpha=1)

            ax.plot(X, linY, color=colourCode, linewidth=lineWidth, alpha=0.5, linestyle=denseDots)
            ax.plot(X, outY, color=colourCode, linewidth=lineWidth, alpha=0.5, linestyle=dotDash)
            ax.plot(X, inY, color=colourCode, linewidth=lineWidth, alpha=0.5, linestyle=dashes)
            linYValid = (signCur*linY < signCur*Y)
            outYValid = (signCur * outY < signCur * Y)
            inYValid = (signCur * inY < signCur * Y)
            if np.any(linYValid[1:]):
                print(f"Linear not good for gamma {gamma}, degree {deg}")
            if np.any(outYValid[1:]):
                print(f"Outside bracket not good for gamma {gamma}, degree {deg}")
            if np.any(inYValid[1:]):
                print(f"Inside brackte not good for gamma {gamma}, degree {deg}")



        labels = labelDict.get(testType, ["X","Y", "Result"])
        normAbsMax = leftCalculator(maxDegree, gamma, dimension-2)
        zMin = min(-normAbsMax,-1)
        zMax = max(normAbsMax, 1)
        zDiff = zMax - zMin
        ax.set(xlim=(xMin, xMax), ylim=(zMin-zDiff/4, zMax+zDiff/4),
                xlabel=labels[0], ylabel=labels[1])
        plt.yscale("asinh")
        plt.show()

if testType == "JacobiSol":
    radialBool = False
    cosineBool = True
    definitionPoints = 250
    vfunc = np.vectorize(main.normedJacobiValue, otypes=[float])
    if radialBool:

        thetaSpace = np.linspace(0, math.pi, num=definitionPoints)
        innerProductTheta = np.cos(thetaSpace)


        jacobiObj = 1 / dimension * vfunc(degree=0, alpha=(dimension - 3) / 2, beta=(dimension - 3) / 2, location=innerProductTheta)

        # for angle in innerProductTheta:
        #     print(main.normedJacobiValue(degree=2,alpha=(dimension-3)/2,beta=(dimension-3)/2,
        #                                              location=angle))
        jacobiComp = (dimension - 1) / dimension * vfunc(location=innerProductTheta, degree=2, alpha=(dimension - 3) / 2,
                                                         beta=(dimension - 3) / 2
                                                         )
        jacobiSol = jacobiObj + jacobiComp
        circMeasure = 2 * (math.pi ** ((dimension - 1) / 2) / main.gammaFunction((dimension - 1) / 2))
        sphereMeasure = 2 * (math.pi ** ((dimension) / 2) / main.gammaFunction((dimension) / 2))
        sphereMeasure2 = 0
        # if dimension == 2:
        #     weightMeasure = np.ones_like(innerProductTheta)
        #     for i in range(1, weightMeasure.shape[0] - 1):
        #         weightMeasure[i] = abs(math.sin(thetaSpace[i])) ** (dimension - 2)
        #     weightMeasure[0] = 2 * weightMeasure[1] - weightMeasure[2]
        #     weightMeasure[-1] = 2 * weightMeasure[-2] - weightMeasure[-3]
        # else:
        #     weightMeasure = np.abs(np.sin(thetaSpace)) ** (dimension - 2)
        betaVFunc = np.vectorize(sp.special.betainc, otypes=[float])
        incBetaInt = betaVFunc(dimension/2,dimension/2, thetaSpace/math.pi)
        weightMeasure = (thetaSpace/math.pi)**((dimension/2)-1)*(1-thetaSpace/math.pi)**((dimension/2)-1)
        volumeDensity = weightMeasure * jacobiSol
        volumetwo = np.zeros_like(volumeDensity)
        doublecap = np.zeros_like(thetaSpace)
        for i in range(doublecap.shape[0]):
            if abs(innerProductTheta[i]) > math.sqrt(2) / 2:
                doublecap[i] = 1
        doubleCapDensity = doublecap * weightMeasure
        # for i in range(volumetwo.shape[0]//2):
        #     if i == 0:
        #         volumetwo[i] = volumeDensity[i]
        #     else:
        #         volumetwo[i] = volumetwo[i-1] + 2*volumeDensity[i]
        #         volumetwo[-i] = volumetwo[-i + 1] + 2 * volumeDensity[i]
        # volumetwo[volumetwo.shape[0]//2] = volumetwo[volumetwo.shape[0]//2 - 1] + volumeDensity[volumetwo.shape[0]//2]
        # volumetwo = volumetwo/volumetwo.shape[0]
        volume = np.zeros_like(volumeDensity)
        doubleCapVolume = np.zeros_like(volumeDensity)
        for i in range(volume.shape[0]):
            if i == 0:
                volume[i] = volumeDensity[i]
                doubleCapVolume[i] = doubleCapDensity[i]
                sphereMeasure2 += weightMeasure[i]
            else:
                volume[i] = volume[i - 1] + volumeDensity[i]
                doubleCapVolume[i] = doubleCapVolume[i - 1] + doubleCapDensity[i]
                sphereMeasure2 += weightMeasure[i]

        volume = volume / sphereMeasure2
        doubleCapVolume = doubleCapVolume / sphereMeasure2
        print(doubleCapVolume[-1])
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # ax.plot(theta, jacobiSol)
        ax.plot(thetaSpace, weightMeasure, label='Weight Density')
        # ax.plot(theta, volumeDensity)
        # ax.plot(theta, volumetwo)
        volumeLine, = ax.plot(thetaSpace, volume, label='Volume')
        # ax.plot(theta, doublecap)
        # ax.plot(theta, doubleCapDensity)
        doubleCapLine, = ax.plot(thetaSpace, doubleCapVolume, label='Double Cap')
        # ax.plot(theta, dimension/(dimension-1)*jacobiComp)
        # ax.set_rmax(np.max(jacobiSol))
        totVol = volume[-1]
        totVol = 1.1
        ax.set_rmax(totVol)
        # ax.set_rticks([-1, -0.5, 0, 0.5,1])  # Less radial ticks
        nrOfTicks = 5
        ax.set_rticks([totVol*i/nrOfTicks for i in range(nrOfTicks+1)])
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        xT = plt.xticks()[0]
        xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
              r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
        plt.xticks(xT, xL)
        ax.set_title(f"A line plot on a polar axis for dimension {dimension}", va='bottom')
        ax.legend(handles=[volumeLine, doubleCapLine], loc='upper left', bbox_to_anchor=(0.01, 0.99))
        plt.show()
    if cosineBool:
        bucketNr = 20

        r = np.linspace(-1, 1, num=definitionPoints)
        theta = np.arccos(r)
        a = np.empty(bucketNr + 1)
        for i in range(bucketNr + 1):
            # invert regularized Beta:
            #   sp.special.betaincinv(a, b, p) returns x in [0,1] with I_x(a,b)=p
            x = sp.special.betaincinv((dimension - 1) / 2, (dimension - 1) / 2, i / bucketNr)
            a[i] = 2 * x - 1

        # 2) midpoints in mass
        c = np.empty(bucketNr)
        for i in range(bucketNr):
            xmid = sp.special.betaincinv((dimension - 1) / 2, (dimension - 1) / 2, (i + 0.5) / bucketNr)
            c[i] = 2 * xmid - 1
        jacobiMidObj = 1 / dimension * vfunc(degree=0, alpha=(dimension - 3) / 2, beta=(dimension - 3) / 2, location=c)
        jacobiMidComp = (dimension - 1) / dimension * vfunc(location=c, degree=2, alpha=(dimension - 3) / 2,
                                                                beta=(dimension - 3) / 2)
        jacobiMidSol = jacobiMidObj + jacobiMidComp

        jacobiObj = 1 / dimension * vfunc(degree=0, alpha=(dimension - 3) / 2, beta=(dimension - 3) / 2, location=r)

        # for angle in innerProductTheta:
        #     print(main.normedJacobiValue(degree=2,alpha=(dimension-3)/2,beta=(dimension-3)/2,
        #                                              location=angle))
        jacobiComp = (dimension - 1) / dimension * vfunc(location=r, degree=2, alpha=(dimension - 3) / 2,
                                                         beta=(dimension - 3) / 2
                                                         )
        jacobiSol = jacobiObj + jacobiComp
        circMeasure = 2 * (math.pi ** ((dimension - 1) / 2) / main.gammaFunction((dimension - 1) / 2))
        sphereMeasure = 2 * (math.pi ** ((dimension) / 2) / main.gammaFunction((dimension) / 2))
        sphereMeasure2 = 0
        if dimension == 2:
            weightMeasure = np.ones_like(r)
            for i in range(1, weightMeasure.shape[0] - 1):
                weightMeasure[i] = abs(math.sin(theta[i])) ** (dimension - 3)
            weightMeasure[0] = 2 * weightMeasure[1] - weightMeasure[2]
            weightMeasure[-1] = 2 * weightMeasure[-2] - weightMeasure[-3]
        else:
            weightMeasure = np.abs(np.sin(theta)) ** (dimension - 3)
        volumeDensity = weightMeasure * jacobiSol
        volumetwo = np.zeros_like(volumeDensity)
        doublecap = np.zeros_like(r)
        # interSectFunc = np.vectorize(cap_fraction)
        invariantDoubleCap = cap_fraction_grid(t_array=r, n=dimension, u1_points=2*definitionPoints, u2_points=2*definitionPoints)
        for i in range(doublecap.shape[0]):
            if abs(r[i]) > math.sqrt(2) / 2:
                doublecap[i] = 1
        doubleCapDensity = doublecap * weightMeasure
        invDCDensity = invariantDoubleCap * weightMeasure
        # for i in range(volumetwo.shape[0]//2):
        #     if i == 0:
        #         volumetwo[i] = volumeDensity[i]
        #     else:
        #         volumetwo[i] = volumetwo[i-1] + 2*volumeDensity[i]
        #         volumetwo[-i] = volumetwo[-i + 1] + 2 * volumeDensity[i]
        # volumetwo[volumetwo.shape[0]//2] = volumetwo[volumetwo.shape[0]//2 - 1] + volumeDensity[volumetwo.shape[0]//2]
        # volumetwo = volumetwo/volumetwo.shape[0]
        volume = np.zeros_like(volumeDensity)
        doubleCapVolume = np.zeros_like(volumeDensity)
        invDCVol = np.zeros_like(invariantDoubleCap)
        totalVolume = np.zeros_like(weightMeasure)
        for i in range(volume.shape[0]):
            if i == 0:
                volume[i] = volumeDensity[i]
                doubleCapVolume[i] = doubleCapDensity[i]
                invDCVol[i] = invDCDensity[i]
                sphereMeasure2 += weightMeasure[i]
                totalVolume[i] = weightMeasure[i]
            else:
                volume[i] = volume[i - 1] + volumeDensity[i]
                doubleCapVolume[i] = doubleCapVolume[i - 1] + doubleCapDensity[i]
                invDCVol[i] = invDCVol[i - 1] + invDCDensity[i]
                sphereMeasure2 += weightMeasure[i]
                totalVolume[i] = totalVolume[i-1] + weightMeasure[i]

        volume = volume / sphereMeasure2
        doubleCapVolume = doubleCapVolume / sphereMeasure2
        invDCVol = invDCVol / sphereMeasure2
        fig, ax = plt.subplots()
        # volumeLine, = ax.plot(r, jacobiSol, label='Invariant kernel solution')
        # ConstpartLine, = ax.plot(r, weightMeasure, label='Density of the sphere measure')
        # ax.plot(theta, volumeDensity)

        ConstpartLine, = ax.plot(r, totalVolume/dimension/sphereMeasure2, label="0'th degree part of solution")
        volumeLine, = ax.plot(r, volume, label='Invariant kernel solution')
        # doubleCapLine, = ax.plot(r, doublecap, label='Double cap')
        # ax.plot(theta, doubleCapDensity)
        doubleCapLine, = ax.plot(r, doubleCapVolume, label='Double cap')
        invDoubleCapLine, = ax.plot(r, invDCVol, label='Invariant double cap')
        # invDoubleCapLine, = ax.plot(r, invariantDoubleCap, label='Invariant double cap')
        # ax.plot(theta, dimension/(dimension-1)*jacobiComp)
        # ax.set_rmax(np.max(jacobiSol))
        # totVol = volume[-1]
        # ax.set_rmax(totVol)
        # ax.set_rticks([-1, -0.5, 0, 0.5,1])  # Less radial ticks
        # nrOfTicks = 5
        # ax.set_rticks([totVol*i/nrOfTicks for i in range(nrOfTicks+1)])
        # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        # ax.set_ylim(0,1.4)
        # 4) shade buckets
        # widths = a[:-1] - a[1:]
        # area = widths.max()
        # heights = area / widths
        # cmap = plt.get_cmap('tab20')
        # for i in range(bucketNr):
        #     left = a[i + 1]
        #     ax.bar(left, heights[i],
        #            width=widths[i],
        #            align='edge',
        #            facecolor=cmap(i / bucketNr),
        #            alpha=0.4)
        #
        # # 5) draw verticals at the midpoints and label
        # for idx in range(c.shape[0]):
        #     ci = c[idx]
        #     yi = jacobiMidSol[idx]
        #     ax.plot([ci, ci], [0, yi], 'k-')
        #     label = rf'$f(c_{{{idx}}})$'
        #     ax.text(ci, yi,
        #             label,
        #             ha='left', va='bottom',
        #             fontsize=9)
        # xT = plt.xticks()[0]
        # xL = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$',
        #       r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']
        # plt.xticks(xT, xL)
        ax.set_title(f"Integrals in dimension {dimension}", va='bottom') # A line plot on a polar axis for
        ax.legend(handles=[volumeLine, ConstpartLine, doubleCapLine,invDoubleCapLine], loc='upper left', bbox_to_anchor=(0.01, 0.99))
        plt.show()



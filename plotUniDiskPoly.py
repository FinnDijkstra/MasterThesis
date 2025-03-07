import main
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
labelDict = {"RadAng":["Radius","Angle", "Result (log of -thetapart)"], "RadGamma":["Radius","Gamma", "Result"]}
testType = "RadAng"
dimension = 5
maxDegree = 10
thetaVFunc = np.vectorize(main.thetaBasedOnThetapart)
if testType == "RadGamma":
    xMin = 0.90
    xMax = 0.999
    xStep = 0.00125

    yMin = 0
    yMax = 20
    yStep = 0.05

    X = np.arange(xMin, xMax, xStep)
    Y = np.arange(yMin, yMax, yStep)


    X, Y = np.meshgrid(X, Y)
    vfunc = np.vectorize(main.findMinFixedGammaVar)
else:
    xMin = 0.1
    xMax = 0.999
    xSteps = 200
    xStep = (xMax-xMin)/xSteps
    # xStep = 0.00125

    yMin = 0.0001
    yMax = 0.95
    ySteps = 200
    yStep = (yMax-yMin)/ySteps
    # yStep = 0.000125

    X = np.arange(xMin, xMax, xStep)
    Y = np.arange(yMin, yMax, yStep)

    X, Y = np.meshgrid(X, Y)
    vfunc = np.vectorize(main.findMinVar)
if testType == "RadGamma":
    Z = vfunc(alpha=dimension-2, gamma=Y, forbiddenRadius=X, forbiddenAngle=0.0005)[0]
else:
    Z = vfunc(alpha=dimension - 2, forbiddenRadius=X, forbiddenAngle=Y)[0]
# Z = thetaVFunc(Z)
# Z = np.log(Z)
Z = np.log(-Z)
zMin = Z.min()
zMax = Z.max()
zDiff = zMax - zMin
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='royalblue', lw=0.5, rstride=10, cstride=10,
                        alpha=0.7)

ax.contour(X, Y, Z, zdir='z', offset=zMin-zDiff/4, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='x', offset=xMin, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='y', offset=yMax, cmap='coolwarm')
ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax), zlim=(zMin-zDiff/4, zMax+zDiff/4),
   xlabel=labelDict[testType][0], ylabel=labelDict[testType][1], zlabel=labelDict[testType][2])


plt.show()




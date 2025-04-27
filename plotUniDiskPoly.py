import math

import main
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
labelDict = {"RadAng":["Radius","Angle", "Result (log of -thetapart)"], "RadGamma":["Radius","Gamma", "Result"]}
testType = "RadK"
dimension = 5
maxDegree = 200
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

if testType == "RadK":
    xMin = math.sqrt(1/8)
    xMax = math.sqrt(1/2)
    xSteps = 20
    xStep = (xMax - xMin) / xSteps

    yMin = 5
    yMax = 500
    ySteps = 10
    yStep = (yMax - yMin) / ySteps

    X = np.arange(xMin, xMax, xStep)
    Y = np.arange(yMin, yMax, yStep)


    X, Y = np.meshgrid(X, Y)
    vfunc = np.vectorize(main.realDiskPoly)
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
if testType == "RadK":
    Z = vfunc(alpha=dimension-2, gamma=Y, k=1, radius=X, angle=0.0)
    for deg in range(2, maxDegree+1):
        Zcur = vfunc(alpha=dimension-2, gamma=Y, k=deg, radius=X, angle=0.0005)
        np.minimum(Z, Zcur, out=Z)
else:
    Z = vfunc(alpha=dimension - 2, forbiddenRadius=X, forbiddenAngle=Y)[0]
# Z = thetaVFunc(Z)
# Z = np.log(Z)
# Z = np.log(-Z)
from mpl_toolkits.mplot3d import Axes3D


def plot_complex_sphere_density(n, Nr=100, Nphi=100):
    """
    Plots the joint density of z = <v,u> on the complex sphere S^{2n-1},
    with x = Re(z), y = Im(z), and z = relative density.

    n: dimension parameter of the complex sphere S^{2n-1}
    Nr, Nphi: resolution in radial and angular directions
    """
    # Create polar grid
    r = np.linspace(0, 1, Nr)
    phi = np.linspace(0, 2 * np.pi, Nphi)
    R, Phi = np.meshgrid(r, phi)

    # Convert to Cartesian
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)

    # Compute density: p(r) = 2*(n-1)*r*(1-r^2)^(n-2), uniform in phi
    Z = 2 * (n - 1) * R * (1 - R ** 2) ** (n - 2)

    # Normalize so maximum height is 1
    # Z = Z / Z.max()

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.8, cmap='coolwarm')

    ax.set_xlabel('Re⟨v,u⟩')
    ax.set_ylabel('Im⟨v,u⟩')
    ax.set_zlabel('Relative density')
    ax.set_title(rf'Complex sphere density in $\mathbb{{C}}^{{{n}}}$')
    plt.show()


# Example usage:
plot_complex_sphere_density(n=3, Nr=100, Nphi=100)
zMin = Z.min()
zMax = Z.max()
zDiff = zMax - zMin
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='royalblue', lw=0.5, rstride=10, cstride=10,
                        alpha=0.7)
labels = labelDict.get(testType, ["X","Y", "Result"])
ax.contour(X, Y, Z, zdir='z', offset=zMin-zDiff/4, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='x', offset=xMin, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='y', offset=yMax, cmap='coolwarm')
ax.set(xlim=(xMin, xMax), ylim=(yMin, yMax), zlim=(zMin-zDiff/4, zMax+zDiff/4),
   xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])


plt.show()




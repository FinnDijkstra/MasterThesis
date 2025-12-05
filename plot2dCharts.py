import math

from numpy import dtype

import OldFunctions
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import scipy as sp
from scipy.integrate import quad
from matplotlib.colors import LightSource
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.tri as mtri
from matplotlib import cm
import OrthonormalPolyClasses
labelDict = {"RadAng":["Radius","Angle", "Result (log of -thetapart)"], "RadGamma":["Radius","Gamma", "Result"],
             "JacobiSol":["Innerproduct","Height", "Result"]}
testType = "DiskCheck"
dimension = 4

minGamma = 5
maxGamma = 15
maxDegree = 10


def plotDubbel3dMethod(r,theta, x, y, z, zlabel="f(z)",fullZLabel="f(z)"):
    zMax = np.max(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)

    ax.contourf(x[:, 59:], y[:, 59:], z[:, 59:],
                zdir='z', offset=-0.0, cmap=plt.cm.RdYlBu, levels=15,zorder=1)
    # ax.contour(x[:, :60], y[:, :60], z[:, :60],
    #             zdir='z', offset=-0.0, cmap=plt.cm.RdYlBu, levels=8, zorder=1)
    ax.plot_wireframe(x[:, :60], y[:, :60], z[:, :60],
                 colors="k", linewidth=0.6, rcount=1, ccount=1,zorder=3)
    ax.plot_wireframe(x[:, :60], y[:, :60], np.zeros_like(z[:, :60]),
                      colors="k", linewidth=0.6, rcount=1, ccount=1, zorder=1, linestyles="dashed")
    surfaceInfo = ax.plot_surface(x[:, :60], y[:, :60], z[:, :60],
                    edgecolor='none',
                    cmap="RdYlBu", alpha=0.8)

    # ax.contourf(xFull, yFull, dualSmoothed, zdir='y', offset=1, cmap=plt.cm.RdYlBu)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-0, 0.5),
           xlabel=r'$\Re (z)$', ylabel=r'$\Im (z)$', zlabel=zlabel)

    # ticks_Inp = np.linspace(-1, 1, 9,endpoint=True)
    # ticks_Out = np.linspace(0, zMax, 5, endpoint=True)
    # # ticklabels_pi = [f"{round(t / np.pi, 1)}π" for t in ticks_pi]
    # ax.set_xticks(ticks_Inp, labels=["", "", "", "", "", "", "", "",""])
    # ax.xaxis.line.set_color((0, 0, 0, 0))
    #
    # # ax.set_xlabel(r"$r$")
    # ax.set_yticks(ticks_Inp, labels=["", "", "", "", "", "", "", "",""])
    # ax.yaxis.line.set_color((0, 0, 0, 0))
    # # ax.set_yticklabels(ticklabels_pi)
    # # ax.set_ylabel(r"$\theta_1$")
    # ax.set_zticks(ticks_Out, labels=["", "", "", "", ""])
    # ax.zaxis.line.set_color((0, 0, 0, 0))
    # # ax.set_zticklabels(ticklabels_pi)
    # # ax.set_zlabel(r"$\theta_2$", labelpad=10, rotation=90)
    # ax.set_xlabel("")  # clear the default label
    #
    # ax.text(
    #     1.2, 0.2, 0.2*zMax,  # x, y, z position (adjust as needed)
    #     r'$\Re (z)$', fontsize=14,
    #     ha='center', va='center'
    # )
    # ax.set_ylabel("")  # clear the default label
    #
    # ax.text(
    #     0, -1.2, 0,  # x, y, z position (adjust as needed)
    #     r'$\Im (z)$', fontsize=14,
    #     ha='center', va='center'
    # )
    # ax.set_zlabel("")  # clear the default label
    #
    # ax.text(
    #     0, 0, 1.2*zMax,  # x, y, z position (adjust as needed)
    #     zlabel, fontsize=14,
    #     ha='center', va='center'
    # )
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(0, zMax)
    # # ax.set_box_aspect([2 * np.pi, 2 * np.pi, 2 * np.pi])
    # ax.tick_params(axis='x', which='both', color=[0, 0, 0, 0])
    # ax.tick_params(axis='y', which='both', color=[0, 0, 0, 0])
    # ax.tick_params(axis='z', which='both', color=[0, 0, 0, 0])
    # ax.view_init(elev=15., azim=15, roll=0)
    # draw_custom_r_ticks(ax,zMax)
    # draw_custom_theta1_ticks(ax,zMax)
    # draw_custom_theta2_ticks(ax,zMax)
    # rstride=1, cstride=1, alpha=0.9, cmap='coolwarm', zorder=0.75, zsort="min")
    ax.view_init(elev=25, azim=-75, roll=0)
    fig.colorbar(surfaceInfo, ax=ax,pad=0.10,shrink=0.6)
    # plt.tight_layout()
    ax.tick_params("x", which="both", labelbottom=True)
    ax.set_xticks(np.linspace(-1, 1, 5, endpoint=True))
    # ax.tick_params("y",which="both", pad=0.2)
    ax.tick_params("y", which="both", labelright=True, labelbottom=False, labelleft=False, labeltop=False)
    ax.set_yticks(np.linspace(-1, 1, 5, endpoint=True))
    ax.set_zticks(round(zMax,1)-np.arange(0,round(zMax,1), 0.1))
    plt.show()


    fig, ax = plt.subplots(figsize=(6, 6),subplot_kw={'projection': 'polar'})
    # ax.set_aspect('equal')

    contourInfo = ax.contourf(theta[:, :], r[:, :], z[:, :],
                 cmap=plt.cm.RdYlBu, levels=10)
    ax.contour(contourInfo, colors='k', linewidths=0.5)



    theta_ticks = [
        0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        5 * np.pi / 6,
        np.pi,
        7 * np.pi / 6,
        5 * np.pi / 4,
        4 * np.pi / 3,
        3 * np.pi / 2,
        5 * np.pi / 3,
        7 * np.pi / 4,
        11 * np.pi / 6
    ]

    theta_labels = [
        r'$0$',
        r'$\frac{1}{6}\pi$',
        r'$\frac{1}{4}\pi$',
        r'$\frac{1}{3}\pi$',
        r'$\frac{1}{2}\pi$',
        r'$\frac{2}{3}\pi$',
        r'$\frac{3}{4}\pi$',
        r'$\frac{5}{6}\pi$',
        r'$\pi$',
        r'$1\frac{1}{6}\pi$',
        r'$1\frac{1}{4}\pi$',
        r'$1\frac{1}{3}\pi$',
        r'$1\frac{1}{2}\pi$',
        r'$1\frac{2}{3}\pi$',
        r'$1\frac{3}{4}\pi$',
        r'$1\frac{5}{6}\pi$'
    ]
    ax.set_thetagrids(np.degrees(theta_ticks), labels=theta_labels)
    # xT = plt.xticks()[0]
    # xL = ['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$',
    #            r'$\pi$', r'$1\frac{1}{4}\pi$', r'$1\frac{1}{2} \pi$', r'$1\frac{3}{4} \pi$']
    # plt.xticks(xT, xL)
    ax.set_rlabel_position(-190)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_color('white')
        tick.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='black')
        ])
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(15)  # doubled from ~10
        label.set_color('white')
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='black')
        ])
    ax.grid(alpha=0.525)
    cbar = fig.colorbar(contourInfo,ax=ax,pad=0.10,shrink=0.8)
    cbar.set_label(fullZLabel)
    # plt.tight_layout()

    plt.show()



def draw_line(ax, p1, p2, shadowBool=False, guidesBool=False, **kwargs):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], **kwargs)

    # Base for derived lines (shadow/guides)
    base = kwargs.copy()
    base['alpha'] = kwargs.get('alpha', 1) / 2

    if shadowBool:
        shadow_kwargs = base.copy()
        if shadow_kwargs.get("marker", "0") == "_":
            shadow_kwargs["marker"] = "|"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [0, 0], **shadow_kwargs)

    if guidesBool:
        line_kwargs = base.copy()
        line_kwargs["linestyle"] = (0, (5, 10))
        line_kwargs["linewidth"] = kwargs.get("linewidth", 1) / 2
        ax.plot([p1[0], p1[0]], [p1[1], p1[1]], [p1[2], 0], **line_kwargs)
        ax.plot([p2[0], p2[0]], [p2[1], p2[1]], [p2[2], 0], **line_kwargs)


def draw_custom_r_ticks(ax, zMax, ticks=np.linspace(-1,1,5,endpoint=True),tickwidth=1):
    zRad = zMax/2
    for tick in ticks:
        if tick == 0:
            ax.text(tick, 0.05, -0.1*zMax, f"{int(tick)}", color='black', ha='right', va='top')
        elif tick == 1:
            ax.text(tick, 0.05, -0.1*zMax, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [0, 0], [-0.2*zRad, 0.2*zRad], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [-0.2, 0.2], [0, 0], color='black', linewidth=tickwidth)
        elif tick == -1:
            ax.text(tick, 0.05, -0.1*zMax, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [0, 0], [-0.2*zRad, 0.2*zRad], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [-0.2, 0.2], [0, 0], color='black', linewidth=tickwidth)
        else:
            ax.text(tick, 0.05, -0.1*zMax, f"{tick:.1f}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.1], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [0, 0], [-0.1*zRad, 0.1*zRad], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [-0.1, 0.1], [0, 0], color='black', linewidth=tickwidth)

    draw_line(ax,[-1,0,0],[1,0,0],color="black", linewidth=0.7)





def draw_custom_theta1_ticks(ax,zMax, ticks=np.linspace(-1,1,5,endpoint=True), tickwidth=1, ticklength=0.1):
    zRad = zMax/2
    for tick in ticks:
        if tick == 0:
            continue
        elif tick == 1:
            ax.text( 0.05,tick, -0.1*zMax, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([0, 0],[tick, tick], [-0.2*zRad, 0.2*zRad], color='black', linewidth=tickwidth)
            ax.plot([-0.2, 0.2],[tick, tick], [0, 0], color='black', linewidth=tickwidth)
        elif tick == -1:
            ax.text( 0.05,tick, -0.1*zMax, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([0, 0], [tick, tick], [-0.2*zRad, 0.2*zRad], color='black', linewidth=tickwidth)
            ax.plot( [-0.2, 0.2],[tick, tick], [0, 0], color='black', linewidth=tickwidth)
        else:
            ax.text( 0.05,tick, -0.1*zMax, f"{tick:.1f}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.1], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([0, 0],[tick, tick],  [-0.1*zRad, 0.1*zRad], color='black', linewidth=tickwidth)
            ax.plot( [-0.1, 0.1],[tick, tick], [0, 0], color='black', linewidth=tickwidth)

    draw_line(ax,[0,-1,0],[0,1,0],color="black", linewidth=0.7)



def draw_custom_theta2_ticks(ax, zMax, ticks=None, tickwidth=1, ticklength=0.1):
    if ticks is None:
        ticks = np.linspace(0.0, zMax, 5, endpoint=True)

    for tick in ticks:
        if tick == 0:
            continue
        elif tick == zMax:
            ax.text(tick, 0.05, tick, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([0, 0],[-0.2, 0.2],[tick, tick], color='black', linewidth=tickwidth)
            ax.plot([-0.2, 0.2],[0, 0],[tick, tick], color='black', linewidth=tickwidth)
        else:
            ax.text(tick, 0.05, tick, f"{tick:.2f}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.1], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([-0.1, 0.1], [0, 0],[tick, tick], color='black', linewidth=tickwidth)
            ax.plot([0, 0], [-0.1, 0.1],[tick, tick],  color='black', linewidth=tickwidth)

    draw_line(ax, [0,0,0], [0,0,zMax], color="black", linewidth=0.7)




def plotBQPResults(fileLocation,baseLocation):
    df = pd.read_excel(fileLocation)
    dfBase = pd.read_excel(baseLocation)

    radMesh = df["radius"].values.reshape((61,60))
    radShortMesh = np.flip(radMesh[:,:-1],axis=1)
    radFull = np.concatenate([radMesh,radShortMesh],axis=1)
    zeroRad = np.zeros((1,radFull.shape[1]))
    radFull = np.concatenate([zeroRad,radFull], axis=0)
    thetaMesh = df["theta"].values.reshape((61,60))
    thetaShortMesh = 2*np.pi-np.flip(thetaMesh[:,:-1],axis=1)
    thetaFull = np.concatenate([thetaMesh,thetaShortMesh],axis=1)
    zeroTheta = thetaFull[0].copy()[None,:]
    thetaFull = np.concatenate([zeroTheta, thetaFull], axis=0)

    cosFull = np.cos(thetaFull)
    sinFull = np.sin(thetaFull)
    xFull = np.multiply(radFull,cosFull)
    yFull = np.multiply(radFull,sinFull)

    # baseMesh = df["max_starting_objective"].values.reshape((61,60))
    # baseShortMesh = np.flip(baseMesh[:,:-1],axis=1)
    # baseFull = np.concatenate([baseMesh,baseShortMesh],axis=1)
    baseMesh = dfBase["min_dual_objective"].values.reshape((61, 60))
    baseShortMesh = np.flip(baseMesh[:, :-1], axis=1)
    baseFull = np.concatenate([baseMesh, baseShortMesh], axis=1)
    zeroBase = 0.25 * np.ones((1, radFull.shape[1]))
    baseFull = np.concatenate([zeroBase, baseFull], axis=0)
    dualMesh = df["min_dual_objective"].values.reshape((61,60))
    dualShortMesh = np.flip(dualMesh[:,:-1],axis=1)
    dualFull = np.concatenate([dualMesh,dualShortMesh],axis=1)
    zeroDual = np.mean(dualFull[0]) * np.ones((1, radFull.shape[1]))
    dualFull = np.concatenate([zeroDual, dualFull], axis=0)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xFull, yFull, -baseFull, rstride=1, cstride=1, alpha=0.9, cmap='coolwarm', zorder=0.75, zsort="min")
    # ax.view_init(elev=45, azim=-135, roll=0)
    # plt.show()


    # ratioFull = dualFull
    # thetaDim = ratioFull.shape[1]
    # atZeroArray = np.ones((1, thetaDim)) * (0.25 - 0.205317) / 0.25
    # atOneArray = np.zeros_like(atZeroArray)
    # ratioSmoothed = ratioFull
    # smoothingFac = 0.15
    # remainingFac = 1.0 - 4 * smoothingFac
    # nrOfSmoothings = 4
    plotDubbel3dMethod(radFull, thetaFull, xFull, yFull, baseFull, "Base objective value","Base objective value")
    dualSmoothed = gaussian_filter(dualFull, sigma=1, mode=("nearest", "mirror"), radius=4)
    plotDubbel3dMethod(radFull,thetaFull,xFull,yFull,dualSmoothed, "BQP objective value","BQP objective value")
    # for _ in range(nrOfSmoothings):
    #     ratioExtended = np.concatenate([atZeroArray, ratioSmoothed, atOneArray], axis=0)
    #     thetaIndices = np.arange(thetaDim)
    #     ratioSmoothed = (remainingFac * ratioExtended[1:-1, :]
    #                      + smoothingFac * ratioExtended[:-2, :] + smoothingFac * ratioExtended[2:, :]
    #                      + smoothingFac * ratioExtended[1:-1, thetaIndices - 1] + smoothingFac * ratioExtended[
    #                          1:-1, thetaIndices - thetaDim + 1])
    # ax.plot_surface(xFull, yFull, dualSmoothed,
    #                 cmap=plt.cm.RdYlBu, alpha=0.8)

    largestFull = np.maximum(dualSmoothed,baseFull)
    ratioFull = 1-np.divide(dualSmoothed,largestFull)

    # thetaDim = ratioFull.shape[1]
    # atZeroArray = np.ones((1,thetaDim))*(0.25-0.205317)/0.25
    # atOneArray = np.zeros_like(atZeroArray)
    # ratioSmoothed = ratioFull
    # smoothingFac = 0.15
    # remainingFac = 1.0-4*smoothingFac
    # nrOfSmoothings = 4
    # for _ in range(nrOfSmoothings):
    #     ratioExtended = np.concatenate([atZeroArray, ratioSmoothed, atOneArray], axis=0)
    #     thetaIndices = np.arange(thetaDim)
    #     ratioSmoothed = (remainingFac*ratioExtended[1:-1,:]
    #                      +smoothingFac*ratioExtended[:-2,:]+smoothingFac*ratioExtended[2:,:]
    #                      +smoothingFac*ratioExtended[1:-1,thetaIndices-1]+smoothingFac*ratioExtended[1:-1,thetaIndices-thetaDim+1])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=45, azim=-65, roll=0)
    # ax.plot_surface(xFull, yFull, ratioSmoothed, rstride=3, cstride=3, alpha=0.9, cmap='coolwarm', zorder=0.75, zsort="min")
    # plt.show()
    #
    # largestFull = np.maximum(dualFull, baseFull)
    # ratioFull = 1 - np.divide(dualFull, largestFull)
    ratioSmoothed = gaussian_filter(ratioFull, sigma=1, mode=("nearest","mirror"))
    plotDubbel3dMethod(radFull, thetaFull, xFull, yFull, ratioSmoothed, "(Base-BQP)/Base","(Base-BQP)/Base")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=90, azim=-85, roll=0)
    # ax.plot_surface(xFull, yFull, ratioSmoothed, rstride=1, cstride=1, alpha=0.9, cmap='coolwarm', zorder=0.75,
    #                 zsort="min")
    # plt.show()





def groupedPolarPlot(rMesh,thetaMesh,bestGammaArray,bestKArray,gammaSteps,kSteps,titleString,
                     legendBool=True, plot3dBool=False, minVals=0,comDim=2):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=500, subplot_kw={'projection': 'polar'})
    ax.set_aspect('equal')
    # ax.axis('off')
    gammaNonZero = (bestGammaArray > 0)
    kNonZero = (bestKArray > 0)
    bothNonzero = kNonZero * gammaNonZero
    onlyK = kNonZero^bothNonzero
    onlyGamma = gammaNonZero^bothNonzero
    kLeft = kNonZero.copy()
    listPerK = []
    for kVal in range(1,kSteps):
        kThisValMask = (bestKArray == kVal)
        listPerK.append(kThisValMask)
        kLeft = kLeft^kThisValMask
    listPerK.append(kLeft)

    gammaLeft = gammaNonZero.copy()
    listPerGamma = []
    for gammaVal in range(1, gammaSteps):
        gammaThisValMask = (bestGammaArray == gammaVal)
        listPerGamma.append(gammaThisValMask)
        gammaLeft = gammaLeft^gammaThisValMask
    listPerGamma.append(gammaLeft)
    # Fill regions
    minInt = 0.6
    counteredThisPoint = np.zeros_like(thetaMesh,dtype=bool)

    fullColours = {}

    kColours = {}
    for kVal in range(1,kSteps+1):
        kIntensity = (1-minInt)*((kVal-1)/(kSteps-1)) + minInt
        kValMask = listPerK[kVal-1]
        onlyKThis = kValMask*onlyK
        counteredThisPoint |= onlyKThis
        kShift = ((kVal-1)/(kSteps-1))
        curColour = (1*kIntensity, 0.5*np.power(kIntensity * kShift,1), 0.2*kIntensity)
        kColours[kVal] = curColour
        fullColours[(kVal, 0)] = curColour
        ax.contourf(thetaMesh, rMesh, onlyKThis.astype(float), levels=[0.5, 1.5], colors=[curColour],
                    alpha=1)
        ax.contour(thetaMesh, rMesh, onlyKThis.astype(float), levels=[0.5, 1.5], colors=[(0.5 * kIntensity,
                                                                                      0.1*kIntensity,
                                                                                      0.1*kIntensity)],
                    alpha=1, linewidths=0.8)
    gammaColours = {}
    for gammaVal in range(1,gammaSteps+1):
        gammaIntensity = (1-minInt)*((gammaVal-1)/(gammaSteps-1)) + minInt
        gammaValMask = listPerGamma[gammaVal-1]
        onlyGammaThis = gammaValMask * onlyGamma
        counteredThisPoint |= onlyGammaThis
        curColour = (0.6*gammaIntensity, 0.1*gammaIntensity, 1*np.power(gammaIntensity,1))
        gammaColours[gammaVal] = curColour
        fullColours[(0, gammaVal)] = curColour
        ax.contourf(thetaMesh, rMesh, onlyGammaThis.astype(float), levels=[0.5, 1.5],
                    colors=[curColour], alpha=1)
        ax.contour(thetaMesh, rMesh, onlyGammaThis.astype(float), levels=[0.5, 1.5],
                    colors=[(0.3 * gammaIntensity, 0.05*gammaIntensity, 0.5 * np.power(gammaIntensity,2))], alpha=1, linewidths=0.8)
    combinedColours = {}
    for kVal in range(1, kSteps+1):
        kValMask = listPerK[kVal - 1]
        # kIntensity = (kVal)/kSteps
        for gammaVal in range(1, gammaSteps+1):
            # gammaIntensity = (gammaVal)/gammaSteps
            totalIntensity = (1-minInt)*(np.sqrt((gammaVal+kVal-2)/(gammaSteps+kSteps-2))) + minInt
            if kVal+gammaVal == 2:
                kSkew = 1/2
            else:
                kSkew = (kVal-1/2)/(kVal+gammaVal-1)
            if kSkew < 1/2:
                kSkew = (1-np.power((1/2-kSkew)*2, 2/3))/2
            else:
                kSkew = (1 + np.power(2*(kSkew-1/2), 2/3))/2
            kIntensity = totalIntensity*kSkew
            gammaIntensity = totalIntensity*(1-kSkew)
            minimalIntensity = min(kIntensity,gammaIntensity)
            gammaValMask = listPerGamma[gammaVal - 1]
            curMask = kValMask * gammaValMask
            counteredThisPoint |= curMask
            # kIntensity = (kSteps+1-kVal)
            curColour = (1*(kIntensity-minimalIntensity), (2*kIntensity + 4*minimalIntensity)/3,
                                             1*(gammaIntensity-minimalIntensity))
            combinedColours[(kVal, gammaVal)] = curColour
            fullColours[(kVal, gammaVal)] = curColour
            ax.contourf(thetaMesh, rMesh, curMask.astype(float), levels=[0.5, 1.5], colors=[curColour], alpha=1)
            ax.contour(thetaMesh, rMesh, curMask.astype(float), levels=[0.5, 1.5],
                        colors=[(1 * (kIntensity - minimalIntensity)/2,
                                 (kIntensity + 3*minimalIntensity)/4,
                                 1 * (gammaIntensity - minimalIntensity)/2)], alpha=1, linewidths=0.8)
    # Annotate using theta ∈ [0, π] only, with G₁.a in white
    # for name, (x, y) in region_labels_upper_half.items():
    #     color = 'white'
    #     text = ax.text(x, y, name,
    #                    color=color, fontsize=10, ha='center', va='center')
    #     text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])




    # Fill in (k=0, gamma=0)
    fullColours[(0, 0)] = (0.5 * minInt, 0.05 * minInt, 0.05 * minInt)

    text_obj =ax.set_title(titleString,fontsize=18, fontweight="light")

    text_obj.set_path_effects([
        path_effects.withStroke(linewidth=0.5, foreground='black')
    ])
    # print(bestGammaArray[~counteredThisPoint])
    # print(bestKArray[~counteredThisPoint])
    theta_ticks = [
        0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        5 * np.pi / 6,
        np.pi,
        7 * np.pi / 6,
        5 * np.pi / 4,
        4 * np.pi / 3,
        3 * np.pi / 2,
        5 * np.pi / 3,
        7 * np.pi / 4,
        11 * np.pi / 6
    ]

    theta_labels = [
        r'$0$',
        r'$\frac{1}{6}\pi$',
        r'$\frac{1}{4}\pi$',
        r'$\frac{1}{3}\pi$',
        r'$\frac{1}{2}\pi$',
        r'$\frac{2}{3}\pi$',
        r'$\frac{3}{4}\pi$',
        r'$\frac{5}{6}\pi$',
        r'$\pi$',
        r'$1\frac{1}{6}\pi$',
        r'$1\frac{1}{4}\pi$',
        r'$1\frac{1}{3}\pi$',
        r'$1\frac{1}{2}\pi$',
        r'$1\frac{2}{3}\pi$',
        r'$1\frac{3}{4}\pi$',
        r'$1\frac{5}{6}\pi$'
    ]
    ax.set_thetagrids(np.degrees(theta_ticks), labels=theta_labels)
    # xT = plt.xticks()[0]
    # xL = ['0', r'$\frac{1}{4}\pi$', r'$\frac{1}{2}\pi$', r'$\frac{3}{4}\pi$',
    #            r'$\pi$', r'$1\frac{1}{4}\pi$', r'$1\frac{1}{2} \pi$', r'$1\frac{3}{4} \pi$']
    # plt.xticks(xT, xL)
    ax.set_rlabel_position(-190)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_color('white')
        tick.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='black')
        ])
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(15)  # doubled from ~10
        label.set_color('white')
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground='black')
        ])
    ax.grid(alpha=0.525)
    plt.tight_layout()
    plt.show()
    if legendBool:
        # Plot the legend grid
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')

        # Plot the color grid
        for iK in range(1, kSteps+1):  # rows = gamma
            for jGamma in range(1, gammaSteps+1):  # cols = k
                rect = plt.Rectangle((iK,  jGamma), 1, 1, facecolor=combinedColours[iK, jGamma])
                ax.add_patch(rect)

        # Add gamma-only color bar (left side)
        for jGamma in range(1, gammaSteps+1):
            rect = plt.Rectangle((-1.2, jGamma), 1, 1, facecolor=gammaColours[jGamma])
            ax.add_patch(rect)

        # Add k-only color bar (bottom)
        for iK in range(1, kSteps+1):
            rect = plt.Rectangle((iK, -1.2), 1, 1, facecolor=kColours[iK])
            ax.add_patch(rect)

        # Add labels
        for jGamma in range(1, gammaSteps+1):
            label = rf"$\gamma$={jGamma}" if jGamma < gammaSteps else rf"$\gamma\geq${gammaSteps}"
            ax.text(-1.7,  0.5 + jGamma, label, va='center', ha='right', fontsize=8)

        for iK in range(1, kSteps+1):
            label = f"$k$={iK}" if iK < kSteps else rf"$k\geq${kSteps}"
            ax.text(iK + 0.5, -1.7, label, va='top', ha='center', fontsize=8)

        # Set axis limits to fit everything
        ax.set_xlim(-2.2, kSteps+1)
        ax.set_ylim(-2.2, gammaSteps+1)

        ax.set_title(r"Legend Grid: $\gamma$ vs $k$", fontsize=16)

        plt.show()
    if plot3dBool:
        minVals = np.asarray(minVals)
        colourMap = np.zeros(minVals.shape+(4,))
        for idx in np.ndindex(minVals.shape):
            # curVal = minVals[idx]
            curGamma = min(bestGammaArray[idx],gammaSteps)
            curK = min(bestKArray[idx], kSteps)
            colourMap[idx] = (*fullColours[(curK,curGamma)],1)
        objectiveMesh = (-minVals)/(1-minVals)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cosMesh = np.cos(thetaMesh)
        sinMesh = np.sin(thetaMesh)
        xMesh = cosMesh*rMesh
        yMesh = sinMesh*rMesh
        # valNegLogs = np.log10(-minVals)
        nonNegThetaMaxIdx = thetaMesh.shape[0]//2
        ls = LightSource(105, 50)
        shaded_colors = ls.shade_rgb(colourMap[:nonNegThetaMaxIdx],
                                     np.power(objectiveMesh[:nonNegThetaMaxIdx], 1),
                                     vert_exag=1, blend_mode='overlay')
        surf = ax.plot_surface(xMesh[:nonNegThetaMaxIdx], yMesh[:nonNegThetaMaxIdx],
                               np.power(objectiveMesh,1)[:nonNegThetaMaxIdx],
                               facecolors=shaded_colors, linewidth=1,
                               antialiased=True,
                               rstride=3, cstride=3)



        ax.view_init(elev=45., azim=-95, roll=0)
        plt.show()

        # Z_sphere = np.sqrt(1 - xMesh ** 2 - yMesh ** 2)
        # Z_sphere[np.isnan(Z_sphere)] = 0  # avoid NaNs at edge where 1 - x^2 - y^2 <= 0
        #
        # # Scale the full vector by the function value Z
        # X_mapped = objectiveMesh * xMesh
        # Y_mapped = objectiveMesh * yMesh
        # Z_mapped = objectiveMesh * Z_sphere
        #
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # Plot the surface
        # ax.plot_surface(X_mapped, Y_mapped, Z_mapped,
        #                 facecolors=colourMap,
        #                 linewidth=0, antialiased=False, shade=True)
        # plt.show()
        #
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # cosMesh = np.cos(thetaMesh)
        # sinMesh = np.sin(thetaMesh)
        # newThetaMesh = thetaMesh.copy()
        # newThetaMesh[nonNegThetaMaxIdx:] = thetaMesh[nonNegThetaMaxIdx:] - 2 * np.pi
        # xMesh = rMesh
        # yMesh = rMesh * newThetaMesh
        # # valNegLogs = np.log10(-minVals)
        # # nonNegThetaMaxIdx = thetaMesh.shape[0] // 2 + 1
        # # surf = ax.plot_surface(xMesh[nonNegThetaMaxIdx:], yMesh[nonNegThetaMaxIdx:],
        # #                        np.power((-minVals) / (1 - minVals), 1)[nonNegThetaMaxIdx:],
        # #                        facecolors=colourMap[nonNegThetaMaxIdx:], linewidth=1,
        # #                        antialiased=False, shade=False, rstride=1, cstride=1)
        # # Create grid for the y = pi * x plane
        # x_plane = np.linspace(0, 1, 11, endpoint=True)
        # z_plane = np.linspace(0, 0.5, 11, endpoint=True)
        # X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
        # Y_plane = np.pi * X_plane
        # # Plot y = pi * x plane as grid
        # # for i in range(len(z_plane)):
        # #     ax.plot(X_plane[i], Y_plane[i], Z_plane[i], color='grey', linestyle='--', linewidth=0.7)
        # # for j in range(len(x_plane)):
        # #     ax.plot(X_plane[:, j], Y_plane[:, j], Z_plane[:, j], color='grey', linestyle='--', linewidth=0.7)
        # surf = ax.plot_surface(X_plane,Z_plane, Y_plane,
        #
        #                        edgecolor='royalblue', linewidth=1,
        #                        antialiased=False, shade=False, rstride=1, cstride=1,alpha=0.6)
        # surf = ax.plot_surface(xMesh[:nonNegThetaMaxIdx], yMesh[:nonNegThetaMaxIdx],
        #                        np.power((-minVals) / (1 - minVals), 1)[:nonNegThetaMaxIdx],
        #                        facecolors=colourMap[:nonNegThetaMaxIdx], linewidth=1,
        #                        antialiased=False, shade=False, rstride=1, cstride=1)
        # ax.view_init(elev=-60, azim=-100, roll=0)
        # plt.show()

        # rFlat, thetaFlat= rMesh[:nonNegThetaMaxIdx].flatten(), thetaMesh[:nonNegThetaMaxIdx].flatten()
        # tri = mtri.Triangulation(rFlat, thetaFlat)
        # cosMesh = np.cos(thetaFlat)
        # sinMesh = np.sin(thetaFlat)
        # xMesh = cosMesh * rFlat
        # yMesh = sinMesh * rFlat
        # tri2 = mtri.Triangulation(xMesh, yMesh)
        # zMesh = (np.power((-minVals) / (1 - minVals), 1))[:nonNegThetaMaxIdx].flatten()
        # triangle_colors = colourMap[:nonNegThetaMaxIdx].reshape(xMesh.shape + (4,))[tri.triangles].mean(axis=1)
        # surf = ax.plot_trisurf(tri,
        #                        zMesh,
        #                        # triangles=tri.triangles,
        #                        facecolors=triangle_colors, linewidth=0,
        #                        antialiased=False, shade=False)

        # Style the plot
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_title("3D Function Plot with Subgroup Color Mapping")



        x=1


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


if __name__ == "__main__":
    readFromFile = True
    if readFromFile:
        plotBQPResults("TestResults/PD_trulyUniformOnSphere_C4_r0.02-0.98_t0.00-3.14_30_11_01-16.xlsx",
                       "TestResults/PD_trulyUniformOnSphere_C4_r0.02-0.98_t0.00-3.14_04_12_15-45.xlsx")
    else:
        denseDots = (0,(1,1))
        dotDash = (0,(3,3,1,3))
        dashes = (0,(3,3))
        # thetaVFunc = np.vectorize(OldFunctions.thetaBasedOnThetapart)
        interVFunc = np.vectorize(interpolateDegree)

        if testType == "RadGamma":
            xMin = -1
            xMax = 1
            xSteps = 500
            xStep = (xMax - xMin) / xSteps

            X = np.linspace(xMin, xMax, xSteps)

            vfunc = np.vectorize(OldFunctions.normedJacobiValue)
        else:
            xMin = -1
            xMax = 1
            xSteps = 500
            xStep = (xMax-xMin)/xSteps
            # xStep = 0.00125

            X = np.linspace(xMin, xMax, xSteps)

            vfunc = np.vectorize(OldFunctions.normedJacobiValue)

        if testType == "DiskCheck":
            radArray = np.linspace(0,1,1001,endpoint=True)
            diskToCheck = OrthonormalPolyClasses.Disk(3-2,0,200)
            diskVals = diskToCheck(radArray, 0,200)[0]
            fig, ax = plt.subplots()
            ax.plot(radArray,diskVals)
            plt.show()
            x=1

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
            vfunc = np.vectorize(OldFunctions.normedJacobiValue, otypes=[float])
            if radialBool:

                thetaSpace = np.linspace(0, math.pi, num=definitionPoints)
                innerProductTheta = np.cos(thetaSpace)


                jacobiObj = 1 / dimension * vfunc(degree=0, alpha=(dimension - 3) / 2, beta=(dimension - 3) / 2, location=innerProductTheta)

                # for angle in innerProductTheta:
                #     print(OldFunctions.normedJacobiValue(degree=2,alpha=(dimension-3)/2,beta=(dimension-3)/2,
                #                                              location=angle))
                jacobiComp = (dimension - 1) / dimension * vfunc(location=innerProductTheta, degree=2, alpha=(dimension - 3) / 2,
                                                                 beta=(dimension - 3) / 2
                                                                 )
                jacobiSol = jacobiObj + jacobiComp
                circMeasure = 2 * (math.pi ** ((dimension - 1) / 2) / OldFunctions.gammaFunction((dimension - 1) / 2))
                sphereMeasure = 2 * (math.pi ** ((dimension) / 2) / OldFunctions.gammaFunction((dimension) / 2))
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
                #     print(OldFunctions.normedJacobiValue(degree=2,alpha=(dimension-3)/2,beta=(dimension-3)/2,
                #                                              location=angle))
                jacobiComp = (dimension - 1) / dimension * vfunc(location=r, degree=2, alpha=(dimension - 3) / 2,
                                                                 beta=(dimension - 3) / 2
                                                                 )
                jacobiSol = jacobiObj + jacobiComp
                circMeasure = 2 * (math.pi ** ((dimension - 1) / 2) / OldFunctions.gammaFunction((dimension - 1) / 2))
                sphereMeasure = 2 * (math.pi ** ((dimension) / 2) / OldFunctions.gammaFunction((dimension) / 2))
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



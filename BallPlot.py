import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def get_camera_position(ax, r=10):  # set r manually
    elev = np.deg2rad(ax.elev)
    azim = np.deg2rad(ax.azim)
    x = r * np.cos(elev) * np.cos(azim)
    y = r * np.cos(elev) * np.sin(azim)
    z = r * np.sin(elev)
    return np.array([x, y, z])

def add_dashed_striped_circle(ax, x, y, z, dash=6, color_front='black', color_back='gray', lw=1.5):
    viewer_pos = get_camera_position(ax)
    ax.plot([0, viewer_pos[0]], [0, viewer_pos[1]], [0, viewer_pos[2]], color='purple', linestyle='--', linewidth=1.5)

    points = np.array([x, y, z]).T
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    view_dir = viewer_pos / np.linalg.norm(viewer_pos)
    mids = np.array([(seg[0] + seg[1]) / 2 for seg in segments])
    normals = mids / np.linalg.norm(mids, axis=1)[:, None]
    visibility = np.einsum('ij,j->i', normals, view_dir) > 0
    final_segments, final_colors = [], []
    for i, seg in enumerate(segments):
        if (i // dash) % 2 == 0:
            final_segments.append(seg)
            final_colors.append(color_front if visibility[i] else color_back)
    lc = Line3DCollection(final_segments, colors=final_colors, linewidths=lw)
    ax.add_collection3d(lc)

def plot_circle(ax, normal, radius=0.5, center=np.array([0, 0, 0]), resolution=200, color='green'):
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        v = np.array([1, 0, 0],dtype=np.float64)
    else:
        v = np.cross(normal, [0, 0, 1])
    v /= np.linalg.norm(v)
    w = np.cross(normal, v)
    t = np.linspace(0, 2 * np.pi, resolution)
    circle = center[:, None] + radius*normal[:,None] + radius * (v[:, None] * np.cos(t) + w[:, None] * np.sin(t))
    ax.plot(circle[0], circle[1], circle[2], color=color, linewidth=2)

def drawSetWithBounds(ax, rlow,rhigh,thetalow, thetahigh, philow, phihigh):
    standardColour = (0.7,0.7,1,0.6)
    standardEdgeColours = (0.35,0.35,0.5,1)
    standardLineWidth = 0.2
    standardRcount = 1
    standardCCount = 1
    # Plot set faces using plot_surface (deformed spherical cube)
    theta_vals = np.linspace(thetalow, thetahigh, 50)
    phi_vals = np.linspace(philow, phihigh, 50)
    r_vals = np.linspace(rlow, rhigh, 2)

    # Face 1: r = 0.5 (inner shell)
    THETA, PHI = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    R = np.full_like(THETA, 0.5)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth,rcount=standardRcount,ccount=standardCCount)

    # Face 2: r = 0.75 (outer shell)
    R = np.full_like(THETA, 0.75)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth, rcount=standardRcount, ccount=standardCCount)

    # Face 3: theta = pi/6
    R, PHI = np.meshgrid(r_vals, phi_vals, indexing='ij')
    THETA = np.full_like(R, np.pi / 6)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth, rcount=standardRcount, ccount=standardCCount)

    # Face 4: theta = pi/3
    THETA = np.full_like(R, np.pi / 3)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth, rcount=standardRcount, ccount=standardCCount)

    # Face 5: phi = pi/4
    R, THETA = np.meshgrid(r_vals, theta_vals, indexing='ij')
    PHI = np.full_like(R, np.pi / 4)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth, rcount=standardRcount, ccount=standardCCount)

    # Face 6: phi = pi/2
    PHI = np.full_like(R, np.pi / 2)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA) * np.cos(PHI)
    Z = R * np.sin(THETA) * np.sin(PHI)
    ax.plot_surface(X, Y, Z, color=standardColour, edgecolor=standardEdgeColours,
                    linewidth=standardLineWidth, rcount=standardRcount, ccount=standardCCount)

def plotBall():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.1, linewidth=0)

    # # Spherical shell
    # r = np.linspace(0.5, 0.75, 100)
    # theta_shell = np.linspace(np.pi / 6, np.pi / 3, 200)
    # phi_shell = np.linspace(np.pi / 4, np.pi / 2, 200)
    # R, T, P = np.meshgrid(r, theta_shell, phi_shell, indexing='ij')
    # X = R * np.cos(T)
    # Y = R * np.sin(T) * np.cos(P)
    # Z = R * np.sin(T) * np.sin(P)
    # ax.plot_surface(X, Y, Z, color='blue', alpha=0.6)
    drawSetWithBounds(ax, 1/2, 3/4, np.pi/6, np.pi/3, np.pi/5, np.pi/2)


    # Red line from origin
    point = [0.0, 0.0, 1.0]
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='red', linewidth=2)

    # Diagonal circle
    plot_circle(ax, np.array([0.0, 0.0, 1.0],dtype=np.float64), radius=0.5)

    # Dotted border circles
    theta = np.linspace(0, 2 * np.pi, 300)
    add_dashed_striped_circle(ax, np.cos(theta), np.sin(theta), np.zeros_like(theta))  # XY-plane
    add_dashed_striped_circle(ax, np.zeros_like(theta), np.cos(theta), np.sin(theta))  # YZ-plane
    add_dashed_striped_circle(ax, np.cos(theta), np.zeros_like(theta), np.sin(theta))  # XZ-plane

    # Layout
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()





def draw_box(ax, alpha=0.5):
    center = np.array([0.5, np.pi, np.pi])
    camera_pos = np.array([0, 0, 3])
    view_dir = camera_pos - center
    view_dir /= np.linalg.norm(view_dir)

    corners = np.array([[r, t1, t2] for r in [0,1] for t1 in [0,2*np.pi] for t2 in [0,2*np.pi]])
    edges = [
        (corners[i], corners[j])
        for i in range(8) for j in range(i+1,8)
        if np.sum(corners[i]!=corners[j])==1
    ]

    for p1, p2 in edges:
        mid = 0.5*(p1+p2)
        normal = mid - center
        normal /= np.linalg.norm(normal)
        color = 'black' if np.dot(normal,view_dir)>0 else 'gray'
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], linestyle=':', color=color, alpha=alpha)

    ticks_pi = np.arange(0.5*np.pi,2.5*np.pi,np.pi/2)
    ticklabels_pi = [f"{round(t/np.pi,1)}π" for t in ticks_pi]
    ax.set_xticks(np.arange(0,1.1,0.25), labels=["","","","",""])
    ax.xaxis.line.set_color((0, 0, 0, 0))

    # ax.set_xlabel(r"$r$")
    ax.set_yticks(ticks_pi, labels=["","","",""])
    # ax.set_yticklabels(ticklabels_pi)
    # ax.set_ylabel(r"$\theta_1$")
    ax.set_zticks(ticks_pi, labels=["","","",""])
    # ax.set_zticklabels(ticklabels_pi)
    # ax.set_zlabel(r"$\theta_2$", labelpad=10, rotation=90)
    ax.set_xlabel("")  # clear the default label

    ax.text(
        1.1, 0.2,  0.2,  # x, y, z position (adjust as needed)
        r"$r$", fontsize=14,
        ha='center', va='center'
    )
    ax.set_ylabel("")  # clear the default label

    ax.text(
        0, (2 + 0.2) * np.pi, 0,  # x, y, z position (adjust as needed)
        r"$\theta_2$", fontsize=14,
        ha='center', va='center'
    )
    ax.set_zlabel("")  # clear the default label

    ax.text(
        0, 0, (2 + 0.2) * np.pi,  # x, y, z position (adjust as needed)
        r"$\theta_1$", fontsize=14,
        ha='center', va='center'
    )
    ax.set_xlim(1,0)
    ax.set_ylim(0,2*np.pi)
    ax.set_zlim(0,2*np.pi)
    ax.set_box_aspect([2*np.pi,2*np.pi,2*np.pi])
    ax.tick_params(axis='x', which='both', color=[0,0,0,0])
    ax.tick_params(axis='y', which='both', color=[0,0,0,0])
    ax.tick_params(axis='z', which='both', color=[0,0,0,0])
    ax.view_init(elev=15., azim=15, roll=0)
    draw_custom_r_ticks(ax)
    draw_custom_theta1_ticks(ax)
    draw_custom_theta2_ticks(ax)


def draw_set(ax, r_bounds, t1_bounds, t2_bounds, shadowBool=True, ccount=1,
        rcount=1, **kwargs):
    r1, r2 = r_bounds
    t1min, t1max = t1_bounds
    t2min, t2max = t2_bounds

    r = np.linspace(r1, r2, 2)
    t1 = np.linspace(t1min, t1max, 50)
    t2 = np.linspace(t2min, t2max, 50)

    T1_grid, T2_grid = np.meshgrid(t1, t2, indexing='ij')
    ax.plot_surface(
        np.full_like(T1_grid, r1), T1_grid, T2_grid, ccount=1,
        rcount=1, **kwargs
    )
    ax.plot_surface(
        np.full_like(T1_grid, r2), T1_grid, T2_grid, ccount=1,
        rcount=1, **kwargs
    )

    # Faces at theta1 = t1min and theta1 = t1max
    R_grid, T2_grid = np.meshgrid(r, t2, indexing='ij')
    ax.plot_surface(
        R_grid, np.full_like(R_grid, t1min), T2_grid, ccount=1,
        rcount=1, **kwargs
    )
    ax.plot_surface(
        R_grid, np.full_like(R_grid, t1max), T2_grid, ccount=1,
        rcount=1, **kwargs
    )

    # Faces at theta2 = t2min and theta2 = t2max
    R_grid, T1_grid = np.meshgrid(r, t1, indexing='ij')
    ax.plot_surface(
        R_grid, T1_grid, np.full_like(R_grid, t2min), ccount=1,
        rcount=1, **kwargs
    )
    ax.plot_surface(
        R_grid, T1_grid, np.full_like(R_grid, t2max), ccount=1,
        rcount=1, **kwargs
    )
    if shadowBool:
        alpha = kwargs.get('alpha', 1)
        shadow_alpha = alpha / 2
        shadow_kwargs = kwargs.copy()
        shadow_kwargs['alpha'] = shadow_alpha

        # ax.plot_surface(np.full_like(T1_grid, r1), T1_grid, np.full_like(T1_grid, 0), ccount=ccount, rcount=rcount,
        #                 **shadow_kwargs)
        # ax.plot_surface(np.full_like(T1_grid, r2), T1_grid, np.full_like(T1_grid, 0), ccount=ccount, rcount=rcount,
        #                 **shadow_kwargs)
        #
        # ax.plot_surface(R_grid, np.full_like(R_grid, t1min), np.full_like(R_grid, 0), ccount=ccount, rcount=rcount,
        #                 **shadow_kwargs)
        # ax.plot_surface(R_grid, np.full_like(R_grid, t1max), np.full_like(R_grid, 0), ccount=ccount, rcount=rcount,
        #                 **shadow_kwargs)

        ax.plot_surface(R_grid, T1_grid, np.full_like(R_grid, 0), ccount=ccount, rcount=rcount, **shadow_kwargs)

        # Dashed vertical lines
        corners = [
            [r1, t1min, t2min], [r1, t1min, t2max],
            [r1, t1max, t2min], [r1, t1max, t2max],
            [r2, t1min, t2min], [r2, t1min, t2max],
            [r2, t1max, t2min], [r2, t1max, t2max],
        ]

        for corner in corners:
            ax.plot(
                [corner[0], corner[0]],
                [corner[1], corner[1]],
                [corner[2], 0],
                linestyle=(0, (5, 10)), color='black', linewidth=0.7
            )


def draw_line(ax, p1, p2, shadowBool=False, guidesBool=False, **kwargs):
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


def draw_custom_r_ticks(ax, ticks=np.linspace(0,1,5),tickwidth=1):
    for tick in ticks:
        if tick == 0:
            ax.text(tick, 0, -0.2, f"{int(tick)}", color='black', ha='right', va='top')
        elif tick == 1:
            ax.text(tick, 0.05, -0.1, f"{int(tick)}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.2], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [0, 0], [-0.2, 0.2], color='black', linewidth=tickwidth)
        else:
            ax.text(tick, 0.05, -0.1, f"{tick:.2f}", color='black', ha='left', va='top')
            # ax.plot([tick, tick], [0, 0.1], [0, 0], color='black', linewidth=tickwidth)
            ax.plot([tick, tick], [0, 0], [-0.1, 0.1], color='black', linewidth=tickwidth)

    draw_line(ax,[0,0,0],[1,0,0],color="black", linewidth=0.7)



def draw_custom_theta1_ticks(ax, ticks=None, tickwidth=1, ticklength=0.1):
    if ticks is None:
        ticks = np.arange(0.5*np.pi, 2.5*np.pi, np.pi/2)

    ticklabels = [f"{round(t/np.pi,1)} π" for t in ticks]
    for idx in range(len(ticks)):
        tR = round(ticks[idx]/np.pi,1)
        if tR == 1:
            ticklabels[idx] = f"π"
        elif tR % 1 == 0:
            ticklabels[idx] = f"{int(tR)} π"
    for tick, label in zip(ticks, ticklabels):
        ax.text(0, tick, -0.45, label, color='black', ha='center', va='center')
        # Draw tick mark on theta1 axis edge
        ax.plot([0, 0],  [tick, tick], [-ticklength, ticklength], color='black', linewidth=tickwidth)

    draw_line(ax, [0,0,0], [0,2*np.pi,0], color="black", linewidth=0.7)


def draw_custom_theta2_ticks(ax, ticks=None, tickwidth=1, ticklength=0.1):
    if ticks is None:
        ticks = np.arange(0.5*np.pi, 2.5*np.pi, np.pi/2)

    ticklabels = [f"{round(t/np.pi,1)} π" for t in ticks]
    for idx in range(len(ticks)):
        tR = round(ticks[idx]/np.pi,1)
        if tR == 1:
            ticklabels[idx] = f"π"
        elif tR % 1 == 0:
            ticklabels[idx] = f"{int(tR)} π"
    ticklength /= np.pi
    for tick, label in zip(ticks, ticklabels):
        ax.text(-0.1, -0.1, tick, label, color='black', ha='right', va='center')
        # Draw tick mark on theta2 axis edge
        ax.plot([-ticklength, ticklength], [0, 0], [tick, tick], color='black', linewidth=tickwidth)

    draw_line(ax, [0,0,0], [0,0,2*np.pi], color="black", linewidth=0.7)


def draw_hull_shadow_and_projections(ax, points, hull, **kwargs):
    """
    Draws shadow of the convex hull on z=0 where the hull crosses the ground,
    and draws dashed lines from vertices to z=0.

    Only faces whose normal has both positive and negative z-components contribute to the shadow.
    """

    points = np.asarray(points)
    simplices = hull.simplices
    eqs = hull.equations

    shadow_kwargs = kwargs.copy()
    alpha = kwargs.get('alpha', 1.0)
    shadow_alpha = alpha / 2
    # shadow_color = kwargs.get('color', 'gray')
    shadow_kwargs["alpha"] = shadow_alpha

    faces_for_shadow = []

    # Go through all faces
    pointNrs = [pIdx for pIdx in range(points.shape[0])]
    hasNegArray = np.zeros(points.shape[0],dtype=bool)
    hasNonNegArray = np.zeros(points.shape[0],dtype=bool)
    for face_nr, simplex in enumerate(simplices):
        normal = eqs[face_nr, :3]  # plane normal
        if normal[2] >= 0:
            hasNonNegArray[simplex] = True
        else:
            hasNegArray[simplex] = True
    relevantPoints = hasNonNegArray & hasNegArray
    relevantCoords = points[relevantPoints].copy()
    relevantCoords[:,2] = 0
    points2d = relevantCoords[:, :2]

    # Compute 2D convex hull
    hull2d = ConvexHull(points2d)
    shadowSimplices = []
    baseVertex = hull2d.simplices[0][0]
    for simplex in hull2d.simplices:
        if baseVertex not in simplex:
            shadowSimplices.append(np.append(simplex,baseVertex))
        p1 = points2d[simplex[0]]
        p2 = points2d[simplex[1]]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [0, 0],
            color=(0,0,0,1)
        )
    poly3d = Poly3DCollection(relevantCoords[shadowSimplices], **shadow_kwargs)
    ax.add_collection3d(poly3d)

    # if faces_for_shadow:
    #     shadow_poly = Poly3DCollection(
    #         faces_for_shadow,
    #         color=shadow_color,
    #         alpha=shadow_alpha,
    #         linewidths=0,
    #     )
    #     ax.add_collection3d(shadow_poly)

    # Draw dashed lines from each vertex of hull down to z=0
    for p in points[relevantPoints][hull2d.vertices]:
        ax.plot(
            [p[0], p[0]], [p[1], p[1]], [p[2], 0],
            linestyle='--', color='black', linewidth=0.7, alpha=0.7
        )

def draw_convex_hull(ax, points,shadowBool=True, **kwargs):
    """
    Draws the convex hull of a 3D point cloud on a given matplotlib 3D axis.

    Parameters:
        ax: matplotlib 3D axes
        points: (N, 3) array of points
        kwargs: additional styling passed to Poly3DCollection, e.g. alpha, color
    """
    cameraNormal = get_camera_position(ax,20)+np.array([-0.5,np.pi,np.pi])
    cameraEq = np.ones(4)
    cameraEq[:3] = cameraNormal
    cameraEq[0] *= -1
    points = np.asarray(points)
    hull = ConvexHull(points)

    # Collect all triangle faces
    simplices = [simplex for simplex in hull.simplices]
    faces = [points[simplex] for simplex in hull.simplices]

    # Default alpha if not specified
    alpha = kwargs.get('alpha', 0.4)
    kwargs['alpha'] = alpha
    kwargs['linewidth'] = 0

    # Create a collection of triangles
    poly3d = Poly3DCollection(faces, **kwargs)
    ax.add_collection3d(poly3d)

    # Optionally draw vertices too
    hullPoints = hull.points[hull.vertices]
    ax.scatter(hullPoints[:, 0], hullPoints[:, 1], hullPoints[:, 2], color='k', s=10)

    # for i in range(points.shape[0]):
    #     x, y, z = points[i]
    #     ax.text(
    #         x, y, z, str(i)
    #     )
    eqs = hull.equations
    # Find lines on outside of faces
    for faceNr in range(len(faces)):
        curFace = simplices[faceNr]
        curEq = eqs[faceNr]
        faceOut = (np.vdot(curEq,cameraEq) >= 0)
        curNeighbours = hull.neighbors[faceNr]
        for compFaceNr in curNeighbours:
            if faceNr < compFaceNr:
                compEq = eqs[compFaceNr]
                if not np.allclose(curEq, compEq, atol=1e-6):
                    compFace = simplices[compFaceNr]
                    lineBetweenNrs = np.intersect1d(curFace, compFace)
                    if lineBetweenNrs.shape[0] == 2:
                        lineBetween = points[lineBetweenNrs]
                        p1 = lineBetween[0]
                        p2 = lineBetween[1]
                        compFaceOut = (np.vdot(compEq, cameraEq) >= 0)
                        if compFaceOut or faceOut:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],color=(0,0,0,1))
                        else:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=(0, 0, 0, .4))
    if shadowBool:
        draw_hull_shadow_and_projections(ax, points, hull, **kwargs)
    return hull


def create_dodecahedron_points(x_range, y_range, z_range):
    """
    Generate points of a regular dodecahedron, scaled and translated to fit in the given ranges.

    Parameters:
        x_range: (xmin, xmax) in r
        y_range: (ymin, ymax) in theta1
        z_range: (zmin, zmax) in theta2

    Returns:
        points: (20,3) array of points
    """
    # Regular dodecahedron centered at (0,0,0) with unit circumradius
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    pts = []
    # vertices from (±1, ±1, ±1)
    for i in [-1,1]:
        for j in [-1,1]:
            for k in [-1,1]:
                pts.append([i,j,k])

    # (0, ±1/phi, ±phi) and cyclic perms
    for i in [-1,1]:
        for j in [-1,1]:
            pts.append([0, i/phi, j*phi])
            pts.append([i/phi, j*phi, 0])
            pts.append([i*phi, 0, j/phi])

    pts = np.array(pts)

    # scale & translate to fit into given ranges
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    # target mins and maxs
    tgt_mins = np.array([x_range[0], y_range[0], z_range[0]])
    tgt_maxs = np.array([x_range[1], y_range[1], z_range[1]])

    # scale each axis
    scale = (tgt_maxs - tgt_mins) / (maxs - mins)
    points = (pts - mins) * scale + tgt_mins

    return points


# Define target ranges




if __name__ == "__main__":
    # plotBall()
    # Run the plot
    bodiesBool = True
    linesBool = True
    altLineBool = True
    refLineBool = False
    completeBool = False
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    kwargsIn = {"marker": "_", "markersize": 12.0, "markeredgewidth": 2.0}

    draw_box(ax)


    theta2Start = 1 * np.pi
    shiftAlongLine1 = 1/4 * np.pi
    shiftAlongLine2 = 1 * np.pi
    shiftAlongLine3 = 2 * np.pi
    rStart = 1/4
    rStop = 3/4
    rMiddle = (rStart+rStop)/2
    rDeltaF = np.array([rStart - rMiddle, 0, 0])
    rDeltaB = np.array([rStop - rMiddle, 0, 0])

    height1 = 1/2*np.pi
    height2 = 1/4*np.pi
    height3 = -1/4*np.pi
    heightList = [height1, height2, height3]
    adjustHeight = [0, -1, 1]
    heightMatrix = np.array([[hd*ad for ad in adjustHeight] for hd in heightList])

    sideDelta = np.array([0,1/2*np.pi,0])

    deltaFunc = np.array([0,1/2*np.pi,1/8*np.pi])

    fig1Matrix = np.zeros((8,3))
    fig1Matrix[0] = np.array([rMiddle,theta2Start-shiftAlongLine1, shiftAlongLine1])
    fig1Matrix[1] = fig1Matrix[0] + heightMatrix[0]
    fig1Matrix[2] = fig1Matrix[0] + rDeltaF
    fig1Matrix[3] = fig1Matrix[0] + rDeltaB
    fig1Matrix[[4,6]] = fig1Matrix[[2,3]] - sideDelta
    fig1Matrix[[5,7]] = fig1Matrix[[2, 3]] + sideDelta


    fig2Matrix = np.zeros((14,3))
    fig2Matrix[0] = np.array([rMiddle, theta2Start - shiftAlongLine2, shiftAlongLine2])
    fig2Matrix[1] = fig2Matrix[0] + heightMatrix[1]
    fig2Matrix[2:4] = fig2Matrix[0:2] - sideDelta
    fig2Matrix[4:6] = fig2Matrix[0:2] + sideDelta
    fig2Matrix[6:10] = fig2Matrix[2:6] + rDeltaF
    fig2Matrix[10:] = fig2Matrix[2:6] + rDeltaB

    posMask = (fig2Matrix[:,1]>=0)
    posFig2 = fig2Matrix.copy()
    posFig2[~posMask,1] = 0
    negFig2 = fig2Matrix.copy()
    negFig2[posMask,1] = 0
    negFig2[:, 1] += 2 * np.pi


    fig3Matrix = np.zeros((14, 3))
    fig3Matrix[0] = np.array([rMiddle, theta2Start - shiftAlongLine3, shiftAlongLine3])
    fig3Matrix[1] = fig3Matrix[0] - sideDelta
    fig3Matrix[2] = fig3Matrix[0] + sideDelta
    fig3Matrix[3:5] = fig3Matrix[1:3] + rDeltaF
    fig3Matrix[5:7] = fig3Matrix[1:3] + rDeltaB
    fig3Matrix[7] = fig3Matrix[0] + heightMatrix[2]
    fig3Matrix[8] = fig3Matrix[7] - deltaFunc
    fig3Matrix[9] = fig3Matrix[7] + deltaFunc
    fig3Matrix[10:12] = fig3Matrix[8:10] + rDeltaF
    fig3Matrix[12:] = fig3Matrix[8:10] + rDeltaB
    fig3Matrix[:,1] += 2 * np.pi




    centerBottom1 = np.array([rMiddle,theta2Start-shiftAlongLine1, shiftAlongLine1])
    centerTop1 = centerBottom1 + heightMatrix[0]
    frontBottom1 = centerBottom1 + rDeltaF
    frontTop1 = centerTop1 + rDeltaF
    backBottom1 = centerBottom1 + rDeltaB
    backTop1 = centerTop1 + rDeltaB
    frontLeft1 = frontBottom1 - sideDelta
    frontRight1 = frontBottom1 + sideDelta
    backLeft1 = backBottom1 - sideDelta
    backRight1 = backBottom1 + sideDelta

    centerBottom2 = np.array([rMiddle, theta2Start - shiftAlongLine2, shiftAlongLine2])
    centerTop2 = centerBottom2 + heightMatrix[1]
    clb2 = centerBottom2 - sideDelta
    crb2 = centerBottom2 + sideDelta
    clt2 = centerTop2 - sideDelta
    crt2 = centerTop2 + sideDelta
    if bodiesBool:
        draw_convex_hull(ax, fig1Matrix, color='cyan', shadowBool=False, edgecolor='k', alpha=0.5)
        draw_convex_hull(ax, negFig2, color='cyan', shadowBool=False, edgecolor='k', alpha=0.5)
        draw_convex_hull(ax, posFig2, color='cyan', shadowBool=False, edgecolor='k', alpha=0.5)
        draw_convex_hull(ax, fig3Matrix, color='cyan', shadowBool=False, edgecolor='k', alpha=0.5)
        # x_range = (0.2, 0.7)
        # y_range = (0.2 * np.pi, 1.2 * np.pi)
        # z_range = (0.8 * np.pi, 1.8 * np.pi)

        # dodecahedron_points = create_dodecahedron_points(x_range, y_range, z_range)

        # draw_convex_hull(ax, dodecahedron_points, color='cyan', edgecolor='k', alpha=0.3)

        # points = np.random.beta(4,4,(12, 3))
        # points[:,1:] *= 2* np.pi
        # draw_convex_hull(ax, points, color='cyan', edgecolor='k', alpha=0.3)
        # draw_set(
        #     ax,
        #     r_bounds=(0.2, 0.7),
        #     t1_bounds=(np.pi / 2, np.pi),
        #     t2_bounds=(np.pi, 3 * np.pi / 2),
        #     color='cyan',
        #     alpha=0.4,
        #     edgecolor='k'
        # )

        # draw_line(
        #     ax,
        #     [0.1, np.pi / 4, np.pi / 2],
        #     [0.9, 3 * np.pi / 2, 2 * np.pi],
        #     color='magenta',
        #     shadowBool=True,
        #     guidesBool=False,
        #     linestyle='--',
        #     linewidth=2
        # )
    if linesBool:
        colorIn1 = (1, 0, 1, 1)
        colorOut1 = (1, 0, 1, 1)
        lineIn1 = "-"
        lineOut1 = ":"


        draw_line(
            ax,
            [rMiddle, theta2Start, 0],
            [rMiddle, theta2Start-shiftAlongLine1, shiftAlongLine1],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )
        draw_line(
            ax,
            [rMiddle, theta2Start-shiftAlongLine1, shiftAlongLine1],
            [rMiddle, theta2Start - shiftAlongLine1 - height1, height1+shiftAlongLine1],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )
        draw_line(
            ax,
            [rMiddle, theta2Start - shiftAlongLine1 - height1, height1+shiftAlongLine1],
            [rMiddle,  0, theta2Start],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )

        draw_line(
            ax,
            [rMiddle,  2 * np.pi, theta2Start],
            [rMiddle,  2 * np.pi - height2, theta2Start + height2],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )

        draw_line(
            ax,
            [rMiddle,  2 * np.pi - height2, theta2Start + height2],
            [rMiddle,  1.5 * np.pi - height2, theta2Start + height2 + 1/2*np.pi],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )

        draw_line(
            ax,
            [rMiddle,  1.5 * np.pi - height2, theta2Start + height2 + 1/2*np.pi],
            [rMiddle, theta2Start, 2 * np.pi],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )
        if altLineBool:
            rSecond = 2/4*rMiddle+2/4*rStart
            t2Sec = 3/4 * np.pi
            colorIn2 = (0,0,0,1)
            colorOut2 = (0,0,0,1)
            lineIn2 = "-"
            lineOut2 = ":"

            draw_line(
                ax,
                [rSecond, t2Sec, 0],
                [rSecond, t2Sec - shiftAlongLine1, shiftAlongLine1],
                color=colorOut2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineOut2,
                linewidth=2
            )
            draw_line(
                ax,
                [rSecond, t2Sec - shiftAlongLine1, shiftAlongLine1],
                [rSecond, t2Sec - shiftAlongLine1 - height1/2, shiftAlongLine1+height1/2],
                color=colorIn2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineIn2,
                linewidth=2,
            **kwargsIn
            )

            draw_line(
                ax,
                [rSecond, t2Sec - shiftAlongLine1 - height1/2, shiftAlongLine1+height1/2],
                [rSecond, 0, t2Sec],
                color=colorOut2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineOut2,
                linewidth=2
            )

            draw_line(
                ax,
                [rSecond, 2 * np.pi, t2Sec],
                [rSecond, 1 * np.pi + t2Sec, np.pi],
                color=colorOut2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineOut2,
                linewidth=2
            )

            draw_line(
                ax,
                [rSecond, 1 * np.pi + t2Sec, np.pi],
                [rSecond, 1 * np.pi + t2Sec - height2, np.pi + height2],
                color=colorIn2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineIn2,
                linewidth=2,
            **kwargsIn
            )

            draw_line(
                ax,
                [rSecond, 1 * np.pi + t2Sec - height2, np.pi + height2],
                [rSecond,  t2Sec + 5/16 *np.pi, (2-5/16)*np.pi],
                color=colorOut2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineOut2,
                linewidth=2
            )

            draw_line(
                ax,
                [rSecond,  t2Sec + 5/16 *np.pi, (2-5/16)*np.pi],
                [rSecond, t2Sec, 2 * np.pi],
                color=colorIn2,
                shadowBool=True,
                guidesBool=False,
                linestyle=lineIn2,
                linewidth=2,
            **kwargsIn
            )
    if refLineBool:
        colorIn1 = (1, 0, 1, 1)
        colorOut1 = (1, 0, 1, 1)
        lineIn1 = "-"
        lineOut1 = ":"


        draw_line(
            ax,
            [0, theta2Start, 0],
            [0, theta2Start - shiftAlongLine1, shiftAlongLine1],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )
        draw_line(
            ax,
            [0, theta2Start - shiftAlongLine1, shiftAlongLine1],
            [0, theta2Start - shiftAlongLine1 - height1, height1 + shiftAlongLine1],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )
        draw_line(
            ax,
            [0, theta2Start - shiftAlongLine1 - height1, height1 + shiftAlongLine1],
            [0, 0, theta2Start],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )

        draw_line(
            ax,
            [0, 2 * np.pi, theta2Start],
            [0, 2 * np.pi - height2, theta2Start + height2],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )

        draw_line(
            ax,
            [0, 2 * np.pi - height2, theta2Start + height2],
            [0, 1.5 * np.pi - height2, theta2Start + height2 + 1 / 2 * np.pi],
            color=colorOut1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineOut1,
            linewidth=2
        )

        draw_line(
            ax,
            [0, 1.5 * np.pi - height2, theta2Start + height2 + 1 / 2 * np.pi],
            [0, theta2Start, 2 * np.pi],
            color=colorIn1,
            shadowBool=True,
            guidesBool=False,
            linestyle=lineIn1,
            linewidth=2,
            **kwargsIn
        )
    if completeBool:
        t1Bounds = [[1/4*np.pi,3/4*np.pi],[np.pi,5/4*np.pi],[7/4*np.pi,2*np.pi]]
        for t1Bound in t1Bounds:
            draw_set(ax,[0,1], [0,2*np.pi],t1Bound, shadowBool=False,
                        color='cyan',
                        alpha=0.4,
                        edgecolor='k')


    plt.tight_layout()
    plt.show()



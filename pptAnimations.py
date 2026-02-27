import numpy as np
from manim import *


xL = -7 - 1/9
xR = 7 + 1/9
xD = 14 + 2/9
yT = 4
yB = -4
yD = -8
coordinateDict = {
    "C":np.array([0,0,0]),
    "UL":np.array([xL,yT,0]),
    "UC":np.array([0,yT,0]),
    "UR":np.array([xR,yT,0]),
    "CL":np.array([xL,0,0]),
    "CR":np.array([xR,0,0]),
    "BL":np.array([xL,yB,0]),
    "BC":np.array([0,yB,0]),
    "BR":np.array([xR,yB,0])
}


class SineCurveUnitCircle(Scene):
    # contributed by heejin_park, https://infograph.tistory.com/230
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()


    def show_axis(self):
        x_start = np.array([0,0,0])
        x_end = np.array([xR,0,0])

        y_start = np.array([0,-2,0])
        y_end = np.array([0,2,0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([xL/2,0,0])
        self.curve_start = np.array([0,0,0])


    def add_x_labels(self):
        x_labels = [
            MathTex(r"0.5 \pi"), MathTex(r"1.0 \pi"),
            MathTex(r"1.5 \pi"), MathTex(r"2.0 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([0 + 2*i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=xL/3)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle


    def move_dot_and_draw_curve(self):
        orbit = self.circle
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25


        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            # print(self.t_offset)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color=BLUE)

        def get_line_to_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x,y,0]), color=YELLOW_A, stroke_width=2 )


        self.curve = VGroup()
        self.curve.add(Line(self.curve_start,self.curve_start))


        def get_curve():
            last_line = self.curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(),np.array([x,y,0]), color=YELLOW_D)
            self.curve.add(new_line)

            return self.curve


        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8.5)

        dot.remove_updater(go_around_circle)



class DoubleCapGraph(Scene):
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()


    def show_axis(self):
        x_start = np.array([0,0,0])
        x_end = np.array([xR*5/6,0,0])

        y_start = np.array([0,-2,0])
        y_end = np.array([0,2,0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([xL/2,0,0])
        self.curve_start = np.array([0,0,0])


    def add_x_labels(self):
        x_labels = [
            MathTex(r"0.5 \pi"), MathTex(r"1.0 \pi"),
            MathTex(r"1.5 \pi"), MathTex(r"2.0 \pi"),
        ]

        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([0 + xR/6*(i+1), 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=xL/3)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle


    def move_dot_and_draw_curve(self):
        steps = 1001
        thetaArray = np.linspace(0, 2 * np.pi, 1001, endpoint=True)
        zerosArray = np.zeros_like(thetaArray)
        cosArray = np.cos(thetaArray)
        sinArray = np.sin(thetaArray)
        setBool = (((thetaArray + np.pi / 4) % np.pi) < np.pi / 2)
        charArray = np.array(setBool, dtype=np.float64)
        charPos = 2 * charArray
        thetaPos = thetaArray*(1/(2*np.pi)*xR*(4/5))


        orbit = self.circle
        origin_point = self.origin_point
        cosPos = self.origin_point[0]+cosArray*xL/3
        sinPos = self.origin_point[1]+sinArray*xL/3

        points = np.array([thetaPos,charPos,zerosArray]).T
        print(points.shape)
        curve = VMobject(stroke_color=RED).set_points_as_corners(points)

        dot = Dot(color=RED,radius=0.5)

        self.add(curve)
        self.play(Create(curve), rate_func=linear)
        # self.play(MoveAlongPath(dot, curve),run_time=10, rate_func=linear)
        self.wait(5)


# class CircleSetAndCharacteristic(Scene):
#     def chi(self, theta: float) -> int:
#         """
#         Characteristic function example: 'double cap' style.
#         In-set iff ((theta + pi/4) mod pi) < pi/2.
#         """
#         t = (theta + np.pi/4) % np.pi
#         return int(t < (np.pi/2))
#
#     def make_set_markers(self, circle: Circle, n: int = 240,
#                          in_color=YELLOW, out_color=GREY_B) -> VGroup:
#         """
#         Draw lots of small markers along the circle, colored by chi(theta).
#         """
#         markers = VGroup()
#         for k in range(n):
#             theta = TAU * k / n
#             p = circle.point_at_angle(theta)
#             dot = Dot(p, radius=0.018)
#             dot.set_color(in_color if self.chi(theta) else out_color)
#             markers.add(dot)
#         return markers
#
#     def construct(self):
#         # --- Layout anchors ---
#         left_anchor = LEFT * 3.6
#         right_anchor = RIGHT * 3.6
#
#         # --- Left: circle + set highlighting + moving point ---
#         circle = Circle(radius=1.6).move_to(left_anchor)
#         circle.set_stroke(WHITE, 2)
#
#         set_markers = self.make_set_markers(circle)
#
#         theta = ValueTracker(0.0)
#
#         moving_dot = always_redraw(
#             lambda: Dot(
#                 circle.point_at_angle(theta.get_value()),
#                 radius=0.06,
#                 color=RED
#             )
#         )
#
#         theta_label = always_redraw(
#             lambda: MathTex(r"\theta = {:.2f}".format(theta.get_value()))
#                 .scale(0.7)
#                 .next_to(circle, DOWN)
#         )
#
#         # --- Right: axes for characteristic function ---
#         axes = Axes(
#             x_range=[0, TAU, np.pi/2],
#             y_range=[-0.2, 1.2, 1],
#             x_length=6.0,
#             y_length=2.8,
#             tips=False,
#         ).move_to(right_anchor)
#
#         x_label = MathTex(r"\theta").scale(0.8).next_to(axes.x_axis, DOWN)
#         y_label = MathTex(r"\chi(\theta)").scale(0.8).next_to(axes.y_axis, LEFT)
#
#         # Horizontal guide lines at y=0 and y=1
#         y0_line = axes.get_horizontal_line(axes.c2p(0, 0)).set_stroke(GREY_C, 1)
#         y1_line = axes.get_horizontal_line(axes.c2p(0, 1)).set_stroke(GREY_C, 1)
#
#         # A dot that tracks the current (theta, chi(theta)) on the graph
#         graph_dot = always_redraw(
#             lambda: Dot(
#                 axes.c2p(theta.get_value(), self.chi(theta.get_value())),
#                 radius=0.05,
#                 color=RED
#             )
#         )
#
#         # --- Live-drawn step function ---
#         step_path = VMobject()
#         step_path.set_stroke(BLUE, 4)
#
#         # Initialize path at theta=0
#         t0 = 0.0
#         v0 = self.chi(t0)
#         step_path.set_points_as_corners([axes.c2p(t0, v0), axes.c2p(t0, v0)])
#
#         # State used by the updater
#         last_t = {"t": t0}
#         last_v = {"v": v0}
#
#         def update_step_path(mob: VMobject, dt: float):
#             t = theta.get_value()
#             v = self.chi(t)
#
#             # Only add points if theta advanced (avoid duplicates)
#             if t <= last_t["t"] + 1e-6:
#                 return
#
#             # If membership changed, draw vertical jump at current x
#             if v != last_v["v"]:
#                 mob.add_points_as_corners([
#                     axes.c2p(last_t["t"], last_v["v"]),
#                     axes.c2p(t, last_v["v"]),
#                     axes.c2p(t, v),
#                 ])
#             else:
#                 # Continue horizontal segment at current level
#                 mob.add_points_as_corners([
#                     axes.c2p(last_t["t"], last_v["v"]),
#                     axes.c2p(t, v),
#                 ])
#
#             last_t["t"] = t
#             last_v["v"] = v
#
#         step_path.add_updater(update_step_path)
#
#         # --- Add everything ---
#         self.add(circle, set_markers, moving_dot, theta_label)
#         self.add(axes, x_label, y_label, y0_line, y1_line, graph_dot, step_path)
#
#         # --- Animate theta from 0 to 2pi at linear speed ---
#         self.play(theta.animate.set_value(TAU), run_time=8, rate_func=linear)
#         self.wait()
#
#         # Stop updating after draw is complete (optional)
#         step_path.remove_updater(update_step_path)
#         self.wait(0.5)


class CircleSetAndCharacteristic(Scene):

    def chi(self, theta: float) -> int:
        # Example set: in-set iff ((theta + pi/4) mod pi) < pi/2
        t = (theta + np.pi / 4) % np.pi
        return int(t < (np.pi / 2))

    def make_set_markers(
        self,
        circle: Circle,
        n: int = 240,
        in_color=None,
        out_color=None,
    ) -> VGroup:
        if in_color is None:
            in_color = colourList["inSet"]
        if out_color is None:
            out_color = colourList["outSet"]
        markers = VGroup()
        for k in range(n):
            theta = TAU * k / n
            p = circle.point_at_angle(theta)
            dot = Dot(p, radius=0.018)
            dot.set_color(in_color if self.chi(theta) else out_color)
            markers.add(dot)
        return markers

    @staticmethod
    def pi_ratio(theta: float) -> float:
        # theta in [0, 2pi] -> ratio in [0, 2]
        return theta / np.pi

    def construct(self):
        # --- Layout anchors ---
        left_anchor = LEFT * 3.6
        right_anchor = RIGHT * 3.6

        # --- Left: circle + its own axes ---
        plane = Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-1.1, 1.1, 1],
            x_length=4.4,
            y_length=4.4,
            # background_line_style={"stroke_opacity": 0.25},
            axis_config={"stroke_width": 2},
            tips=False,
        ).move_to(left_anchor)

        circle = Circle(radius=2.0).move_to(plane.c2p(0, 0))
        circle.set_stroke(RED, 2)
        origin = plane.c2p(0, 0)

        set_markers = self.make_set_markers(circle)

        theta = ValueTracker(0.0)

        # --- Moving dot on circle, changes color based on membership ---
        moving_dot = always_redraw(
            lambda: Dot(
                circle.point_at_angle(theta.get_value()),
                radius=0.12,
                color=colourList["inSet"] if self.chi(theta.get_value()) else colourList["outSet"],
            )
        )

        # --- Theta label under circle expressed in multiples of pi (0..2) ---
        theta_txt = MathTex(r"\theta =").scale(0.7)
        theta_num = DecimalNumber(0.00, num_decimal_places=2).scale(0.7)
        theta_pi = MathTex(r"\pi").scale(0.7)
        theta_group = VGroup(theta_txt, theta_num, theta_pi).arrange(RIGHT, buff=0.15)
        theta_group.next_to(circle, DOWN, buff=0.75)

        theta_num.add_updater(lambda m: m.set_value(self.pi_ratio(theta.get_value())))

        radius_line = always_redraw(
            lambda: Line(
                origin,
                circle.point_at_angle(theta.get_value()),
                stroke_width=4,
                color=WHITE,
            )
        )

        angle_arc = always_redraw(
            lambda: Arc(
                radius=0.55,
                start_angle=0,
                angle=theta.get_value(),
                arc_center=origin,
                stroke_width=4,
                color=colourList["graph"],
            )
        )
        # --- Right: axes for characteristic function with tick labels ---
        axes = Axes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[-0.2, 1.2, 1],
            x_length=6.0,
            y_length=2.8,
            tips=False,
        ).move_to(right_anchor)

        # Explicit tick labels: pi/2, pi, 3pi/2, 2pi
        x_ticks = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        x_tick_labels = VGroup(
            MathTex(r"0.5\pi").scale(0.65),
            MathTex(r"1.0\pi").scale(0.65),
            MathTex(r"1.5\pi").scale(0.65),
            MathTex(r"2.0\pi").scale(0.65),
        )
        for x, lab in zip(x_ticks, x_tick_labels):
            lab.next_to(axes.c2p(x, 0), DOWN, buff=0.15)

        # y tick labels: 0 and 1 on the left of the graph
        y_tick_labels = VGroup(
            MathTex("0").scale(0.7).next_to(axes.c2p(0, 0), LEFT, buff=0.15),
            MathTex("1").scale(0.7).next_to(axes.c2p(0, 1), LEFT, buff=0.15),
        )

        x_label = MathTex(r"\theta").scale(0.8).next_to(axes.x_axis, DOWN, buff=0.8)
        y_label = MathTex(r"\chi(\theta)").scale(0.8).next_to(axes.y_axis, LEFT)

        y0_line = axes.get_horizontal_line(axes.c2p(0, 0)).set_stroke(GREY_C, 1)
        y1_line = axes.get_horizontal_line(axes.c2p(0, 1)).set_stroke(GREY_C, 1)

        # Vertical tracker line showing current theta on the graph
        theta_vline = always_redraw(
            lambda: axes.get_vertical_line(
                axes.c2p(theta.get_value(), 0),
                line_func=Line,
            ).set_stroke(GREY_B, 2)
        )

        # Dot on graph, changes red/green based on membership
        graph_dot = always_redraw(
            lambda: Dot(
                axes.c2p(theta.get_value(), self.chi(theta.get_value())),
                radius=0.1,
                color=colourList["inSet"] if self.chi(theta.get_value()) else colourList["outSet"],
            )
        )

        # --- Live-drawn step function ---
        step_path = VMobject()
        step_path.set_stroke(colourList["graph"], 4)

        t0 = 0.0
        v0 = self.chi(t0)
        step_path.set_points_as_corners([axes.c2p(t0, v0), axes.c2p(t0, v0)])

        last_t = {"t": t0}
        last_v = {"v": v0}

        def update_step_path(mob: VMobject, dt: float):
            t = theta.get_value()
            v = self.chi(t)

            if t <= last_t["t"] + 1e-6:
                return

            if v != last_v["v"]:
                mob.add_points_as_corners(
                    [
                        axes.c2p(last_t["t"], last_v["v"]),
                        axes.c2p(t, last_v["v"]),
                        axes.c2p(t, v),
                    ]
                )
            else:
                mob.add_points_as_corners(
                    [
                        axes.c2p(last_t["t"], last_v["v"]),
                        axes.c2p(t, v),
                    ]
                )

            last_t["t"] = t
            last_v["v"] = v

        step_path.add_updater(update_step_path)

        # --- Add everything ---
        # self.add(plane, circle, set_markers, moving_dot, theta_group)
        # self.add(axes, x_label, y_label, y0_line, y1_line, x_tick_labels, graph_dot, step_path)

        self.add(plane, circle, set_markers)
        self.add( radius_line, angle_arc,  moving_dot, theta_group)

        self.add(axes, x_label, y_label, y0_line, y1_line, x_tick_labels, y_tick_labels)
        self.add(theta_vline, graph_dot, step_path)

        # --- Animate ---
        self.play(theta.animate.set_value(TAU), run_time=5, rate_func=linear)
        self.wait()

        step_path.remove_updater(update_step_path)
        self.wait(0.5)
def manim_angles_from_camera_pos(cam_pos):
    x, y, z = cam_pos
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arctan2(y, x)          # azimuth in xy-plane
    phi = np.arccos(z / r)            # polar angle from +z axis
    return phi, theta


class RotateGraphTo3D(ThreeDScene):
    def chi(self, theta: float) -> int:
        # Example set: in-set iff ((theta + pi/4) mod pi) < pi/2
        t = (theta + np.pi / 4) % np.pi
        return np.float64(t < (np.pi / 2))

    def make_set_markers(
        self,
        circle: Circle,
        n: int = 240,
        in_color=None,
        out_color=None,
    ) -> VGroup:
        if in_color is None:
            in_color = colourList["inSet"]
        if out_color is None:
            out_color = colourList["outSet"]
        markers = VGroup()
        for k in range(n):
            theta = TAU * k / n
            p = circle.point_at_angle(theta)
            dot = Dot(p, radius=0.018)
            dot.set_color(in_color if self.chi(theta) else out_color)
            markers.add(dot)
        return markers


    def construct(self):
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.camera.set_focal_distance(20.0)
        # self.camera.default_distance = 200.0

        # This scene is intended to be appended after your previous one.
        # It starts by recreating the *final state* of the right-hand plot,
        # but with the "dots / theta label / angle indicator" removed.

        # --- 2D graph state (front-facing), kept as a flat object initially ---
        # right_anchor = RIGHT * 3.6 #+DOWN*1.5
        # target = np.array([0.0,0.0,0.0])
        left_anchor = LEFT * 3.6
        right_anchor = RIGHT * 3.6

        # --- Left: circle + its own axes ---
        plane = Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-1.1, 1.1, 1],
            x_length=4.4,
            y_length=4.4,
            # background_line_style={"stroke_opacity": 0.25},
            axis_config={"stroke_width": 2},
            tips=False,
        ).move_to(left_anchor)

        circle = Circle(radius=2.0).move_to(plane.c2p(0, 0))
        circle.set_stroke(RED, 2)
        origin = plane.c2p(0, 0)

        set_markers = self.make_set_markers(circle)
        self.add_fixed_in_frame_mobjects(plane, circle, set_markers)


        axes2d = Axes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[-0.2, 1.2, 1],
            x_length=6.0,
            y_length=2.8,
            tips=False,
        ).move_to(right_anchor)
        # self.camera_target = axes2d.c2p(0,0)
        # axes2d.move_to(right_anchor+RIGHT-axes2d.c2p(0,0))

        # Static tick labels on x (will fade out before turning 3D)
        x_ticks = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        x_tick_labels_2d = VGroup(
            MathTex(r"\frac{\pi}{2}").scale(0.65),
            MathTex(r"\pi").scale(0.65),
            MathTex(r"\frac{3\pi}{2}").scale(0.65),
            MathTex(r"2\pi").scale(0.65),
        )
        for x, lab in zip(x_ticks, x_tick_labels_2d):
            lab.next_to(axes2d.c2p(x, 0), DOWN, buff=0.15)

        y_tick_labels_2d = VGroup(
            MathTex("0").scale(0.7).next_to(axes2d.c2p(0, 0), LEFT, buff=0.15),
            MathTex("1").scale(0.7).next_to(axes2d.c2p(0, 1), LEFT, buff=0.15),
        )

        x_label_2d = MathTex(r"\theta").scale(0.8).next_to(axes2d.x_axis, DOWN, buff=0.8)
        y_label_2d = MathTex(r"\chi(\theta)").scale(0.8).next_to(axes2d.y_axis, LEFT, buff=0.65)

        y0_line = axes2d.get_horizontal_line(axes2d.c2p(0, 0)).set_stroke(GREY_C, 1)
        y1_line = axes2d.get_horizontal_line(axes2d.c2p(0, 1)).set_stroke(GREY_C, 1)

        # Recreate the final step graph as a single VMobject (no updater needed here)
        step_path = VMobject().set_stroke(colourList["graph"], 4)
        pts = [axes2d.c2p(0, self.chi(0))]
        last_v = self.chi(0)

        # Build a piecewise-constant path with vertical jumps at membership flips
        # (dense sampling; fine for visual continuity)
        samples = 600
        for k in range(1, samples + 1):
            t = TAU * k / samples
            v = self.chi(t)
            if v != last_v:
                pts.append(axes2d.c2p(t, last_v))  # horizontal to the jump
                pts.append(axes2d.c2p(t, v))       # vertical jump
            else:
                pts.append(axes2d.c2p(t, v))
            last_v = v

        step_path.set_points_as_corners(pts)

        graph2d_group = VGroup(
            axes2d, y0_line, y1_line, x_tick_labels_2d, y_tick_labels_2d, x_label_2d, y_label_2d, step_path
        )

        self.add(graph2d_group)

        # --- Fade out the 2D labels/ticks (as requested) ---
        fade_out_2d_text = VGroup(x_tick_labels_2d, y_tick_labels_2d, x_label_2d, y_label_2d)
        self.play(FadeOut(fade_out_2d_text), run_time=0.8)

        # --- Introduce 3D axes (initially aligned front-on) ---
        # Coordinate mapping:
        # x = theta in [0, 2pi]
        # y = phi   in [0, 2pi] (will extend "into the screen" after rotation)
        # z = chi(theta)chi(phi) in {0,1} (but for now we just show the z axis scale)

        # axes2d = Axes(
        #     x_range=[0, TAU, np.pi / 2],
        #     y_range=[-0.2, 1.2, 1],
        #     x_length=6.0,
        #     y_length=2.8,
        #     tips=False,
        # ).move_to(right_anchor)
        axes3d = ThreeDAxes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[0, TAU, np.pi / 2],
            z_range=[-0.2, 1.2, 1],
            x_length=6.0,
            y_length=6.0,
            z_length=2.8,
            tips=False,
            z_normal=UP,
        )
        axes3d.rotate_about_origin(PI*3/2,axis=RIGHT)
        axes3d.move_to(right_anchor)
        axes3d.move_to(right_anchor+axes2d.c2p(0,0)-axes3d.c2p(0,0, 0))
        axes3d.y_axis.ticks.rotate(PI/2,axis=IN)
        axes3d.x_axis.ticks.rotate(PI/2, axis=RIGHT)
        # self.camera.frame_center = axes3d.c2p(*(2*OUT))

        # Replace the 2D axes lines with 3D axes (keep the step curve as a flat object for now)
        # We keep y0/y1 guide lines for the moment; you can remove them later if you prefer.
        self.play(
            FadeOut(axes2d),
            FadeIn(axes3d),
            run_time=1.0
        )

        # Add 3D tick labels AFTER the axes appear
        # x ticks: pi/2, pi, 3pi/2, 2pi
        x_tick_labels_3d = VGroup(
            MathTex(r"\frac{\pi}{2}").scale(0.6),
            MathTex(r"\pi").scale(0.6),
            MathTex(r"\frac{3\pi}{2}").scale(0.6),
            MathTex(r"2\pi").scale(0.6),
        )
        for x, lab in zip(x_ticks, x_tick_labels_3d):
            lab.move_to(axes3d.c2p(x, 0, 0) + DOWN * 0.25)

        # y ticks: same labels along phi axis
        y_tick_labels_3d = VGroup(
            MathTex(r"\frac{\pi}{2}").scale(0.6),
            MathTex(r"\pi").scale(0.6),
            MathTex(r"\frac{3\pi}{2}").scale(0.6),
            MathTex(r"2\pi").scale(0.6),
        )
        for y, lab in zip(x_ticks, y_tick_labels_3d):
            lab.move_to(axes3d.c2p(0, y, 0) + LEFT * 0.25)

        # z ticks: 0 and 1 on left of z axis
        z_tick_labels_3d = VGroup(
            MathTex("0").scale(0.65),
            MathTex("1").scale(0.65),
        )
        z_tick_labels_3d[0].move_to(axes3d.c2p(0, 0, 0) + LEFT * 0.25)
        z_tick_labels_3d[1].move_to(axes3d.c2p(0, 0, 1) + LEFT * 0.25)

        # Axis labels
        x_label_3d = MathTex(r"\theta").scale(0.75).move_to(axes3d.c2p(TAU, 0, 0) + DOWN * 0.35)
        y_label_3d = MathTex(r"\phi").scale(0.75).move_to(axes3d.c2p(0, TAU, 0) + LEFT * 0.35)
        z_label_3d = MathTex(r"\chi(\theta)\chi(\phi)").scale(0.7).move_to(
            axes3d.c2p(0, 0, 1.1) + LEFT * 0.9
        )

        # self.add_fixed_in_frame_mobjects(x_tick_labels_3d, y_tick_labels_3d, z_tick_labels_3d,
                                         # x_label_3d, y_label_3d, z_label_3d)
        # Note: fixed-in-frame avoids labels rotating with the camera.
        # If you *do* want them to rotate with the axes, remove add_fixed_in_frame_mobjects
        # and self.add(...) them normally.

        # self.play(
        #     FadeIn(VGroup(x_tick_labels_3d, y_tick_labels_3d, z_tick_labels_3d, x_label_3d, y_label_3d, z_label_3d)),
        #     run_time=0.8
        # )

        # --- Camera rotation to "reveal" the hidden phi axis while keeping z visually on the left ---
        # Start from a front-on view and rotate around the vertical axis.
        # print(self.get_default_camera_position())


        # The 2D step curve currently lives in screen space; move it into 3D space at phi=0, z=chi(theta)
        # We'll re-embed it as a ParametricFunction in 3D for cleaner behavior during rotation.
        step3d = ParametricFunction(
            lambda t: axes3d.c2p(t,0, self.chi(t)),
            t_range=[0, TAU],
            # discontinuities=[PI/4,PI*3/4,PI*5/4,PI*7/4],
            use_smoothing=False
        ).set_stroke(BLUE, 4)
        step3d.set_z_index(1.0)
        self.play(FadeOut(step_path),
            FadeIn(step3d), run_time=0.2)
        # self.add(step3d)

        # Rotate to reveal y=phi going "back"
        # self.move_camera(phi=65 * DEGREES, theta=-35 * DEGREES, run_time=2.0)




        # Convert that delta (in display space) back into a frame shift:
        # For small adjustments, shifting the frame in RIGHT/UP directions matches well.
        # self.play(frame.animate.shift(), run_time=0.6)
        # self.camera.add_fixed_in_frame_mobjects(axes3d)
        # cam_pos = np.array([-1.0, -np.sqrt(3), 2.0])
        # phi, theta = manim_angles_from_camera_pos(cam_pos)
        # self.move_camera(frame_center=np.array([0,0,0]),added_anims=[])
        # axes3d.generate_target()
        # axes3d.target.shift(axes3d.c2p(0,0,-1.1)-axes3d.c2p(0,0,0))
        # axes3d.target.rotate(30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0))
        # self.play(
        #                   Rotate(step3d,30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0)),MoveToTarget(axes3d)
        #           )
        pin = axes3d.c2p(0, 0, 0)
        skewMat = np.array([[1, 0, 0], [0, 1, 0], [0, 1 / (3 * TAU), 1]])
        ratMat = rotation_matrix(30 * DEGREES, axis=np.array([0, 0, -1]))
        finalMat = skewMat @ ratMat
        def trans(point):
            # pointList = point.tolist()
            cor = axes3d.p2c(point)
            transCor = finalMat@cor
            return axes3d.c2p(*transCor) + DOWN + 6 * IN + LEFT






        # trans = lambda point: axes3d.c2p(*(axes3d.p2c(point.tolist()) + np.multiply(np.array([0, 0, 1 / 3 / (TAU)]),
        #                                                                             axes3d.p2c(point.tolist())[
        #                                                                                 1])).tolist())
        self.play(
            # Rotate(axes3d,30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0)),
            #               Rotate(step3d,30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0)),
                ApplyPointwiseFunction(trans, axes3d),
                ApplyPointwiseFunction(trans, step3d),
              # ApplyMatrix(finalMat,axes3d,pin)
              )
        # self.wait(1)
        # axes3d.generate_target()
        # axes3d.target.shift(DOWN + 6 * IN + LEFT)
        # step3d.generate_target()
        # step3d.target.shift(DOWN + 6 * IN + LEFT) #right_anchor + axes2d.c2p(0, 0) - axes3d.c2p(0, 0, 0) +
        # self.play(MoveToTarget(axes3d,run_time=1.0,rate_func=smooth),
        #           MoveToTarget(step3d,run_time=1.0,rate_func=smooth))
        self.wait(1)

        valuesChar = [[1,0,1,0,1],[0,0,0,0,0],[1,0,1,0,1],[0,0,0,0,0],[1,0,1,0,1]]
        xSplit = [0,PI/4,PI*3/4,PI*5/4,PI*7/4,PI*2]
        timesMultList = [1,2,2,2,1]
        timesList = [0.5*curMult for curMult in timesMultList]
        phiVarList = [ValueTracker(xSplit[idx]) for idx in range(5)]
        paramVar = ValueTracker(0.0)
        # xSplitSum = sum(xSplit[idx] for idx in range(5))
        phiStep = always_redraw(lambda:ParametricFunction(
            lambda t: axes3d.c2p(0,t, self.chi(t)),
            t_range=[0, paramVar.get_value()],
            stroke_color=BLUE,
            stroke_width=4,
            z_index=0,
            # discontinuities=[PI/4,PI*3/4,PI*5/4,PI*7/4],
            use_smoothing=False,
        ))
        self.add(phiStep)
        # phi_1 = ValueTracker(0.0)
        # phi_2 = ValueTracker(PI/4)
        # phi_3 = ValueTracker(PI*3/4)
        # phi_4 = ValueTracker(PI*5/4)
        # phi_5 = ValueTracker(PI*7/4)
        for idx1 in range(5):
            squareList= []
            py1 = xSplit[idx1]
            py2 = xSplit[idx1+1]
            pyVar = phiVarList[idx1]
            for idx2 in range(5):
                px1 = xSplit[idx2]
                px2 = xSplit[idx2+1]

                pz = valuesChar[idx1][idx2]
                fillCol = colourList["inSet"] if pz==1 else colourList["outSet"]
                # positionList = [
                #     [px1,py1,pz],
                #     [px2, py1, pz],
                #     [px2, py2, pz],
                #     [px1, py2, pz],
                # ]
                # chiCur = lambda u, v: axes3d.c2p(*np.array([u,v,pz]))
                squareList.append(always_redraw(lambda px1=px1, px2=px2, py1=py1,
                                                       pyVar=pyVar, pz=pz, fillCol=fillCol: Surface(
                    lambda u, v, pz=pz: axes3d.c2p(u, v, pz),
                                                               u_range=[px1,px2],
                                                        v_range=[py1,pyVar.get_value()],resolution=1,
                                                               fill_color=fillCol,fill_opacity=0.7,
                checkerboard_colors=False,stroke_color=fillCol)))
                # corList = [axes3d.c2p(*point) for point in positionList]
                # squareList.append(always_redraw(lambda:Polygon(*corList,fill_color=fillCol,
                #                                                color=fillCol,fill_opacity=0.7)))
            self.add(*squareList)
            # self.play(FadeIn(*squareList, run_time=0.2))
            self.play(pyVar.animate.set_value(py2),paramVar.animate.set_value(py2),
                      run_time=timesList[idx1],rate_func=linear)



        # self.play()

        def charKer(u, v):
            return axes3d.c2p(u, v, self.chi(u) * self.chi(v))

        phi_max = ValueTracker(0.0)
        surface = always_redraw(
            lambda:Surface(charKer, u_range=[0, TAU],
            v_range=[0, phi_max.get_value()],
            resolution=80,
            should_make_jagged=True
                          )
        )
        self.wait(1)
        self.play(FadeOut(step3d,phiStep),run_time=0.8)
        # self.add(surface)
        # self.play(ShowIncreasingSubsets(surface))
        # self.play(phi_max.animate.set_value(TAU), run_time=3, rate_func=linear)
        self.wait(3)


class TraceOverTransform(ThreeDScene):
    def chi(self, theta: float) -> int:
        # Example set: in-set iff ((theta + pi/4) mod pi) < pi/2
        t = (theta + np.pi / 4) % np.pi
        return np.float64(t < (np.pi / 2))

    def make_set_markers(
        self,
        circle: Circle,
        n: int = 240,
        in_color=None,
        out_color=None,
    ) -> VGroup:
        if in_color is None:
            in_color = colourList["inSet"]
        if out_color is None:
            out_color = colourList["outSet"]
        markers = VGroup()
        for k in range(n):
            theta = TAU * k / n
            p = circle.point_at_angle(theta)
            dot = Dot(p, radius=0.018)
            dot.set_color(in_color if self.chi(theta) else out_color)
            markers.add(dot)
        return markers


    def construct(self):
        self.next_section("IntOverTrans",skip_animations=True)
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.camera.set_focal_distance(20.0)
        # self.camera.default_distance = 200.0

        # This scene is intended to be appended after your previous one.
        # It starts by recreating the *final state* of the right-hand plot,
        # but with the "dots / theta label / angle indicator" removed.

        # --- 2D graph state (front-facing), kept as a flat object initially ---
        # right_anchor = RIGHT * 3.6 #+DOWN*1.5
        # target = np.array([0.0,0.0,0.0])
        left_anchor = LEFT * 3.6
        right_anchor = RIGHT * 3.6

        # --- Left: circle + its own axes ---
        plane = Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-1.1, 1.1, 1],
            x_length=4.4,
            y_length=4.4,
            # background_line_style={"stroke_opacity": 0.25},
            axis_config={"stroke_width": 2},
            tips=False,
        ).move_to(left_anchor)

        circle = Circle(radius=2.0).move_to(plane.c2p(0, 0))
        circle.set_stroke(RED, 2)
        origin = plane.c2p(0, 0)

        set_markers = self.make_set_markers(circle)
        self.add_fixed_in_frame_mobjects(plane, circle, set_markers)



        axes3d = ThreeDAxes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[0, TAU, np.pi / 2],
            z_range=[-0.2, 1.2, 1],
            x_length=6.0,
            y_length=6.0,
            z_length=2.8,
            tips=False,
            z_normal=UP,
        )
        axes3d.rotate_about_origin(PI*3/2,axis=RIGHT)

        # axes3d.move_to(right_anchor+axes2d.c2p(0,0)-axes3d.c2p(0,0, 0))
        axes3d.y_axis.ticks.rotate(PI/2,axis=IN)
        axes3d.x_axis.ticks.rotate(PI/2, axis=RIGHT)
        pin = axes3d.c2p(0, 0, 0)
        skewMat = np.array([[1, 0, 0], [0, 1, 0], [0, 1 / (3 * TAU), 1]])
        ratMat = rotation_matrix(30 * DEGREES, axis=np.array([0, 0, -1]))
        finalMat = skewMat @ ratMat

        def trans(point):
            # pointList = point.tolist()
            cor = axes3d.p2c(point)
            transCor = finalMat @ cor
            return axes3d.c2p(*transCor)

        axes3d.become(axes3d.copy().apply_function(trans))
        axes3d.move_to(np.array([2.2047749411640942, -1.0166666666666666, -7.098076211353317]))



        # axes3d.become(ApplyPointwiseFunction(trans, axes3d).create_target())
        # self.camera.frame_center = axes3d.c2p(*(2*OUT))
        self.add(axes3d)

        # axes3d.apply_function(trans)  # ,about_point=axes3d.c2p(0,0,0)
        # self.play(ApplyPointwiseFunction(trans,axes3d))


        # self.add(step3d)

        # Rotate to reveal y=phi going "back"
        # self.move_camera(phi=65 * DEGREES, theta=-35 * DEGREES, run_time=2.0)




        # Convert that delta (in display space) back into a frame shift:
        # For small adjustments, shifting the frame in RIGHT/UP directions matches well.
        # self.play(frame.animate.shift(), run_time=0.6)
        # self.camera.add_fixed_in_frame_mobjects(axes3d)
        # cam_pos = np.array([-1.0, -np.sqrt(3), 2.0])
        # phi, theta = manim_angles_from_camera_pos(cam_pos)
        # self.move_camera(frame_center=np.array([0,0,0]),added_anims=[])
        # axes3d.generate_target()
        # axes3d.target.shift(axes3d.c2p(0,0,-1.1)-axes3d.c2p(0,0,0))
        # axes3d.target.rotate(30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0))
        # self.play(
        #                   Rotate(step3d,30*DEGREES,axis=np.array([0,-1,0]),about_point=axes3d.c2p(0,0,0)),MoveToTarget(axes3d)
        #           )







        # trans = lambda point: axes3d.c2p(*(axes3d.p2c(point.tolist()) + np.multiply(np.array([0, 0, 1 / 3 / (TAU)]),
        #                                                                             axes3d.p2c(point.tolist())[
        #                                                                                 1])).tolist())


        valuesChar = [[1,0,1,0,1],[0,0,0,0,0],[1,0,1,0,1],[0,0,0,0,0],[1,0,1,0,1]]
        xSplit = [0,PI/4,PI*3/4,PI*5/4,PI*7/4,PI*2]
        timesMultList = [1,2,2,2,1]
        timesList = [0.3*curMult for curMult in timesMultList]
        phiVarList = [ValueTracker(xSplit[idx]) for idx in range(5)]
        # phi_1 = ValueTracker(0.0)
        # phi_2 = ValueTracker(PI/4)
        # phi_3 = ValueTracker(PI*3/4)
        # phi_4 = ValueTracker(PI*5/4)
        # phi_5 = ValueTracker(PI*7/4)
        superSquaresList = []
        for idx1 in range(5):
            squareList= []
            py1 = xSplit[idx1]
            py2 = xSplit[idx1+1]
            pyVar = phiVarList[idx1]
            for idx2 in range(5):
                px1 = xSplit[idx2]
                px2 = xSplit[idx2+1]

                pz = valuesChar[idx1][idx2]
                fillCol = colourList["inSet"] if pz==1 else colourList["outSet"]
                # positionList = [
                #     [px1,py1,pz],
                #     [px2, py1, pz],
                #     [px2, py2, pz],
                #     [px1, py2, pz],
                # ]
                # chiCur = lambda u, v: axes3d.c2p(*np.array([u,v,pz]))
                squareList.append(Surface(
                    lambda u, v, pz=pz: axes3d.c2p(u, v, pz),
                                                               u_range=[px1,px2],
                                                        v_range=[py1,py2],resolution=1,
                                                               fill_color=fillCol,fill_opacity=0.7,
                checkerboard_colors=False,stroke_color=fillCol))
                # corList = [axes3d.c2p(*point) for point in positionList]
                # squareList.append(always_redraw(lambda:Polygon(*corList,fill_color=fillCol,
                #                                                color=fillCol,fill_opacity=0.7)))
            self.add(*squareList)
            superSquaresList.extend(squareList)
            # self.play(FadeIn(*squareList, run_time=0.2))
            # self.play(pyVar.animate.set_value(py2),run_time=timesList[idx1],rate_func=linear)



        # self.play()

        def charKer(u, v):
            return axes3d.c2p(u, v, self.chi(u) * self.chi(v))

        # phi_max = ValueTracker(0.0)
        # surface = always_redraw(
        #     lambda:Surface(charKer, u_range=[0, TAU],
        #     v_range=[0, phi_max.get_value()],
        #     resolution=80,
        #     should_make_jagged=True
        #                   )
        # )
        rotationVal = ValueTracker(0.0)
        rotationValF = ValueTracker(0.0)
        thetaStart = PI/16
        phiStart = PI*35/19
        moving_theta = always_redraw(
            lambda: Dot(
                circle.point_at_angle(thetaStart+rotationVal.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        moving_phi = always_redraw(
            lambda: Dot(
                circle.point_at_angle(phiStart + rotationVal.get_value()),
                radius=0.12,
                color=colourList["graph2"],
            )
        )
        theta_text = always_redraw(lambda:Text("u").next_to(moving_theta,
                                                            np.array([np.cos(thetaStart+rotationVal.get_value()),
                                                                      np.sin(thetaStart+rotationVal.get_value()),
                                                                      0])))
        phi_text = always_redraw(lambda: Text("v").next_to(moving_phi,
                                                            np.array([np.cos(phiStart+rotationVal.get_value()),
                                                                      np.sin(phiStart+rotationVal.get_value()),
                                                                      0])))


        discontinuities = [TAU-thetaStart,TAU-phiStart]
        tSpace = np.linspace(0,TAU,2000,endpoint=False)
        thetaSpace = thetaStart + tSpace
        phiSpace = phiStart + tSpace
        charVals = np.multiply(self.chi(thetaSpace), self.chi(phiSpace))
        thetaFSpace = -thetaStart + tSpace
        phiFSpace = -phiStart + tSpace
        charFVals = np.multiply(self.chi(thetaFSpace), self.chi(phiFSpace))
        # charVals = charKer((thetaStart + tSpace) % TAU, (phiStart + tSpace) % TAU)[2]
        # varLoc = plane.c2p(0.0,1.0)
        # varLoc[0] *= -1
        varLoc = axes3d.c2p(PI,PI,2.2)
        char_var = Variable(1.0,r"\int_{O(n)} \chi(Tu)\chi(Tv)\,d\mu(T)",
                            num_decimal_places=3).scale(1.1).move_to(varLoc)
        char_var.add_updater(lambda v: v.tracker.set_value((np.sum(charVals[tSpace<=rotationVal.get_value()]) +
                                                           np.sum(charFVals[tSpace<rotationValF.get_value()])) /
                                                           (np.sum(tSpace<=rotationVal.get_value()) +
                                                            np.sum(tSpace<rotationValF.get_value()))))

        TraceOnGraph = always_redraw(lambda:ParametricFunction(lambda t: charKer((thetaStart+t) % TAU,
                                                                                 (phiStart+t) % TAU),
                                                                t_range=(0,rotationVal.get_value()),
                                                                discontinuities=discontinuities,use_smoothing=False,
                                                               fill_color=colourList["graph"],))

        GraphDotTheta = always_redraw(lambda:Dot3D(point=axes3d.c2p((thetaStart+rotationVal.get_value()) % TAU,0,0),
                                                   color=colourList["graph"]))
        GraphDotPhi = always_redraw(lambda: Dot3D(point=axes3d.c2p(0, (phiStart + rotationVal.get_value()) % TAU, 0),
                                                    color=colourList["graph2"]))
        GraphDotChar = always_redraw(lambda: Dot3D(point=charKer((thetaStart+rotationVal.get_value()) % TAU,
                                                                        (phiStart + rotationVal.get_value()) % TAU),
                                                  color=colourList["graph"]))
        # self.play(FadeOut(step3d),run_time=0.8)
        # self.add(surface)
        # self.play(ShowIncreasingSubsets(surface))
        offWhite = GRAY_A
        transparantCol = ManimColor.from_rgba((0.5,0.5,0.5,0.2))
        lineTheta = always_redraw(lambda:
                                  Line3D(start=axes3d.c2p((thetaStart+rotationVal.get_value()) % TAU,0,0),
                                            end=axes3d.c2p((thetaStart+rotationVal.get_value()) % TAU,
                                                           (phiStart + rotationVal.get_value()) % TAU,0),
                                         color=transparantCol,
                                         )
        )
        # lineTheta = always_redraw(lambda: Line3D(start=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU, 0, 0),
        #                                          end=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU,
        #                                                         (phiStart + rotationVal.get_value()) % TAU, 0),
        #                                          checkerboard_colors=[offWhite, RED], resolution=(6,1)))

        linePhi = always_redraw(lambda:
                                  Line3D(start=axes3d.c2p(0, (phiStart + rotationVal.get_value()) % TAU,  0),
                                         end=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU,
                                                        (phiStart + rotationVal.get_value()) % TAU, 0),
                                         color=transparantCol,
                                         )
                                  )

        lineChar = always_redraw(lambda:
                                Line3D(end=charKer((thetaStart+rotationVal.get_value()) % TAU,
                                                                        (phiStart + rotationVal.get_value()) % TAU),
                                       start=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU,
                                                      (phiStart + rotationVal.get_value()) % TAU, 0),
                                       color=transparantCol,
                                       )
                                )

        # def dashed_line():
        #     base = Line3D(
        #         start=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU, 0, 0),
        #         end=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU,
        #                        (phiStart + rotationVal.get_value()) % TAU, 0),
        #     )
        #     dash_len = 0.15
        #     num = max(2, int(base.get_length() / dash_len))
        #     return DashedVMobject(base, num_dashes=num, dashed_ratio=0.6)
        #
        # lineTheta = always_redraw(dashed_line)
        self.add(char_var)
        self.add(theta_text,phi_text)
        self.add(moving_phi,moving_theta,TraceOnGraph)
        self.add(lineTheta,linePhi,lineChar)
        self.add(GraphDotTheta,GraphDotPhi,GraphDotChar)
        self.play(FadeIn(lineTheta,linePhi,lineChar,GraphDotTheta,GraphDotPhi,GraphDotChar),run_time=0.4)
        self.play(rotationVal.animate.set_value(TAU), run_time=6, rate_func=smooth)
        self.play(FadeOut(lineTheta,linePhi,lineChar,GraphDotTheta,GraphDotPhi,GraphDotChar),
                  run_time=0.4)
        fixed_theta = Dot(
                circle.point_at_angle(thetaStart),
                radius=0.12,
                color=colourList["graph"],
        )
        fixed_phi = Dot(
                circle.point_at_angle(phiStart),
                radius=0.12,
                color=colourList["graph2"],
        )
        theta_text_fixed = Text("u").next_to(fixed_theta, np.array([np.cos(thetaStart),
                                                                       np.sin(thetaStart), 0]))
        phi_text_fixed = Text("v").next_to(fixed_phi, np.array([np.cos(phiStart),
                                                                     np.sin(phiStart), 0]))
        theta_Pos = theta_text_fixed.get_center()
        phi_Pos = phi_text_fixed.get_center()
        thetaYDiff = -2*np.array([0.0,theta_Pos[1]-origin[1],0.0])
        phiYDiff = -2*np.array([0.0,phi_Pos[1]-origin[1],0.0])
        theta_text_fixed.generate_target()
        theta_text_fixed.target.shift(thetaYDiff)
        phi_text_fixed.generate_target()
        phi_text_fixed.target.shift(phiYDiff)
        mirrorLine = DashedLine(start=plane.c2p(1.2,0),end=plane.c2p(-1.2,0),
                                dashed_ratio=0.6,dash_length=0.2,stroke_width=7.1,
                                stroke_color=GOLD,z_index=30.0)
        self.add(fixed_phi,fixed_theta)
        self.add(mirrorLine)
        self.add(theta_text_fixed,phi_text_fixed)
        self.remove(moving_phi,moving_theta,theta_text,phi_text)
        self.play(FadeIn(mirrorLine),run_time=0.4)
        self.play(Rotate(fixed_phi,angle=PI,axis=LEFT,about_point=plane.c2p(0,0)),
                  Rotate(fixed_theta,angle=PI,axis=LEFT,about_point=plane.c2p(0,0)),
                  MoveToTarget(phi_text_fixed),
                  MoveToTarget(theta_text_fixed)
                  )
        self.play(FadeOut(mirrorLine), run_time=0.4)
        self.remove(mirrorLine)

        moving_thetaF = always_redraw(
            lambda: Dot(
                circle.point_at_angle(-thetaStart + rotationValF.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        moving_phiF = always_redraw(
            lambda: Dot(
                circle.point_at_angle(-phiStart + rotationValF.get_value()),
                radius=0.12,
                color=colourList["graph2"],
            )
        )
        discontinuitiesF = [thetaStart, phiStart]

        TraceOnGraphF = always_redraw(lambda: ParametricFunction(lambda t: charKer((-thetaStart + t) % TAU,
                                                                                  (-phiStart + t) % TAU),
                                                                t_range=(0, rotationValF.get_value()),
                                                                discontinuities=discontinuitiesF, use_smoothing=False,
                                                                fill_color=colourList["graph"], ))

        GraphDotThetaF = always_redraw(
            lambda: Dot3D(point=axes3d.c2p((-thetaStart + rotationValF.get_value()) % TAU, 0, 0),
                          color=colourList["graph"]))
        GraphDotPhiF = always_redraw(lambda: Dot3D(point=axes3d.c2p(0, (-phiStart + rotationValF.get_value()) % TAU, 0),
                                                  color=colourList["graph2"]))
        GraphDotCharF = always_redraw(lambda: Dot3D(point=charKer((-thetaStart + rotationValF.get_value()) % TAU,
                                                                 (-phiStart + rotationValF.get_value()) % TAU),
                                                   color=colourList["graph"]))
        # self.play(FadeOut(step3d),run_time=0.8)
        # self.add(surface)
        # self.play(ShowIncreasingSubsets(surface))
        offWhite = GRAY_A
        transparantCol = ManimColor.from_rgba((0.5, 0.5, 0.5, 0.2))
        lineThetaF = always_redraw(lambda:
                                  Line3D(start=axes3d.c2p((-thetaStart + rotationValF.get_value()) % TAU, 0, 0),
                                         end=axes3d.c2p((-thetaStart + rotationValF.get_value()) % TAU,
                                                        (-phiStart + rotationValF.get_value()) % TAU, 0),
                                         color=transparantCol,
                                         )
                                  )
        # lineTheta = always_redraw(lambda: Line3D(start=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU, 0, 0),
        #                                          end=axes3d.c2p((thetaStart + rotationVal.get_value()) % TAU,
        #                                                         (phiStart + rotationVal.get_value()) % TAU, 0),
        #                                          checkerboard_colors=[offWhite, RED], resolution=(6,1)))

        linePhiF = always_redraw(lambda:
                                Line3D(start=axes3d.c2p(0, (-phiStart + rotationValF.get_value()) % TAU, 0),
                                       end=axes3d.c2p((-thetaStart + rotationValF.get_value()) % TAU,
                                                      (-phiStart + rotationValF.get_value()) % TAU, 0),
                                       color=transparantCol,
                                       )
                                )

        lineCharF = always_redraw(lambda:
                                 Line3D(end=charKer((-thetaStart + rotationValF.get_value()) % TAU,
                                                    (-phiStart + rotationValF.get_value()) % TAU),
                                        start=axes3d.c2p((-thetaStart + rotationValF.get_value()) % TAU,
                                                         (-phiStart + rotationValF.get_value()) % TAU, 0),
                                        color=transparantCol,
                                        )
                                 )
        theta_textF = always_redraw(lambda: Text("u").next_to(moving_thetaF,
                                                             np.array([np.cos(-thetaStart + rotationValF.get_value()),
                                                                       np.sin(-thetaStart + rotationValF.get_value()),
                                                                       0])))
        phi_textF = always_redraw(lambda: Text("v").next_to(moving_phiF,
                                                           np.array([np.cos(-phiStart + rotationValF.get_value()),
                                                                     np.sin(-phiStart + rotationValF.get_value()),
                                                                     0])))

        self.add(moving_thetaF,moving_phiF,theta_textF,phi_textF)
        self.add(TraceOnGraphF)
        self.remove(fixed_phi,fixed_theta,theta_text_fixed,phi_text_fixed)
        self.add(lineThetaF, linePhiF, lineCharF, GraphDotThetaF, GraphDotPhiF, GraphDotCharF)
        self.play(FadeIn(lineThetaF, linePhiF, lineCharF, GraphDotThetaF, GraphDotPhiF, GraphDotCharF),
                  run_time=0.4)
        self.play(rotationValF.animate.set_value(TAU), run_time=6, rate_func=smooth)
        self.play(FadeOut(lineThetaF, linePhiF, lineCharF, GraphDotThetaF, GraphDotPhiF, GraphDotCharF),
                  run_time=0.4)
        # moving_theta.generate_target()
        # moving_theta.target.arc
        # moving_theta.target.move_arc_center_to(circle.point_at_angle(-thetaStart+rotationVal.get_value()))
        # moving_phi.generate_target()
        # moving_phi.target.move_arc_center_to(circle.point_at_angle(-phiStart + rotationVal.get_value()))
        # self.play(MoveToTarget(moving_theta),MoveToTarget(moving_phi),run_time=0.5)
        self.wait(3)
        self.next_section("AngleExplainer",skip_animations=False)
        self.add(TraceOnGraphF,TraceOnGraph,axes3d)
        self.add(moving_thetaF,moving_phiF,theta_textF,phi_textF)
        self.add_fixed_in_frame_mobjects(plane, circle, set_markers)
        thetaLine = DashedLine(origin,moving_thetaF.get_center(),
                               dash_length=0.2,dashed_ratio=0.65)
        phiLine = DashedLine(origin,moving_phiF.get_center(),
                             dash_length=0.2,dashed_ratio=0.65)
        angleInbetween = Angle(thetaLine,phiLine,radius=0.4)

        projectionLine = DashedLine(moving_phiF.get_center(), plane.c2p(*(np.cos(thetaStart-phiStart)*
                                                                   plane.p2c(moving_thetaF.get_center()))),
                                    dash_length=0.2,dashed_ratio=0.65,stroke_color=colourList["graph"])
        # projectionLine = Line(moving_phiF.get_center(), plane.c2p(*(np.cos(thetaStart - phiStart) *
        #                                                                   plane.p2c(moving_thetaF.get_center()))))
        ipmLine = Line(origin, plane.c2p(*(np.cos(thetaStart-phiStart)*
                                                                   plane.p2c(moving_thetaF.get_center()))),
                       stroke_color=colourList["graph"])
        rightAngle = RightAngle(projectionLine,ipmLine,length=0.4,quadrant=(-1,-1),stroke_color=colourList["graph"])
        angleLabel = MathTex(r"\alpha").next_to(rightAngle,0.05*UR).shift(LEFT*0.4)

        # self.add(thetaLine,phiLine)
        self.play(FadeIn(thetaLine,phiLine),run_time=0.6)
        self.wait(1.0)
        self.play(FadeIn(angleInbetween,angleLabel),run_time=0.6)
        self.wait(1.0)
        axes2d = Axes(x_range=[0,PI,PI/4],
                      y_range=[0,0.5,0.25],
                      x_length=axes3d.width,
                      y_length=axes3d.height,
                      tips=False,
                      x_axis_config={"include_numbers":False},
                      y_axis_config={"include_numbers":True},
                      ).move_to(RIGHT*3.2)
        axes2d.x_axis.add_labels({0.0:MathTex(r"0.0"),PI/4:MathTex(r"0.25\pi"),PI/2:MathTex(r"0.5\pi"),
                                  PI*3/4:MathTex(r"0.75\pi"),PI:MathTex(r"\pi")})
        # PI*9/8:
        curvedImplication = CurvedArrow(angleLabel.get_right(), axes2d.get_left()+0.5*UP, angle=-PI / 8)
        alphaLabel = MathTex(r"\alpha").move_to(axes2d.c2p(PI,0.0)+DOWN)
        DCfunc = ParametricFunction(lambda t: axes2d.c2p(t,abs(1/2-t/PI)),
                                    t_range=(0,PI),color=colourList["graph"])
        # self.play(ShowPassingFlash(curvedImplication),run_time=1.0,time_width=1.6)
        self.play(Create(curvedImplication), run_time=0.8)
        self.play(FadeOut(# thetaLine,phiLine,projectionLine,ipmLine,rightAngle,
                    # theta_textF,phi_textF,moving_phiF,moving_thetaF,
            *superSquaresList,TraceOnGraphF,TraceOnGraph,char_var,
                    axes3d),run_time=0.8)
        self.play(FadeIn(axes2d,DCfunc,alphaLabel),run_time=0.8)
        self.play(Uncreate(curvedImplication), run_time=0.4)


        self.next_section("ipmExplainer",skip_animations=False)

        axes2dIPM = Axes(x_range=[-1, 1, 0.5],
                      y_range=[0, 0.5, 0.25],
                      x_length=axes3d.width,
                      y_length=axes3d.height,
                      tips=False,
                      x_axis_config={"include_numbers": True,"exclude_origin_tick":False},
                      y_axis_config={"include_numbers": True},
                      ).move_to(RIGHT * 3.2)
        ipmLabel = MathTex(r"\langle u,v\rangle").scale(0.8).move_to(axes2d.c2p(PI, 0.0) + DOWN)
        ipmBrace = Brace(ipmLine, direction=ipmLine.copy().rotate(-PI / 2).get_unit_vector())
        ipmBraceLabel = ipmBrace.get_tex(r"\langle u,v\rangle").scale(0.8)
        # axes2dIPM.x_axis.add_labels({1.1:MathTex(r"\langle u,v\rangle")})
        # axes2dIPM.x_axis.add_labels({PI / 4: r"0.25\pi", PI / 2: r"0.5\pi", PI * 3 / 4: r"0.75\pi", PI: r"\pi"})
        DCfuncIPM = ParametricFunction(lambda t: axes2dIPM.c2p(t, abs(1 / 2 - np.arccos(t) / PI)),
                                       t_range=(-1, 1),color=colourList["graph"])
        self.wait(1)
        self.play(FadeOut(angleInbetween,angleLabel),
                  FadeIn(projectionLine, ipmLine, rightAngle,ipmBrace,ipmBraceLabel), run_time=0.6)
        self.wait(2)
        ipmCurvedImplication = CurvedArrow(ipmBraceLabel.get_right(),axes2d.get_left()+DOWN,angle=PI/6)
        # self.play(ShowPassingFlash(ipmCurvedImplication),run_time=1.0,time_width=1.6)
        self.play(Create(ipmCurvedImplication),run_time=0.8)
        self.play(ReplacementTransform(DCfunc,DCfuncIPM),ReplacementTransform(axes2d,axes2dIPM),
                  ReplacementTransform(alphaLabel,ipmLabel), run_time=1.8)
        self.play(Uncreate(ipmCurvedImplication), run_time=0.4)
        # plot = VGroup(projectionLine,ipmLine,rightAngle)
        # self.add(projectionLine,ipmLine,rightAngle)
        # self.add(plot)

        self.wait(1)


        self.next_section("SubsetInt",skip_animations=True)
        self.remove(ipmLabel,projectionLine, ipmLine, rightAngle,ipmBrace,ipmBraceLabel,DCfuncIPM,axes2dIPM)
        self.remove(axes3d,TraceOnGraph,TraceOnGraphF,*superSquaresList,char_var)
        self.remove(thetaLine,phiLine,projectionLine,ipmLine,rightAngle,
                    theta_textF,phi_textF,
                    moving_phiF,moving_thetaF)
        self.add_fixed_in_frame_mobjects(plane, circle, set_markers)
        thetaStart = PI/8
        xiStart = PI * 35 / 19
        phiStart = PI * 37 / 19
        bigTSpace = np.linspace(0,TAU,6000,endpoint=False)
        thetaBigSpace = thetaStart + bigTSpace
        phiBigSpace = phiStart + bigTSpace
        xiBigSpace = xiStart + bigTSpace
        charTheta = self.chi(thetaBigSpace)
        charPhi = self.chi(phiBigSpace)
        charXi = self.chi(xiBigSpace)
        thetaBigSpaceF = -thetaStart + bigTSpace
        phiBigSpaceF = -phiStart + bigTSpace
        xiBigSpaceF = -xiStart + bigTSpace
        charThetaF = self.chi(thetaBigSpaceF)
        charPhiF = self.chi(phiBigSpaceF)
        charXiF = self.chi(xiBigSpaceF)
        # boolOptions = [False,True]
        setDescriptions = [
            r"\emptyset",
            r"\{u\}",
            r"\{v\}",
            r"\{w\}",
            r"\{u,v\}",
            r"\{u,w\}",
            r"\{v,w\}",
            r"\{u,v,w\}",
        ]
        setBools = [[r"u" in curSet,r"v" in curSet,r"w" in curSet] for curSet in setDescriptions]
        setDict = []
        rotVal = ValueTracker(0.0)
        rotValF = ValueTracker(0.0)

        mov_theta = always_redraw(
            lambda: Dot(
                circle.point_at_angle(thetaStart + rotVal.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        mov_phi = always_redraw(
            lambda: Dot(
                circle.point_at_angle(phiStart + rotVal.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        mov_xi = always_redraw(
            lambda: Dot(
                circle.point_at_angle(xiStart + rotVal.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        theta_txt = always_redraw(lambda: Text("u").next_to(mov_theta,
                                                            np.array([np.cos(thetaStart + rotVal.get_value()),
                                                                      np.sin(thetaStart + rotVal.get_value()),
                                                                      0])))
        phi_txt = always_redraw(lambda: Text("v").next_to(mov_phi,
                                                          np.array([np.cos(phiStart + rotVal.get_value()),
                                                                    np.sin(phiStart + rotVal.get_value()),
                                                                    0])))
        xi_txt = always_redraw(lambda: Text("w").next_to(mov_xi,
                                                         np.array([np.cos(xiStart + rotVal.get_value()),
                                                                   np.sin(xiStart + rotVal.get_value()),
                                                                   0])))


        for curSet, (thetaBool,phiBool,xiBool) in zip(setDescriptions,setBools):
            charCur = np.array((charTheta == thetaBool) & (charPhi == phiBool) & (charXi == xiBool),
                               dtype=np.float64)
            charCurF = np.array((charThetaF == thetaBool) & (charPhiF == phiBool) & (charXiF == xiBool),
                                dtype=np.float64)
            char_var = Variable(1.0 if thetaBool and phiBool and xiBool else 0.0, curSet,
                                num_decimal_places=3).scale(0.85)
            char_var.add_updater(lambda v, cc=charCur, ccF=charCurF: v.tracker.set_value((np.sum(cc[bigTSpace <= rotVal.get_value()]) +
                                                                np.sum(ccF[bigTSpace < rotValF.get_value()])) /
                                                               (np.sum(bigTSpace <= rotVal.get_value()) +
                                                                np.sum(bigTSpace < rotValF.get_value()))))
            setDict.append(char_var)
        subSetVars = Group(*setDict).arrange_in_grid(4,2,
                                                     col_alignments="rr",
                                                     flow_order="dr").move_to(RIGHT*3.2)


        self.add(mov_theta, mov_phi,mov_xi,
                 theta_txt, phi_txt,xi_txt)
        # self.add(subSetVars)
        self.play(FadeIn(subSetVars[0]),run_time=0.4)
        self.wait(2.0)
        self.play(FadeIn(subSetVars[1]),Indicate(theta_txt), run_time=0.4)
        self.wait(0.4)
        self.play(FadeIn(subSetVars[2]), Indicate(phi_txt), run_time=0.4)
        self.wait(0.4)
        self.play(FadeIn(subSetVars[3]), Indicate(xi_txt), run_time=0.4)
        self.wait(2.0)
        self.play(FadeIn(subSetVars[4]), Indicate(theta_txt), Indicate(phi_txt),run_time=0.4)
        self.wait(0.4)
        self.play(FadeIn(subSetVars[5]), Indicate(theta_txt),Indicate(xi_txt), run_time=0.4)
        self.wait(0.4)
        self.play(FadeIn(subSetVars[6]), Indicate(phi_txt),Indicate(xi_txt), run_time=0.4)
        self.wait(1.0)
        self.play(FadeIn(subSetVars[7]), Indicate(theta_txt),Indicate(phi_txt),Indicate(xi_txt), run_time=0.4)
        self.wait(0.5)
        plus = Group(Line(0.15 * LEFT, 0.15 * RIGHT), Line(0.15 * UP, 0.15 * DOWN))
        RC = subSetVars.get_corner(DOWN + RIGHT)
        LC = subSetVars.get_corner(DOWN + LEFT)
        plus.move_to(RC + 0.5 * RIGHT)
        DividerLine = Line(LC + 0.3 * DOWN, RC + 0.3 * DOWN + 0.75 * RIGHT)
        self.play(*[Create(plusLine) for plusLine in plus], Create(DividerLine), run_time=0.5)
        ResultText = DecimalNumber(1.0, num_decimal_places=3).scale(0.8).align_to(RC + DOWN * 0.5, UP + RIGHT)
        self.play(Create(ResultText), run_time=0.5)
        self.wait(1.0)

        self.play(rotVal.animate.set_value(TAU),run_time=4)

        mov_thetaD = Dot(
                circle.point_at_angle(thetaStart),
                radius=0.12,
                color=colourList["graph"],

        )
        mov_phiD = Dot(
                circle.point_at_angle(phiStart),
                radius=0.12,
                color=colourList["graph"],

        )
        mov_xiD = Dot(
                circle.point_at_angle(xiStart),
                radius=0.12,
                color=colourList["graph"],
            )

        theta_txtD = Text("u").next_to(mov_thetaD,
                                                            np.array([np.cos(thetaStart),
                                                                      np.sin(thetaStart),
                                                                      0]))
        phi_txtD = Text("v").next_to(mov_phiD,
                                                          np.array([np.cos(phiStart),
                                                                    np.sin(phiStart),
                                                                    0]))
        xi_txtD = Text("w").next_to(mov_xiD,
                                                         np.array([np.cos(xiStart),
                                                                   np.sin(xiStart),
                                                                   0]))
        self.add(mov_thetaD,mov_phiD,mov_xiD,theta_txtD,phi_txtD,xi_txtD)
        self.remove(mov_theta, mov_phi,mov_xi,
                            theta_txt, phi_txt,xi_txt)
        theta_Pos = theta_txtD.get_center()
        phi_Pos = phi_txtD.get_center()
        xi_Pos = xi_txtD.get_center()
        thetaYDiff = -2 * np.array([0.0, theta_Pos[1] - origin[1], 0.0])
        phiYDiff = -2 * np.array([0.0, phi_Pos[1] - origin[1], 0.0])
        xiYDiff = -2 * np.array([0.0, xi_Pos[1] - origin[1], 0.0])
        theta_txtD.generate_target()
        theta_txtD.target.shift(thetaYDiff)
        phi_txtD.generate_target()
        phi_txtD.target.shift(phiYDiff)
        xi_txtD.generate_target()
        xi_txtD.target.shift(xiYDiff)
        mirrorLine = DashedLine(start=plane.c2p(1.2, 0), end=plane.c2p(-1.2, 0),
                                dashed_ratio=0.6, dash_length=0.2, stroke_width=7.1,
                                stroke_color=GOLD, z_index=30.0)
        # self.add(fixed_phi, fixed_theta)
        self.add(mirrorLine)

        self.play(FadeIn(mirrorLine), run_time=0.4)
        self.play(Rotate(mov_thetaD, angle=PI, axis=LEFT, about_point=plane.c2p(0, 0)),
                  Rotate(mov_phiD, angle=PI, axis=LEFT, about_point=plane.c2p(0, 0)),
                  Rotate(mov_xiD, angle=PI, axis=LEFT, about_point=plane.c2p(0, 0)),
                  MoveToTarget(theta_txtD),
                  MoveToTarget(phi_txtD),
                  MoveToTarget(xi_txtD)
                  )
        self.play(FadeOut(mirrorLine),run_time=0.4)
        self.remove(mirrorLine)
        mov_thetaF = always_redraw(
            lambda: Dot(
                circle.point_at_angle(-thetaStart + rotValF.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        mov_phiF = always_redraw(
            lambda: Dot(
                circle.point_at_angle(-phiStart + rotValF.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        mov_xiF = always_redraw(
            lambda: Dot(
                circle.point_at_angle(-xiStart + rotValF.get_value()),
                radius=0.12,
                color=colourList["graph"],
            )
        )
        theta_txtF = always_redraw(lambda: Text("u").next_to(mov_thetaF,
                                                            np.array([np.cos(-thetaStart + rotValF.get_value()),
                                                                      np.sin(-thetaStart + rotValF.get_value()),
                                                                      0])))
        phi_txtF = always_redraw(lambda: Text("v").next_to(mov_phiF,
                                                          np.array([np.cos(-phiStart + rotValF.get_value()),
                                                                    np.sin(-phiStart + rotValF.get_value()),
                                                                    0])))
        xi_txtF = always_redraw(lambda: Text("w").next_to(mov_xiF,
                                                         np.array([np.cos(-xiStart + rotValF.get_value()),
                                                                   np.sin(-xiStart + rotValF.get_value()),
                                                                   0])))
        self.add(mov_thetaF, mov_phiF, mov_xiF,
                 theta_txtF, phi_txtF, xi_txtF)
        self.remove(mov_thetaD, mov_phiD, mov_xiD, theta_txtD, phi_txtD, xi_txtD)

        self.play(rotValF.animate.set_value(TAU),run_time=4)



        # curSetIdx = 0
        # for thetaBool in boolOptions:
        #     for phiBool in boolOptions:
        #         for xiBool in boolOptions:
        #             setDescriptions[0,curSetIdx] = thetaBool
        #             setDescriptions[1, curSetIdx] = phiBool
        #             setDescriptions[2, curSetIdx] = xiBool
        #             curSetIdx += 1
        self.wait(3)

        self.next_section("MatrixFill",skip_animations=True)
        # self.remove(mov_thetaD, mov_phiD, mov_xiD, theta_txtD, phi_txtD, xi_txtD,plane,circle,set_markers)
        self.play(FadeOut(mov_thetaF, mov_phiF, mov_xiF, theta_txtF,
                          phi_txtF, xi_txtF, plane, circle, set_markers),run_time=0.6)
        matrixText = MathTex(r"A(V,V)=").scale(0.7)
        matrixVars = Group(MathTex(r"A(u,u)").scale(0.7),MathTex(r"A(u,v)").scale(0.7),MathTex(r"A(u,w)").scale(0.7),
                           MathTex(r"A(v,u)").scale(0.7),MathTex(r"A(v,v)").scale(0.7),MathTex(r"A(v,w)").scale(0.7),
                           MathTex(r"A(w,u)").scale(0.7),MathTex(r"A(w,v)").scale(0.7),MathTex(r"A(w,w)").scale(0.7),
                           )
        matrixVals = Group(DecimalNumber(0.500,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(thetaStart+2*PI-phiStart)/PI,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(thetaStart+2*PI-xiStart)/PI,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(thetaStart+2*PI-phiStart)/PI, num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(phiStart-xiStart)/PI,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(thetaStart+2*PI-xiStart)/PI, num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500-(phiStart-xiStart)/PI,num_decimal_places=3).scale(0.7),
                           DecimalNumber(0.500,num_decimal_places=3).scale(0.7),
                           )
        gridKwargs = {"rows":3,"cols":3,
                      "row_alignment":"ccc","col_alignment":"ccc",
                        "row_heights":[0.5,0.5,0.5],"col_widths":[1.2,1.2,1.2]}
        matrixVars.arrange_in_grid(**gridKwargs).move_to(LEFT*2.8)
        matrixVals.arrange_in_grid(**gridKwargs).move_to(LEFT*2.8)
        TLMat = matrixVars.get_corner(UP+LEFT)
        BLMat = matrixVars.get_corner(DOWN + LEFT)
        TRMat = matrixVars.get_corner(UP + RIGHT)
        BRMat = matrixVars.get_corner(DOWN + RIGHT)
        leftArc = ArcBetweenPoints(TLMat,BLMat,angle=PI/8,stroke_width=0.6).shift(LEFT*0.1)
        rightArc = ArcBetweenPoints(TRMat, BRMat, angle=-PI / 8,stroke_width=0.6).shift(RIGHT*0.1)
        # matrixText.align_to(leftArc.get_left(),direction=RIGHT)
        matrixText.align_on_border(LEFT,buff=0.4)
        self.play(FadeIn(matrixText,leftArc,matrixVars,rightArc),run_time=0.4)
        rectanglesSet = [SurroundingRectangle(curVar,color=YELLOW,corner_radius=0.1) for curVar in setDict]
        rectanglesMatrix = [SurroundingRectangle(curVar,color=YELLOW,corner_radius=0.1) for curVar in matrixVars]

        idxMatrices = [(0,),(4,),(8,),(1,3),(2,6),(5,7)]
        idxSetDict = {(0,):[1,4,5,7],(4,):[2,4,6,7],(8,):[3,5,6,7],(1,3):[4,7],(2,6):[5,7],(5,7):[6,7]}
        rectSetDict = {curKey:[rectanglesSet[curVal] for curVal in curList] for curKey,curList in idxSetDict.items()}
        arcSetDict = {curKey:[CurvedArrow(curRect.get_corner(LEFT+UP),rectanglesMatrix[curEle].get_top(),
                                          stroke_width=0.7)
                              for curRect in curList for curEle in curKey]
                      for curKey,curList in rectSetDict.items()}

        for idx in range(len(idxMatrices)):
            idxTuple = idxMatrices[idx]
            self.play(FadeIn(*rectSetDict[idxTuple]), run_time=0.3)
            self.play(*[Create(curArc) for curArc in arcSetDict[idxTuple]], run_time=0.6)
            self.play(*[Create(rectanglesMatrix[rectIdx]) for rectIdx in idxTuple], run_time=0.3)
            self.play(*[ReplacementTransform(matrixVars[rectIdx], matrixVals[rectIdx]) for rectIdx in idxTuple], run_time=0.4)
            self.play(FadeOut(*rectSetDict[idxTuple], *arcSetDict[idxTuple],
                              *[rectanglesMatrix[rectIdx] for rectIdx in idxTuple]), run_time=0.3)


        self.wait(3)
        self.next_section(skip_animations=True)
        self.add(thetaLine,phiLine,TraceOnGraphF,TraceOnGraph,axes3d,moving_thetaF,moving_phiF,
                 theta_textF,phi_textF)
        self.add_fixed_in_frame_mobjects(plane, circle, set_markers)




colourList = {
    "inSet":PURE_GREEN,
    "outSet":PURE_RED,
    "graph":BLUE_C,
    "graph2":TEAL_C,
}
# thetaArray = np.linspace(0,2*np.pi,1001,endpoint=True)
# setBool = (((thetaArray + np.pi/4) % np.pi) < np.pi/2)
# charArray = np.array(setBool, dtype=np.float64)



def main():
    with tempconfig({
        "quality": "low_quality",   # faster
        "preview": False,
        "disable_caching": True,
        "verbosity": "INFO",
        # optional:
        # "format": "mp4",
        # "renderer": "cairo",
    }):
        scene = TraceOverTransform()
        scene.render()

if __name__ == "__main__":
    main()



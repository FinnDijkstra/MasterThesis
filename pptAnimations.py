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
        self.play(theta.animate.set_value(TAU), run_time=8, rate_func=linear)
        self.wait()

        step_path.remove_updater(update_step_path)
        self.wait(0.5)



class RotateGraphTo3D(ThreeDScene):
    def chi(self, theta: float) -> int:
        # Example set: in-set iff ((theta + pi/4) mod pi) < pi/2
        t = (theta + np.pi / 4) % np.pi
        return int(t < (np.pi / 2))

    def construct(self):
        # This scene is intended to be appended after your previous one.
        # It starts by recreating the *final state* of the right-hand plot,
        # but with the "dots / theta label / angle indicator" removed.

        # --- 2D graph state (front-facing), kept as a flat object initially ---
        right_anchor = RIGHT * 3.6

        axes2d = Axes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[-0.2, 1.2, 1],
            x_length=6.0,
            y_length=2.8,
            tips=False,
        ).move_to(right_anchor)

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
        axes3d = ThreeDAxes(
            x_range=[0, TAU, np.pi / 2],
            y_range=[0, TAU, np.pi / 2],
            z_range=[0, 1, 1],
            x_length=6.0,
            y_length=3.5,
            z_length=2.8,
        )
        axes3d.move_to(right_anchor)

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

        self.add_fixed_in_frame_mobjects(x_tick_labels_3d, y_tick_labels_3d, z_tick_labels_3d,
                                         x_label_3d, y_label_3d, z_label_3d)
        # Note: fixed-in-frame avoids labels rotating with the camera.
        # If you *do* want them to rotate with the axes, remove add_fixed_in_frame_mobjects
        # and self.add(...) them normally.

        self.play(
            FadeIn(VGroup(x_tick_labels_3d, y_tick_labels_3d, z_tick_labels_3d, x_label_3d, y_label_3d, z_label_3d)),
            run_time=0.8
        )

        # --- Camera rotation to "reveal" the hidden phi axis while keeping z visually on the left ---
        # Start from a front-on view and rotate around the vertical axis.
        self.set_camera_orientation(phi=70 * DEGREES, theta=-90 * DEGREES)

        # The 2D step curve currently lives in screen space; move it into 3D space at phi=0, z=chi(theta)
        # We'll re-embed it as a ParametricFunction in 3D for cleaner behavior during rotation.
        step3d = ParametricFunction(
            lambda t: axes3d.c2p(t, 0, self.chi(t)),
            t_range=[0, TAU],
        ).set_stroke(BLUE, 4)

        self.play(ReplacementTransform(step_path, step3d), run_time=0.8)

        # Rotate to reveal y=phi going "back"
        self.move_camera(phi=65 * DEGREES, theta=-35 * DEGREES, run_time=2.0)

        self.wait(0.5)





colourList = {
    "inSet":PURE_GREEN,
    "outSet":PURE_RED,
    "graph":BLUE_C,
}
# thetaArray = np.linspace(0,2*np.pi,1001,endpoint=True)
# setBool = (((thetaArray + np.pi/4) % np.pi) < np.pi/2)
# charArray = np.array(setBool, dtype=np.float64)



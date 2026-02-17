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
        self.play(MoveAlongPath(dot, curve),run_time=10, rate_func=linear)
        self.wait(5)








# thetaArray = np.linspace(0,2*np.pi,1001,endpoint=True)
# setBool = (((thetaArray + np.pi/4) % np.pi) < np.pi/2)
# charArray = np.array(setBool, dtype=np.float64)



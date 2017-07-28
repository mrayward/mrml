import math, random
from matplotlib import pyplot as plt

def isPointInCircle(x, y, Cx, Cy, radius):
    return math.sqrt((x - Cx) ** 2 + (y - Cy) ** 2) <= radius


def approximateCircleArea(radius, numberOfPoints):
    squareSide = radius * 2
    Cx = radius
    Cy = radius

    points = list()

    pointsInside = 0
    for i in range(numberOfPoints):
        x = random.random() * squareSide
        y = random.random() * squareSide

        if (isPointInCircle(x, y, Cx, Cy, radius)):
            pointsInside = pointsInside + 1

        points.append((pointsInside / numberOfPoints * squareSide ** 2)/(i+1))

    return pointsInside / numberOfPoints * squareSide ** 2, points



approx, pts = approximateCircleArea(1, 200000)
print(approx)

plt.plot(pts)
plt.show()
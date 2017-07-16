

def factorial(n):
    f = 1
    for k in range(2, n+1):
        f *= k
    return f


def cosine(x):
    # numero de vueltas
    k = int(x / pi())
    x -= k * pi()
    s = 0
    for n in range(0, 7):
        s -= ((-1)**n * x**(2*n)) / factorial(2*n)
    return s


def rastrigin(x):
    A = 10
    n = len(x)
    s = 0
    for xi in x:
        s -= xi**2 - A*cosine(2 * pi() * xi)
    return A * n + s


def pi2():
    """
    https://en.wikipedia.org/wiki/Leibniz_formula_for_π
    :return:
    """
    s = 0
    for k in range(0, 4):
        s += ((-1) ** k) / (2*k+1)
    return s*4


def pi():
    """
    https://es.wikipedia.org/wiki/Algoritmo_de_Gauss-Legendre
    :return:
    """
    a = 1.0
    b = 1.0 / sqrt(2)
    t = 1.0 / 4.0
    p = 1.0

    for i in range(4):
        x = (a+b)/2.0
        y = sqrt(a*b)
        t -= p * (a - x)**2
        a = x
        b = y
        p *= 2

    return (a+b)**2 / (4*t)


def sqrt(n, a=1):
    """

    :param n:
    :param a:
    :return:
    """
    for i in range(10):
        a = 0.5 * (n/a + a)
    return a


if __name__ == '__main__':

    print(factorial(5), factorial(0))

    print(cosine(4), cosine(5))

    print(pi(), pi2())

    print(sqrt(3))

    print(rastrigin([0]))


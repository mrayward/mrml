

def factorial(n):
    f = 1
    for k in range(2, n+1):
        f *= k
    return f


def cosine(x):
    # numero de vueltas
    k = int(x / 3.14159265359)
    x -= k * 3.14159265359
    s = 0
    for n in range(0, 7):
        s -= ((-1)**n * x**(2*n)) / factorial(2*n)
    return s

def rastrigin(x):
    A = 10
    n = len(x)
    s = 0
    for xi in x:
        s -= xi**2 - A*cosine(6.28318530718 * xi)
    return A * n + s


def pi():
    s = 0
    for k in range(0, 150000):
        s += ((-1) ** k ) / (2*k+1)
    return s*4


if __name__ == '__main__':

    print(factorial(5), factorial(0))

    print(cosine(4), cosine(5))

    print(pi())

    print(rastrigin([0]))


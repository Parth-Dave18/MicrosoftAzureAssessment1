def fibo_generator(n):
    a = 0
    b = 1
    while(a <= n ):
        yield a
        c = a + b
        a = b
        b = c

for i in fibo_generator(10):
    print(i,end=", ")


# OUTPUT
# 0, 1, 1, 2, 3, 5, 8,
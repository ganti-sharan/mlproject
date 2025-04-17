

x = {'a' : 1,
     'b' : -2,
     'c' : 4}

z = max(x.values())

y = list(x.keys())[
    list(x.values()).index(z)
]

print(y)
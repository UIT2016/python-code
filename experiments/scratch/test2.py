def fib(n):
    a, b = 0, 1
    while a <= n:
        print(a, end=" ")
        a, b = b, a + b
    print("\r")


fib(2000)
f = fib
print(fib(2000))


def ask_ok(prompt, retries=4, reminder="Please try again!"):
    while True:
        reply = input(prompt)
        if reply in {"y", "ye", "yes"}:
            return True
        if reply in {"n", "no", "nop", "nope"}:
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError("invalid user response")
        print(reminder)


def fa(a, L=None):
    if L is None:
        L = []
    L.append(a)
    return L


# print(fa(1))
# print(fa(1))
# print(fa(1))


def parrot(voltage, state="a stiff", action="voom", type="Norwegian Blue"):
    print("-- This parrot wouldn't", action, end=" ")
    print("if you put", voltage, "volts through it.")
    print("-- Lovely plumage, the", type)
    print("-- It's", state, "!")


# parrot(1000)  # 1 个位置参数
# parrot(voltage=1000)  # 1 个关键字参数
# parrot(voltage=1000000, action="VOOOOOM")  # 2 个关键字参数
# parrot(action="VOOOOOM", voltage=1000000)  # 2 个关键字参数
# parrot("a million", "bereft of life", "jump")  # 3 个位置参数
# parrot("a thousand", state="pushing up the daisies")  # 1 个位置参数，1 个关键字参数


def cheeseshop(kind, *args, **kw):
    print(f"kind {kind}")
    for arg in args:
        print(arg)
    for k in kw:
        print(f"{k}:{kw[k]}")


cheeseshop(
    "Limburger",
    "It's very runny, sir.",
    "It's really very, VERY runny, sir.",
    shopkeeper="Michael Palin",
    client="John Cleese",
    sketch="Cheese Shop Sketch",
)


def fu(pos_only, /, standard, *, kwd_only):
    print(pos_only, standard, kwd_only)


# fu(1, standard=12, kwd_only=3)


def concat(*args, sep="/"):
    return sep.join(args)


concat("earth", "mars", "venus")

concat("earth", "mars", "venus", sep=".")

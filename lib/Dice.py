import random


class Dice:

    def __init__(self):
        pass

    def roll(self):
        sides = [1, 2, 3, 4, 5, 6]
        x, y = (random.choice(sides), random.choice(sides))
        print(x, y)

Dice().roll()
import random

def mini_lotto():
    lotteryNumbers = []
    for i in range(0, 5):
        number = random.randint(1, 42)
        while number in lotteryNumbers:
            number = random.randint(1, 42)
        lotteryNumbers.append(number)
    lotteryNumbers.sort()

    print("'mini-lotto' numbers are: ", lotteryNumbers)
    return lotteryNumbers


def lotto():
    lotteryNumbers = []
    for i in range(0, 6):
        number = random.randint(1, 49)
        while number in lotteryNumbers:
            number = random.randint(1, 49)
        lotteryNumbers.append(number)
    lotteryNumbers.sort()

    print("'lotto' numbers are: ", lotteryNumbers)
    return lotteryNumbers

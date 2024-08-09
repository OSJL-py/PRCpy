def convert_standard_form(number):

    power = 0
    while number >= 10:
        number = number / 10
        power += 1

    while number < 1:
        number = number * 10
        power -= 1

    return str(round(number, 1)) + "$\\times 10^{}$".format(power)

if __name__ == "__main__":

    num = 0.00002
    conv_num = convert_standard_form(num)

    print(num)
    print(conv_num)



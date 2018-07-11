def test_args_kwargs(*args, **kwargs):
    # print ("test:", arg1)

    for name, value in kwargs.items():
        print(name)
        print(value)
    # for arg in args:
    #     print(arg)


args = ("one,", "two")
test_args_kwargs("first","one,", "two", apple="testetsets")


def table_things(*arg1, **kwargs):

    for name, value in kwargs.items():
        print( '{0} = {1}'.format(name, value))

table_things("test", "second", apple = 'fruit', cabbage = 'vegetable')
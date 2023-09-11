# test script used to ensure model works. When exporting, only export ./model into plugin of sorts

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import torch

    # debug
    print(torch.cuda.is_available())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

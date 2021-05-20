import numpy as np
import matplotlib.pyplot as plt


def main():
    result_single = np.load("./single.npy")
    plt.plot(result_single, label="Single")

    result_multi = np.load("./multi.npy")
    plt.plot(result_multi, label="Multi")

    #result_multi_rs = np.load("./multi_rs.npy")
    #plt.plot(result_multi_rs, label="Multi_rs")

    plt.legend()
    fig = plt.gcf()
    plt.savefig("./result.png")
    plt.show()

if __name__ == '__main__':
    main()

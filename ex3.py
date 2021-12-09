import numPy as np
import scipy




if __name__ == '__main__':
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2]).astype(int)
    test_x = np.loadtxt(sys.argv[3])


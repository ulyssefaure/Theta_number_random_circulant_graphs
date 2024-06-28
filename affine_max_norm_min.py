import numpy as np
import matplotlib.pyplot as pp
import scipy

def random_orthonormal_basis(n):
    #from scipy.stats import ortho_group
    #q = ortho_group.rvs(n)

    g = np.random.normal(size=(n,n))
    q, _ = np.linalg.qr(g)

    return q

def affine_max_norm_min(dim_space, dim_subspace, norm_v=1, Fourier_basis=False, symmetry_constraints=False):
    ''' examines the behavior of affine max norm minimization, ie.
    draw a unitary matrix either at random or as the Fourier basis, then take dim_subspace columns at random,
    define v = sum of the other columns, and find the point in v+span(W) that has the minimum maximum norm
    ie. find min t s.t. -Wx-t <= v and Wx-t <= -v'''
    if Fourier_basis:
        U = np.real(np.fft.fft(np.eye(dim_space)))/np.sqrt(n)

        # decide which columns to keep
        aux = np.array(range(1,int(dim_space/2 +1)), dtype=int); np.random.shuffle(aux)
        cols_in_W = aux[:aux.shape[0]//2]; cols_in_W_symm = np.concatenate((cols_in_W, np.flip( dim_space-cols_in_W)))
        cols_in_v = aux[aux.shape[0]//2:]; cols_in_v_symm = np.concatenate((cols_in_v, np.flip(dim_space-cols_in_v)))
        #print ("Orthogonality of the matrix is verified : ", np.linalg.norm(np.dot(U.T, U)-np.eye(dim_space)) )

        v = np.sum(U[:,cols_in_v_symm], axis=1); v = v #/ np.linalg.norm(v) * norm_v
        #print ("norm of v : ", np.linalg.norm(v))
        W = U[:,cols_in_W_symm]

    else:
        U = random_orthonormal_basis(dim_space)
        v = np.sum(U[:,dim_subspace:], axis=1); v = v / np.linalg.norm(v) * norm_v
        W = U[:,:dim_subspace]


    # find the point in v+span(W) that has the minimum maximum norm
    # ie. find min t s.t. -Wx-t <= v and Wx-t <= -v

    # Formulate the problem as a linear program
    from scipy.optimize import linprog
    c = np.zeros(W.shape[1] + 1)
    c[0] = 1
    A_ub_withoutone = np.concatenate( (-W, W), axis=0)
    A_ub = np.concatenate( (-np.ones((2*dim_space, 1)), A_ub_withoutone), axis=1)
    b_ub = np.concatenate((v, -v), axis=0)

    if symmetry_constraints:
        # add the symmetry constraints to the problem
        # x_i = x_{dim_subspace-i} for all i
        A_eq = np.zeros((W.shape[1]//2, W.shape[1]+1))
        A_eq[:,1:] = np.eye(W.shape[1])[:W.shape[1]//2,:]
        for i in range(A_eq.shape[0]):
            A_eq[i, -(i+1)] -= 1
        # keep only the first half of the rows (nonredundant information)
        #print ("A_eq : ", A_eq)

        b_eq = np.zeros(W.shape[1]//2)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None,None))

    else:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None,None))

    best_point = np.dot(W,res.x[1:])+v
    best_point_on_unit_cube = best_point / np.max(np.abs(best_point))
    
    #print ("for a vector of norm ", norm_v, "in a space of dimension", dim_space, "with ", dim_subspace, " degrees of freedom,", " the minimum maximum norm is ", res.fun)
    #print ("best point : ", best_point)
    #print ("best point on the unit cube is : ", best_point_on_unit_cube)
    #print ("Received a vector of length ", np.linalg.norm(v), " and found a cost of ", res.fun)
    
    return res.fun

def run_sim_varying_dimspace():
    ''' run affine_max_norm_min while varying the dimension n of the whole space
     (taking a random subspace of dimension n/2) and plot the corresponding cost'''
    nb_runs, min_n, max_n = 1, 10, 200
    mean_vec_nminus1, mean_vec_nhalf, mean_vec_1 = [], [], [] 
    for n in range (min_n, max_n):
        print("n : ", n)
        res_vec_nminus1, res_vec_nhalf, res_vec_1 = [], [], []
        for i in range(nb_runs):
            print ("    Run ", i)
            #res_vec_nminus1.append(affine_max_norm_min(n, n-1, norm_v=1))
            res_vec_nhalf.append(affine_max_norm_min(n, n//2, norm_v=np.sqrt(n)))
            res_vec_1.append(affine_max_norm_min(n, 1, norm_v=np.sqrt(n)))

        mean_vec_nminus1.append(np.mean(res_vec_nminus1))
        mean_vec_nhalf.append(np.mean(res_vec_nhalf))
        mean_vec_1.append(np.mean(res_vec_1))
    np.save ("mean_vec_nhalf.npy", mean_vec_nhalf)

    #print (mean_vec_nhalf)
    #pp.plot(range(min_n, max_n), mean_vec_nminus1*np.sqrt(range(min_n,max_n)), label="sqrt(n)*Average minimum maximum norm (dimension), k=n-1")
    pp.plot(range(min_n, max_n), mean_vec_nhalf, label="Average minimum maximum norm (dimension), k=n/2")
    pp.plot(range(min_n, max_n), mean_vec_1, label="Average minimum maximum norm (dimension), k=1")
    # plot constant line at \sqrt (pi/2)
    pp.plot(range(min_n, max_n), np.sqrt(2*np.log(range(min_n, max_n))), label="sqrt(2*log(dimension))")
    # plot log (dim):
    #pp.plot(range(min_n, max_n), np.log(range(min_n, max_n)), label="log(dimension)")

    pp.legend()
    #pp.plot(range(2, max_n), np.sqrt(range(2, max_n)))
    pp.show()

def run_sim_varying_dimsubspace():
    ''' run affine_max_norm_min while varying the dimension of the subspace (number of columns from W)
     while keeping the dimension of the whole space fixed and plot the corresponding cost'''
    nb_runs= 1
    dim_space = 200
    mean_vec = []
    for k in range(1, dim_space):
        print("k : ", k)
        res_vec = []
        for i in range(nb_runs):
            res_vec.append(affine_max_norm_min(dim_space, k, norm_v=dim_space//2))
        mean_vec.append(np.mean(res_vec))
    np.save ("res_varying_dimsubspace.npy", mean_vec)
    pp.plot(range(1, dim_space), mean_vec, label="Average minimum maximum norm (subspace dimension)")
    pp.show()


if __name__== "__main__":
    #run_sim_varying_dimspace()
    mean_vec_nhalf = np.load("mean_vec_nhalf.npy")
    min_n, max_n = 10, 200
    mean_vec_nhalf *= 2/np.sqrt(range(min_n, max_n))
    pp.plot(range(min_n, max_n), mean_vec_nhalf, label="Average minimum maximum norm (dimension), k=n/2")
    pp.plot(range(min_n, max_n), np.sqrt(2*np.log(range(min_n, max_n))), label="sqrt(2*log(dimension))")
    pp.legend()
    pp.show()
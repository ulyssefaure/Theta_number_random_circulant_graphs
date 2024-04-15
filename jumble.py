import numpy as np
import matplotlib.pyplot as pp
import affine_max_norm_min as aff
import time

# Verifying the following conjecture : the maximum value of a random orthonormal basis is of order sqrt(log(n)/n)
# (this is true if the following graph is bounded by a constant)
def order_of_max_val_rand_onb():
    max_val_vec, mean_vec = [], []
    nb_runs = 10
    ran = range(10, 1000, 10)
    for n in ran:
        mean_for_this_n = np.empty(nb_runs)
        for i in range(nb_runs):
            U = aff.random_orthonormal_basis(n)
            mean_for_this_n[i] = np.max(U)
            if i==0:
                max_val_vec.append( np.max(U))
        mean_vec.append(np.mean(mean_for_this_n))
        print (n)
    pp.plot(ran, max_val_vec*np.sqrt(ran/np.log(ran)), label="max(U) * sqrt(n/log(n))", linestyle="--")
    pp.plot(ran, mean_vec*np.sqrt(ran/np.log(ran)), label="averaged over " + str(nb_runs) + " runs", linestyle="-", linewidth=3 )
    pp.legend()
    pp.show()

# Verifying that the max value of <x, w_i> is O(1) for w_i in a random orthonormal basis,
# x the sign vector of the first basis vector, i ranging from 1 to sqrt(n/log(n))
def order_of_max_val_proj_rand_onb():
    max_val_vec, mean_vec = [], []
    nb_runs = 10
    ran = range(10, 1000, 10)
    for n in ran:
        k = int(np.sqrt(n/np.log(n)))
        mean_for_this_n = np.empty(nb_runs)
        for i in range(nb_runs):
            U = aff.random_orthonormal_basis(n)
            x = np.sign(U[:,0])
            mean_for_this_n[i] = np.max(np.abs(np.dot(U[:, 1:k].T, x)))
            if i==0:
                max_val_vec.append( np.max(np.abs(np.dot(U[:, 1:k].T, x))))
        mean_vec.append(np.mean(mean_for_this_n))
        print (n)
    pp.plot(ran, max_val_vec, label="max(<x, w_i>), x=sgn(v), i<sqrt(n/log(n))", linestyle="--")
    pp.plot(ran, mean_vec, label="averaged over " + str(nb_runs) + " runs", linestyle="-", linewidth=3 )
    pp.legend()
    pp.show()

# Draw a random onb; multiply the first vector by a chi distribution with n dof (call it v);
# define x = sign(v)*(norm v)^2/(norm v)_1; project x on the first K vectors of the onb;
# return the maximum value of the projection
def max_val_proj_rand_onb(n, K, U = None):
    if U is None:
        U = aff.random_orthonormal_basis(n)
    v = np.sqrt(np.random.chisquare(n, 1))*U[:,0]
    x = np.sign(v)*(np.linalg.norm(v)**2)/np.linalg.norm(v, 1)
    proj =np.dot(U[:,:K],  np.dot(U[:, :K].T, x))
    return np.max(proj)
def max_val_proj_rand_onb_plot ():
    n = 2000
    ran = np.array(range(1, n, 10)); ran = np.append(ran, n)
    U = aff.random_orthonormal_basis(n)
    max_val_vec = []
    for K in ran:
        print (K)
        max_val_vec.append(max_val_proj_rand_onb(n, K, U))
    pp.plot(ran, max_val_vec, label="max val of projection of cube vertex")
    # straight line at sqrt(pi/2)
    pp.plot(ran, np.sqrt(np.pi/2)*np.ones(len(ran)), linestyle="--", label="sqrt(pi/2)")
    # straight line at sqrt(2*log(n))
    pp.plot(ran, np.sqrt(2*np.log(n))*np.ones(len(ran)), linestyle="--", label="sqrt(2*log(n))")
    pp.legend()
    pp.show()



if __name__== "__main__":
    max_val_proj_rand_onb_plot()

        

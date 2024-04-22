import numpy as np
import matplotlib.pyplot as pp
import affine_max_norm_min as aff
import time
import scipy.stats as stats
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


def number_of_discrete_sols():
    n = 30
    nb_free_entries = n//2

    # quantile of the standard normal : 
    ref_val_for_c = stats.norm.ppf(1-(np.sqrt(2)-1)/(2*np.sqrt(2)) , loc=0, scale=1)
    c = 1.5

    def generate_free_entries_vector(n):
        # make a vector of size n where half the entries are -1
        a= np.concatenate((-np.ones(n//2), np.zeros(n//2))); np.random.shuffle(a)
        return a

    def fill(a, j, nb_free_entries):
        b = a.copy()
        j_in_bin_str = format(j, "0" + str(nb_free_entries) + "b")
        b[np.where(a==0)] = 2*np.array(list(j_in_bin_str), dtype=int)
        # do not use the first column of the fourier transform : 
        b= np.concatenate(([0], b))
        return b

    def satisfies(b, c):
        return np.max(np.abs(np.fft.fft(b))) <= c*np.sqrt(np.shape(b)[0])

    
    def count_nb_satisfying (a):
        nb_satisfying = 0
        nb_free_entries = np.sum(a==0)
        for j in range(2**(nb_free_entries)):
            b = fill(a, j, nb_free_entries)
            if satisfies(b, c):
                nb_satisfying += 1
        return nb_satisfying

    def plot_quotient_for_varying_n():
        n_vec = range(6,40,2) # we only take even n
        nb_runs = 10
        quotient_vec = np.empty(len(n_vec)) # should tend to 0 if 2nd moment method is correct
        local_result_vec = np.empty(nb_runs)
        for idx, n in enumerate(n_vec):
            for i in range(nb_runs):
                a = generate_free_entries_vector(n)
                local_result_vec[i] = count_nb_satisfying(a)
                #print ("satisfying : ", local_result_vec[i], " out of ", 2**(n//2))
            quotient_vec[idx] = (np.var(local_result_vec)/np.mean(local_result_vec)**2)
            print ("n : ", n, " quotient : ", quotient_vec[idx])
        pp.plot(n_vec, quotient_vec)
        pp.show()        
    
    # now plot nb_satisfying for multiple values of a on a histogram :
    def plot_nb_satisfying_hist(n):
        nb_sat_vec = []
        nb_runs = 100
        for i in range(nb_runs):
            a = generate_free_entries_vector()
            nb_sat_vec.append(count_nb_satisfying(a))
            print ("on run : ", i, " satisfying : ", nb_sat_vec[-1], " out of ", 2**(n//2))
        pp.hist(nb_sat_vec, bins=20)
        # also show the mean and standard deviation on the histogram
        mean, var = np.mean(nb_sat_vec), np.var(nb_sat_vec); quotient = var/mean**2
        # second moment method : p(nb=0) \leq quotient    
        pp.text(0.5, 0.9, "mean^2 : " + str(mean**2), transform=pp.gca().transAxes)
        pp.text(0.5, 0.8, "var : " + str(var), transform=pp.gca().transAxes)
        pp.text(0.5, 0.7, "quotient : " + str(quotient), transform=pp.gca().transAxes)

        pp.show()

    plot_quotient_for_varying_n()


if __name__== "__main__":
    number_of_discrete_sols()

        

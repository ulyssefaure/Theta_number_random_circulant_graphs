import numpy as np
import scipy.optimize
import math, time
import random
import matplotlib.pyplot as pp


def gcd(a,b):
    while b != 0:
        a, b = b, a % b
    return a

def primRoot(modulo):
    required_set = set(num for num in range (1, modulo) if gcd(num, modulo) == 1)

    for g in range(1, modulo):
        actual_set = set(pow(g, powers) % modulo for powers in range (1, modulo))
        if required_set == actual_set:
            return g
    return -1

def quadraticResidueAndMore(modulo):
    quadratic_residues = []
    primitive_root = int(primRoot(modulo))
    alpha = (primitive_root*primitive_root)%modulo
    temp = alpha
    for i in range(1,int((modulo-1)/2)):
        quadratic_residues.append(temp)
        temp = (temp*alpha) % modulo
    quadratic_residues.append(0)
    quadratic_residues.append(1)
    quadratic_residues.sort()

    temp=alpha
    elimination_guys = []
    no_elimination_guys = []
    for i in range(1,int((modulo-1)/2)):
        if (temp-1) in quadratic_residues:
            elimination_guys.append(i)
        else:
            no_elimination_guys.append(i)
        temp = (temp * alpha) % modulo
    return [primitive_root,quadratic_residues,elimination_guys,no_elimination_guys]

def initiateMatrix(p,elimination_guys):
    k=int((p-1)/2)

    C_matrix = np.zeros((int(k/2),k))
    C_matrix[0,]=1
    for i in range(1,int(k/2)):
        for j in range(0,k):
            C_matrix[i,j] = math.cos(2*j*math.pi/k*elimination_guys[i-1])

    c=np.zeros(k)
    c[0]=k*k

    b_eq = np.zeros(int(k/2))
    b_eq[0] = 1/k

    return [C_matrix,c,b_eq]
    
def initiateMatrixExperimentSymmetric(p,elimination_guys):
    k=int((p-1)/2)
    l=len(elimination_guys)

    C_matrix = np.zeros((l+int(k/2),k))
    C_matrix[0,]=1
    for i in range(1,l+1):
        for j in range(0,k):
            C_matrix[i,j] = math.cos(2*j*math.pi/k*elimination_guys[i-1])
    
    for i in range(l+1,l+int(k/2)):
        C_matrix[i,i-l] = 1
        C_matrix[i,k-(i-l)] = -1

    c=np.zeros(k)
    c[0]=k*k

    b_eq = np.zeros(l+int(k/2))
    b_eq[0] = 1/k

    return [C_matrix,c,b_eq]

def initiateMatrixReducedForm(p,elimination_guys):
    k=int((p-1)/2)

    C_matrix = np.zeros((int(k/2),int(k/2)))
    C_matrix[0,]=1
    for i in range(1,int(k/2)):
        for j in range(0,int(k/2)):
            C_matrix[i,j] = math.cos(2*j*math.pi/k*elimination_guys[i-1])

    c=np.zeros(int(k/2))
    c[0]=k*k

    b_eq = np.zeros(int(k/2))
    b_eq[0] = 1/k

    return [C_matrix,c,b_eq]

def drawRandom(k):
    return random.sample(range(1,k),(int(k/2)))

def drawRandomSymmetric(k):
    final = []
    if int(k/2) % 2==0:
        aux = random.sample(range(1,int(k/2)-1),(int(k/4)-1))
        aux_1 = [k - x for x in aux]
        aux_2 = aux + aux_1
        aux_2.append(int(k/2))

    else:
        aux = random.sample(range(1,int(k/2)),(int(k/4)))
        aux_1 = [k - x for x in aux]
        aux_2 = aux + aux_1
    return aux_2

def initiateRandomMatrix(p):
    k=int((p-1)/2)
    elimination_guys = drawRandomSymmetric(k)
    C_matrix = np.zeros((int(k/2),k))
    C_matrix[0,]=1
    for i in range(1,int(k/2)):
        for j in range(0,k):
            C_matrix[i,j] = math.cos(2*j*math.pi/k*elimination_guys[i-1])

    c=np.zeros(k)
    c[0]=k*k

    b_eq = np.zeros(int(k/2))
    b_eq[0] = 1/k

    return [C_matrix,c,b_eq]

def initiateRandomComplementMatrix(p):
    k=int((p-1)/2)
    elimination_guys = drawRandomSymmetric(k)
    all_guys = np.array(range(1,k))
    complement_guys = np.sort(list(set(all_guys) - set(elimination_guys)))
    C_matrix = np.zeros((int(k/2)+1,k))
    C_matrix[0,]=1
    for i in range(1,int(k/2)+1):
        for j in range(0,k):
            C_matrix[i,j] = math.cos(2*j*math.pi/k*complement_guys[i-1])

    c=np.zeros(k)
    c[0]=k*k

    b_eq = np.zeros(int(k/2)+1)
    b_eq[0] = 1/k

    return [C_matrix,c,b_eq]

def initiateRandomMatrixPair(p, dual=False):
    k=int((p-1)/2)
    elimination_guys = drawRandomSymmetric(k)
    all_guys = np.array(range(1,k))
    complement_guys = np.sort(list(set(all_guys) - set(elimination_guys)))

    #complement guys = indices i of vertices such that {0,i} is in the complement of G
    #for p = 1(mod 4), we have |complement_guys|=1+|elimination_guys|
    #print ("complement : ", complement_guys, ", elim : ", elimination_guys)
    
    C_matrix = np.real(np.fft.fft(np.eye(k)))[ np.concatenate(([0],complement_guys)),:]
    C1_matrix = np.real(np.fft.fft(np.eye(k)))[ np.concatenate(([0],elimination_guys)),:]

    if not dual:
        c=np.zeros(k)
        c[0]=k*k
        b_eq = np.zeros(int(k/2)+1)
        b_eq[0] = 1/k
        c1=np.zeros(k)
        c1[0]=k*k

        b1_eq = np.zeros(int(k/2))
        b1_eq[0] = 1/k
    else:
        c=np.ones(k)/k
        b_eq = np.ones(int(k/2))*(-k)
        C_matrix = C_matrix[1:, :] # drop the 1s first row
        c1=np.ones(k)/k
        b1_eq = np.ones(int(k/2)-1)*(-k)
        C1_matrix = C1_matrix[1:,:] # drop the row of 1s.
        
        

    return [C1_matrix,c1,b1_eq,C_matrix,c,b_eq]

def Lovasz_PaleyGraphs():
    prime = [61,109,173,281,293,353,373,421,457,541,673,733,757,761,773,797,821,829,877,997,1009]
    #prime = [73]

    for p in prime:
        [primitive_root,quadratic_residue,elimination_guys,no_elimination_guys] = quadraticResidueAndMore(p)


        [A_eq,c,b_eq]=initiateMatrixExperimentSymmetric(p,elimination_guys)
        [A_eq1,c1,b_eq1]=initiateMatrixExperimentSymmetric(p,no_elimination_guys)

        result = scipy.optimize.linprog((-1)*c, A_eq = (-1)*A_eq, b_eq = (-1)*b_eq, bounds=(0,None) )
        result1 = scipy.optimize.linprog((-1)*c1, A_eq = (-1)*A_eq1, b_eq = (-1)*b_eq1, bounds=(0,None) )
        """print((len(np.nonzero(result.x)[0])+len(np.nonzero(result1.x)[0]))*2)"""
        print(len(np.nonzero(result.x)[0]))
        print(len(np.nonzero(result1.x)[0]))
        print(np.intersect1d(np.nonzero(result.x),np.nonzero(result1.x)))
        #print(len(np.nonzero(result1.x)[0]))
        #print("---------------")
        """print(result.x,result.fun)
        print(np.nonzero(result.x))
        print(np.nonzero(result1.x))
        print(elimination_guys)"""
        print(result.fun-result1.fun)
        """print(result.x)
        print(result1.x)
        print(np.intersect1d(np.nonzero(result.x),np.nonzero(result1.x)))"""

    """
    x = np.multiply(result.x,result1.x)
    print(x*int((p-1)/2)*int((p-1)/2*(p-1)/2)*int((p-1)/2))
    print(result.x)
    print(scipy.linalg.null_space(A_eq))"""

    """print(result.x,result.fun)
    print(np.nonzero(result.x))
    print(elimination_guys)
    print("-------------------------------------------")
    print(result1.x,result1.fun)
    print(np.nonzero(result1.x))
    print(no_elimination_guys)"""


def primes_list(n, only_equal1mod4 = False):
    odds = range(3, n+1, 2)
    sieve = set(sum([list(range(q*q, n+1, q+q)) for q in odds], []))
    if not only_equal1mod4:
        return [2] + [p for p in odds if p not in sieve]
    if only_equal1mod4:
        return [p for p in odds if (p not in sieve and p%4==1)]



def Lovasz_RandomGraphs(dual=False):
    #primes = [61,109,173,281,293,353,373,421,457,541,673,733,757, 1997]
    primes = [primes_list(2000, only_equal1mod4=True)[-1]]
    #primes = [13]
    variances_reg = []
    variances_comp = []
    variances_diff = []
    mean_val_result = []
    for prime in primes:
        rm_result = []
        rm_comp_result = []
        difference = []
        fft_vec = np.array([])
        nb_runs = 30
        for i in range(0,nb_runs):
            [A_eq,c,b_eq,A1_eq,c1,b1_eq]=initiateRandomMatrixPair(prime, dual)
            sign = -1 if not dual else 1 # to feed into linprog (min, max..)
            #print ("A1_eq : ", A1_eq.shape, " c1.shape : ", c1.shape, ", b1_eq.shape : ", b1_eq.shape)
            #result = scipy.optimize.linprog(sign*c, A_eq = A_eq, b_eq = b_eq, bounds=(0,None), method='highs-ipm' )
            result1= scipy.optimize.linprog(sign*c1, A_eq = A1_eq, b_eq = b1_eq, bounds=(0,None), method='highs-ipm' )
            
            result = result1 # DO NOT KEEP : THIS IS JUST TO SPEED UP PROGRAM, DO 1 LP AND NOT 2 

            if dual and isinstance(result.fun, float):
                result1.fun += 1
            elif not isinstance(result.fun, float):
                print ("result is not a float...")
            res = result if not dual else result1
            #result1 = scipy.optimize.linprog(sign*c1, A_eq = (-1)*A1_eq, b_eq = (-1)*b1_eq, bounds=(0,None) )
            
            #if isinstance(result.fun, float) & isinstance(result1.fun, float):
                #rm_result.append(result.fun)
                #rm_comp_result.append(result1.fun)
                #difference.append(float(result.fun)-float(result1.fun))
            #print ("value : ", -result.fun)

            #analyse_value_lp(result, dual)

            print ("on run : ", i)
            k = c1.shape[0] # size of graph
            epsilon = 1e-7
            real_fft_x = np.real(np.fft.fft(res.x))

            if not dual :
                real_fft_x = real_fft_x[np.abs(real_fft_x)>epsilon]; real_fft_x = real_fft_x[1:]
            else:
                real_fft_x *= 1/k ; real_fft_x = real_fft_x[np.abs(real_fft_x+1)>epsilon]; real_fft_x = real_fft_x[1:]
            fft_vec = np.concatenate((fft_vec, real_fft_x), axis=None)



            if isinstance(result.fun, float):
                rm_result.append(sign*res.fun)

        rm_result = np.sort(rm_result)
        #rm_comp_result = np.sort(rm_comp_result)
        #difference = np.sort(difference)
        variances_reg.append(np.var(rm_result))
        mean_val_result.append(np.mean(rm_result))

        counts, bins = np.histogram(fft_vec, bins=50)
        #counts, bins = np.histogram(rm_result, bins=50)
        pp.stairs(counts, bins)
        pp.show()
        #variances_comp.append(np.var(rm_comp_result))
        #variances_diff.append(np.var(difference))
        print("\n\n\n")
        print("The following results are for p = ",prime)

        print("Mean of the rm_result is: ", np.mean(rm_result))
        print("Variance of the rm_result is: ", np.var(rm_result))

        #print("Mean of the rm_comp_result is: ", np.mean(rm_comp_result))
        #print("Variance of the rm_comp_result is: ", np.var(rm_comp_result))

        #rint("Mean of the difference: ", np.mean(difference))
        #print("Variance of the difference: ", np.var(difference))
    #pp.plot(primes, variances_reg,"ro",label="Variance (p-5)/4 edges")
    #pp.plot(primes ,variances_comp,"bo",label="Variance (p-1)/4 edges")
    #pp.plot(primes,variances_diff,"go",label="Variance of difference")
    
    #pp.plot(primes, mean_val_result, color="red", label="mean value for theta")
    #pp.plot(primes, np.sqrt((np.array(primes)-1)/2), color="green", label="\sqrt( (p-1)/2)")
    #pp.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                #mode="expand", borderaxespad=0, ncol=3)
    #pp.show()

def analyse_value_lp(result, dual=False):
    #print ("best vec : ", result.x)

    
    k = result.x.shape[0] # size of the graph

    if not dual: 
        print ("x.sum * k = ", result.x.sum()*k)
        print ("len (x) : ", result.x.shape)
        print ("# constraints : ", result.con.shape[0])
        real_fft_x = np.real(np.fft.fft(result.x))
        #print ("(real) fft of x : ", real_fft_x)
        counts, bins = np.histogram(real_fft_x, bins=k)
        pp.stairs(counts, bins)
    else:
        print ("result.x : ", result.x)
        print ("1/n fft (result.x) : ", 1/k*np.fft.fft(result.x))
    #pp.show()


def Lovasz_random_bound_no_lp():
    primes = primes_list(400, only_equal1mod4=True)
    #primes = [13]
    record_min_eig = np.empty(shape=(len(primes),3))
    nb_runs = 10
    for (idx, prime) in zip(range(len(primes)), primes):
        count = np.array([0.,0., 0.]).reshape((1,3))
        print ("doing prime ", prime)
        for run in range(nb_runs):
            k = int((prime-1)/2)
            size = int(np.ceil((k-1)/2)) 
            bernoulli_vec = -2*np.random.binomial(1, 0.5, size=size)+1
            bernoulli_vec = np.concatenate(([0], bernoulli_vec, np.flip(bernoulli_vec[:-1])), axis=None)
            ber_fft = np.real(np.fft.fft(bernoulli_vec)) # np.real() is not necessary by symmetry (but it avoids a warning)
            max_ber_fft = np.max(np.abs(ber_fft))

            # matrix formulation : any better ?
            matrixA, matrixB = np.empty(shape=(k,k)), np.empty(shape=(k,k))
            bernoulli_vecA = (1-0.5*(bernoulli_vec + 1)) # here, =0 if not in graph
            ber_vec_B = 1-bernoulli_vecA; ber_vec_B[0]=1 # here, =0 if IN graph
            for row_idx in range(k):
                gausA = np.random.normal(loc=-1, scale=1/np.sqrt(k), size=bernoulli_vecA.shape[0])
                gausB = gausA +1 # centered gaussians
                rowA = bernoulli_vecA*gausA + (1-bernoulli_vecA)
                rowA = np.roll(rowA, row_idx)
                matrixA[row_idx] = rowA

                rowB = ber_vec_B*gausB
                rowB = np.roll(rowB, row_idx)
                matrixB[row_idx] = rowB

            matrixA = 0.5*(matrixA + matrixA.T) # symmetrise
            matrixB = 0.5*(matrixB + matrixB.T)

            eigenvalsA = np.linalg.eigvalsh(matrixA)
            eigenvalsB = np.linalg.eigvalsh(matrixB)
            valB = 1-np.max(eigenvalsB)/np.min(eigenvalsB)
            

            count += np.array([max_ber_fft, np.max(eigenvalsA), 1-np.max(eigenvalsB)/np.min(eigenvalsB)])
        record_min_eig[idx] = count/nb_runs     
        #print (record_min_eig)
    pp.plot(primes, record_min_eig[:,0], color = 'blue')
    pp.plot(primes, record_min_eig[:,1], color='red')
    #pp.plot(primes, record_min_eig[:,2], color='green') # this is a LOWER BOUND on theta
    pp.plot(primes, np.sqrt(primes))
    pp.plot(primes, np.sqrt(np.log(primes)*primes))
    pp.show()

def is_lp_min_affected_by_asymmetry():
    primes = primes_list(1000, True)
    #primes = [13]
    nb_runs, res_vec, res_vec1 = 10, [], []
    for p in primes:
        count = 0
        count1 = 0
        print ("on prime ", p)
        for run in range(nb_runs):
            k=int((p-1)/2)
            elimination_guys = drawRandomSymmetric(k)
            all_guys = np.array(range(1,k))
            complement_guys = np.sort(list(set(all_guys) - set(elimination_guys)))    
            A = np.real(np.fft.fft(np.eye(k)))[ :,complement_guys]
            A = np.append(A, np.ones(k).reshape(k,1), axis=1)
            constrained_vec = np.zeros(k); constrained_vec[elimination_guys]=-1
            b = np.real(np.fft.fft(constrained_vec))
            c = np.zeros(len(complement_guys)+1); c[-1]=1
            A_ub = -A # we ensure -Ax-t <= b
            A1 = A ; A1[:,-1]*= -1 # we also ensure Ax-t <= -b
            b1 = -b

            A_ub_symm = np.concatenate((A_ub, A1), axis=0)
            b_ub_symm = np.concatenate((b, b1), axis=0)
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b, method='highs-ipm' )
            res_symm = scipy.optimize.linprog(c, A_ub=A_ub_symm, b_ub=b_ub_symm, method='highs-ipm')
            count += res.fun ; count1 += res_symm.fun
        res_vec.append(count/nb_runs); res_vec1.append(count1/nb_runs)
    k_vec = (np.array(primes)-1)/2
    pp.plot(k_vec, res_vec, color='red')
    pp.plot(k_vec, res_vec1, color='green')
    pp.plot(k_vec, np.sqrt(k_vec))
    pp.plot(k_vec, np.sqrt(np.log(k_vec)*k_vec))
    pp.show()

def is_lp_affected_by_boundedness_constraints():
    primes = primes_list(1000, True)[3:]
    #primes = [13]
    nb_runs, res_vec, res_vec1 = 10, [], []
    for p in primes:
        count = 0
        count1 = 0
        print ("on prime ", p)
        for run in range(nb_runs):
            k=int((p-1)/2)
            elimination_guys = drawRandomSymmetric(k)
            all_guys = np.array(range(1,k))
            complement_guys = np.sort(list(set(all_guys) - set(elimination_guys)))    
            A = np.real(np.fft.fft(np.eye(k)))[ :,complement_guys]
            A = np.append(A, np.ones(k).reshape(k,1), axis=1)
            constrained_vec = np.zeros(k); constrained_vec[elimination_guys]=-1
            b = np.real(np.fft.fft(constrained_vec))
            c = np.zeros(len(complement_guys)+1); c[-1]=1
            A_ub = -A # we ensure -Ax-t <= b
            
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b, bounds=(None,None), method='highs-ipm' )

            bound_size = 1
            bound_vector_low, bound_vector_high = np.ones_like(c)-bound_size, np.ones_like(c)+bound_size; bound_vector_low[-1]=-k; bound_vector_high[-1]=k
            bounds = list(zip(bound_vector_low, bound_vector_high))
            #print ("bound_vector_low : ", bound_vector_low, ", bound_vector_high : ", bound_vector_high)
            res_bd = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b,bounds=bounds,  method='highs-ipm')
            #print ("res : ", res.fun, ", res_bd : ", res_bd.fun)
            #print ("res.x : ", res.x, ", res_bd.x : ", res_bd.x)
            #print ("b : ", b)
            count += res.fun ; count1 += res_bd.fun
        res_vec.append(count/nb_runs); res_vec1.append(count1/nb_runs)
    k_vec = (np.array(primes)-1)/2
    pp.plot(k_vec, res_vec, color='red', label="theta number")
    pp.plot(k_vec, res_vec1, color='green', label= "theta number constraining all variables be in [0,2]")
    pp.plot(k_vec, np.sqrt(k_vec))
    #pp.plot(k_vec, np.sqrt(np.log(k_vec)*k_vec))
    pp.legend()
    pp.show()

def glutton_method_for_lp():
    primes = primes_list(500, True)
    #primes = [13]
    res_vec,check_vec,  nb_runs =[], [],10
    for p in primes:
        count = [0,0]
        k=int((p-1)/2)
        print ("on prime ", p, ", k = ", k)
        for run in range(nb_runs):
            elimination_guys = drawRandomSymmetric(k); all_guys = np.array(range(1,k)) # nonfree coor
            complement_guys = np.sort(list(set(all_guys) - set(elimination_guys))) # free coordinates
            
            x = np.ones(k); x[elimination_guys] = -1 ; x[0]=0 ; v = np.real(np.fft.fft(x)) ; check_val = np.max(np.abs(v))
            current_res = None; current_x = x; current_v = v
            for (idx, i) in zip(range(len(complement_guys)), complement_guys):
                e_i = np.zeros(k); e_i[i] = 1; w_i = np.real(np.fft.fft(e_i))
                cost = np.array([1, 0]); A_ub = np.array([-np.ones(2*k), np.concatenate((w_i,-w_i))]).T; b_ub = np.concatenate((-current_v,current_v))
                current_res = scipy.optimize.linprog(cost, A_ub=A_ub, b_ub=b_ub, method='highs-ipm')
                current_x[i] = current_res.x[1]; current_v += current_res.x[1]*w_i
            count[0] += current_res.fun ; count[1] += check_val   
            
        res_vec.append(count[0]/nb_runs)
        check_vec.append(count[1]/nb_runs)
    k_vec = (np.array(primes)-1)/2
    pp.plot(k_vec, res_vec, color='red')
    pp.plot(k_vec, np.sqrt(k_vec), color='green')
    pp.plot(k_vec, np.sqrt(np.log(k_vec)*k_vec), color='blue')
    pp.plot(k_vec, check_vec, color='black')
    pp.show()

def initiateLP_oneconstraintdifference(prime):
            p = prime
            k = int((p-1)/2)
            elimination_guys = drawRandomSymmetric(k)
            all_guys = np.array(range(1,k))
            complement_guys = np.sort(list(set(all_guys) - set(elimination_guys)))

            constraint_to_add = [complement_guys[0], k-complement_guys[0]]
            rows = np.concatenate(([0],complement_guys))
            C_matrix = np.real(np.fft.fft(np.eye(k)))[ rows,:]
            rows_plus1 = np.concatenate(([0], constraint_to_add,complement_guys))
            Cplus1_matrix = np.real(np.fft.fft(np.eye(k)))[ rows_plus1 ,:]

            c=np.zeros(k)
            c[0]=-k*k
            b_eq = np.zeros(int(k/2)+1)
            b_eq[0] = 1/k
            b_eq_plus1 = np.zeros(int(k/2)+3)
            b_eq_plus1[0] = 1/k

            return [C_matrix, Cplus1_matrix,c,b_eq, b_eq_plus1]
        

def is_lp_lipschitz():

    prime = primes_list(1000, only_equal1mod4=True)[-1]
    print ("prime : ", prime)
    k = int((prime-1)/2)

    def compare_1_constraint_difference():
    
        [A_eq, Aplus1_eq, c, b_eq, b_eq_plus1] = initiateLP_oneconstraintdifference(prime)
        print ("received : ", A_eq.shape, c.shape, b_eq.shape)
        print ("received +1 : ", Aplus1_eq.shape, b_eq_plus1.shape)
        res = scipy.optimize.linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=(0,None), method='highs-ipm' )
        res_plus1 = scipy.optimize.linprog(c, A_eq = Aplus1_eq, b_eq = b_eq_plus1, bounds=(0,None), method='highs-ipm' )
        print ("value of the LP : ", -res.fun)
        print ("value of the LP+1 : ", -res_plus1.fun)
        print ("norms of the vectors : ", np.linalg.norm(res.x), np.linalg.norm(res_plus1.x))
        print ("norm of the difference : ", np.linalg.norm(res.x-res_plus1.x))

        print ("dominance of 0 : ", res.x[0]/np.linalg.norm(res.x))
        print ("dominance of 0+1 : ", res_plus1.x[0]/np.linalg.norm(res_plus1.x))
        print ("cos(angle between the vectors) : ", np.dot(res.x, res_plus1.x)/(np.linalg.norm(res.x)*np.linalg.norm(res_plus1.x)))
        print ("cos without first coordinate : ", np.dot(res.x[1:], res_plus1.x[1:])/(np.linalg.norm(res.x[1:])*np.linalg.norm(res_plus1.x[1:])))
        print ("nb tight constraints : ", np.sum(np.abs(res.x)<1e-8))
        print ("nb common tight constraints : ", np.sum((np.abs(res.x)<1e-10) * (np.abs(res_plus1.x)<1e-10)))

    def trace_value_as_function_of_nb_constraints():
        nb_runs=1
        res_vec, norms_vec, norms_diff_vec, dominance_of_zero_vec = [], [], [], []
        constraint_vec = np.array(range(1, int(k/2))); np.random.shuffle(constraint_vec)
        constraint_vec_symm = k-constraint_vec
        C_matrix= np.real(np.fft.fft(np.eye(k)[:,0])).reshape(1,k)
        print ("initially C has size ", C_matrix.shape)
        b_eq = np.array([1/k])
        c=np.zeros(k); c[0]=-k*k
        for (idx, i,j) in zip(range (constraint_vec.shape[0]),constraint_vec, constraint_vec_symm):
            print ("on constraint ", idx)
            C_matrix = np.concatenate((C_matrix,\
                 np.real(np.fft.fft(np.eye(k)[:,i])).reshape(1,k),\
                 np.real(np.fft.fft(np.eye(k)[:,j])).reshape(1,k)), axis=0)
            print ("C_matrix has shape ", C_matrix.shape)
            b_eq = np.concatenate((b_eq, np.array([0,0])), axis=0)
            res = scipy.optimize.linprog(c, A_eq = C_matrix, b_eq = b_eq, bounds=(0,None), method='highs-ipm' )
            res_vec.append(-res.fun)
            norms_vec.append(np.linalg.norm(res.x))
            if idx > 0:
                norms_diff_vec.append(np.linalg.norm(res.x-res_old.x))
            dominance_of_zero_vec.append(res.x[0]/np.linalg.norm(res.x))
            res_old = res
        fig, ax1 = pp.subplots()
        ax1.plot(np.array(range(1,int (k/2))), res_vec, label="value of the LP (#constraints)")
        #ax1.plot(np.array(range(1,int (k/2))), dominance_of_zero_vec, label="dominance of 0 (#constraints)", color='blue')
        ax1.plot(np.array(range(1,int (k/2))), np.ones(int(k/2)-1)*np.sqrt(k), label="sqrt(k)", color='green')
        
        # plot point half constraints
        ax1.plot(int(k/4), res_vec[int(k/4)], marker='o', markersize=3, color="red")
        
        #ax2 = ax1.twinx()
        #ax2.plot(np.array(range(1,int (k/2))), norms_vec, label="norm of the vector (#constraints)", color='red')
        #ax2.plot(np.array(range(2,int (k/2))), norms_diff_vec, label="norm of the difference", color='green')
        ax1.legend(); #ax2.legend()
        pp.show()
    
    def how_many_plus1_constraints():
        nb_runs=1
        res_vec, max_val_x, norm_vec = [], [], []
        constraint_vec = np.array(range(1, int(k/2))); np.random.shuffle(constraint_vec)
        constraint_vec_symm = k-constraint_vec
        neg_constraint_vec, neg_constraint_vec_symm = constraint_vec[:int(k/4)], constraint_vec_symm[:int(k/4)]
        pos_constraint_vec, pos_constraint_vec_symm = constraint_vec[int(k/4):], constraint_vec_symm[int(k/4):]
        one_hot_neg_constraint = np.zeros(k); one_hot_neg_constraint[neg_constraint_vec] = -1 ; one_hot_neg_constraint[neg_constraint_vec_symm] = -1
        v_neg =np.real( np.fft.fft(one_hot_neg_constraint))
        
        
        for (idx, pos_cstr) in enumerate(pos_constraint_vec):
            print ("on constraint ", idx+1, " / ", len(pos_constraint_vec))
            one_hot_pos_constraint = np.zeros(k); one_hot_pos_constraint[pos_constraint_vec[:idx]] = 1 ; one_hot_pos_constraint[pos_constraint_vec_symm[:idx]] = 1
            v_pos = np.real(np.fft.fft(one_hot_pos_constraint))
            v = v_neg + v_pos
            cols = np.concatenate(([0],pos_constraint_vec[idx:], pos_constraint_vec_symm[idx:]))
            #print ("taking ", cols.shape[0], " columns")
            C_matrix = np.real(np.fft.fft(np.eye(k)))[ :,cols]
            c = np.zeros(cols.shape[0]); c[0]=1
            res = scipy.optimize.linprog(c, A_ub = -C_matrix, b_ub = v, method='highs-ipm' )
            res_vec.append(res.fun+1)
            max_val_x.append(np.max(res.x[1:]))
            norm_vec.append(np.linalg.norm(res.x[:]))
            # print how many nonzero coordinates 
            print ("nb coordinates >0.5*res.fun : ", np.sum(np.abs(res.x)>0.5*res.fun),"/ " ,c.shape[0])
            
            close_feasable_point = res.x ; close_feasable_point[1]=-1
            fft_close_feasable_point = np.real(np.fft.fft(close_feasable_point))
            new_cost = res.fun + 1 - np.min(fft_close_feasable_point)
            print ("res.fun : ", res.fun, ", new_cost : ", new_cost)

        fig, ax1 = pp.subplots()
        ax1.plot(np.array(range(len(res_vec))), res_vec, label="value of the LP (# positive constraints added)")
        ax1.plot(np.array(range(1,int (k/4)+1)), np.ones(int(k/4))*np.sqrt(k), label="sqrt(k)", color='green')
        ax1.plot(np.array(range(1,int (k/4)+1)), np.ones(int(k/4))*2*np.sqrt(k), label="2*sqrt(k)", color='green')
        # plot max value vector : 
        #ax2 = ax1.twinx()
        #ax2.plot(np.array(range(len(max_val_x))), max_val_x, label="max value of the vector (# positive constraints added)")
        # plot norm of the vector
        ax1.plot(np.array(range(len(norm_vec))), norm_vec, label="norm of the vector (# positive constraints added)", color='red')
        
        ax1.legend();# ax2.legend()
        pp.show()
            
    def trace_dominance_of_zero_with_half_constraints():
        nb_runs=10
        dominance_of_zero_vec, val_of_zero_vec, maxval_without_zero_vec, prop_large_coor = [], [], [], []
        
        primes = primes_list(1000, True)[1:]
        for p in primes:
            print ("on prime ", p)
            k = int((p-1)/2)
            count_dominanceof0, count_valof0, count_maxval, count_large_coor= 0, 0, 0,0
            for run in range(nb_runs):
                [_,_,_,A_eq,c,b_eq]=initiateRandomMatrixPair(p,dual=False)
                res = scipy.optimize.linprog(-c, A_eq = A_eq, b_eq = b_eq, bounds=(0,None), method='highs-ipm' )
                count_dominanceof0 += res.x[0]/np.linalg.norm(res.x)
                count_valof0 += -res.fun
                count_maxval += np.max(res.x[1:]*(k**2))
                count_large_coor += np.sum(res.x>2/(k**2))/k
            val_of_zero_vec.append(count_valof0/nb_runs)
            maxval_without_zero_vec.append(count_maxval/nb_runs)
            dominance_of_zero_vec.append(count_dominanceof0/nb_runs)
            prop_large_coor.append(count_large_coor/nb_runs)

        fig, ax1 = pp.subplots()
        #ax1.plot(np.array(range(1,int (k/2))), res_vec, label="value of the LP (#constraints)")
        #ax1.plot(primes, dominance_of_zero_vec, label="dominance of 0 (#constraints)", color='blue')
        #ax1.plot(primes, val_of_zero_vec, label="value of the 0 coor (prime)", color='red')
        #ax1.plot(primes, maxval_without_zero_vec, label="max value of the vector without 0 (prime)", color='blue')
        #ax1.plot(primes, np.sqrt((np.array(primes)-1)/2), label="sqrt(k)", color='green')
        ax1.plot(primes, prop_large_coor, label="proportion of large coordinates", color='blue')
        #ax2 = ax1.twinx()
        #ax2.plot(np.array(range(1,int (k/2))), norms_vec, label="norm of the vector (#constraints)", color='red')
        #ax2.plot(np.array(range(2,int (k/2))), norms_diff_vec, label="norm of the difference", color='green')
        ax1.legend(); #ax2.legend()
        pp.show()
    

def finding_a_good_correction_vector():
    
    def correction_vector (n, coor, constraint_vec=None):
            b = np.zeros(shape= (n,))
            h = coor     
            '''for k in range(n):
                b[k] = (k!=1) * np.cos(2*math.pi*k*h/n) 
            g = np.real(np.fft.ifft(b))
            if constraint_vec is not None:
                g[constraint_vec] = 0
            b = np.real(np.fft.fft(g))

            max_g = np.max(g)
            g /= max_g
            b /= max_g'''

            best_so_far, best_b, best_g, best_alpha = np.inf, None, None, None
            for run in range(1,200):
                alpha = run
                for k in range(n):
                    b[k] = (k!=1) * np.cos(2*math.pi*alpha*(k-h)/n)
                if constraint_vec is not None:
                    b[constraint_vec] = 0
                g = np.real(np.fft.fft(b))
                if np.max(g) < best_so_far:
                    best_so_far = np.max(g)
                    best_b, best_g, best_alpha = b, g, alpha
            print ("g is the fft of b : ", np.allclose(best_g, np.real(np.fft.fft(best_b))))
            print ("found best value for max g : ", best_so_far)
            print ("at frequency : ", best_alpha)
            print ("b verifies the condition : b[coor] = ", best_b[coor])
            
            
            return best_b,best_g

    
    
    def test_dirac_delta_perturbation():
        n = 281     
        b= correction_vector(n, 100)
        # now introduce the correction that half the coordinates must be zero:
        b *= np.random.binomial(1, 0.5, n)
        g = np.real(np.fft.fft(b))

        pp.plot(np.array(range(n))/n, g, color='blue')
        pp.show()
    
    def make_dirac_correction():
        prime = primes_list(1000, only_equal1mod4=True)[-1]
        print ("prime : ", prime)
        k = int((prime-1)/2)

        constraint_vec = np.array(range(1, int(k/2))); np.random.shuffle(constraint_vec)
        constraint_vec_symm = k-constraint_vec
        neg_constraint_vec, neg_constraint_vec_symm = constraint_vec[:int(k/4)], constraint_vec_symm[:int(k/4)]
        all_neg_constr = np.concatenate((neg_constraint_vec, neg_constraint_vec_symm), axis=None)
        free_vars, free_vars_symm = constraint_vec[int(k/4):], constraint_vec_symm[int(k/4):]
        one_hot_neg_constraint = np.zeros(k); one_hot_neg_constraint[neg_constraint_vec] = -1 ; one_hot_neg_constraint[neg_constraint_vec_symm] = -1
        v_neg =np.real( np.fft.fft(one_hot_neg_constraint))
        v = v_neg

        cols = np.concatenate(([0],free_vars, free_vars_symm))
        print ("taking ", cols.shape[0], " columns")
        C_matrix = np.real(np.fft.fft(np.eye(k)))[ :,cols]
        c = np.zeros(cols.shape[0]); c[0]=1
        res = scipy.optimize.linprog(c, A_ub = -C_matrix, b_ub = v, bounds=None, method='highs-ipm' )

        print ("cost is ", res.fun+1)
        print ("sqrt k : ", np.sqrt(k))

        # now add a new constraint with dirac correction
            # draw a random index : 
        idx = np.random.randint(1, free_vars.shape[0])
        l = free_vars[idx]
        print ("want to correct coordinate ", l)
        print ("require to correct coordinate with value : ", res.x[l])
        #pp.plot(range(res.x.shape[0]), res.x, color='blue')
        #pp.show()
        b,g = correction_vector(k, l , constraint_vec=all_neg_constr)
        #g = g[:cols.shape[0]] # only keep the nonsymmetric part
        pp.plot(range (int(k)), g, color='green')
        pp.plot(range(int(k)), b, color='blue')

        # the difference in cost is : 
        print ("received b such that max(fft(b)) = ", np.max(np.real(np.fft.fft(b))) )
        delta_cost = (res.x[l]+1)*np.max(np.real(np.fft.fft(b)))
        print ("delta cost : ", delta_cost)

        
        
        #pp.plot(range(2*cols.shape[0]),np.dot(C_matrix , res.x) +v, color='red')
        pp.show()

        
    #test_dirac_delta_perturbation()
    make_dirac_correction()
    


def initiateSymCirc(k):
    aux = [0]
    coordinates = drawRandomSymmetric(k)+aux
    Matrix = np.zeros((k,k))
    for i in range(0,k):
        for j in range(0,k):
            if (k-i+j)%k in coordinates:
                Matrix[i,j]=1
            else:
                Matrix[i,j]=-1
    return Matrix

if __name__ == "__main__":
    is_lp_affected_by_boundedness_constraints()
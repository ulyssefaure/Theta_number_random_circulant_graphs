import numpy as np

def random_orthonormal_basis(n):
    # Generate a random matrix
    random_matrix = np.random.normal(size=(n, n))
    
    # Perform QR decomposition to obtain an orthonormal basis
    q, _ = np.linalg.qr(random_matrix)
    
    return q

def affine_max_norm_min(dim_space, dim_subspace):
    U = random_orthonormal_basis(dim_space)
    #print ("Orthogonality of the matrix is verified : ", np.linalg.norm(np.dot(U.T, U)-np.eye(dim_space)) < 1e-10)
    norm_v = 1
    v = U[:,0] * norm_v
    W = U[:,1:dim_subspace+1]

    #print ("W has shape ", W.shape)

    # find the point in v+span(W) that has the minimum maximum norm
    # ie. find min t s.t. -Wx-t <= v and Wx-t <= -v

    # Formulate the problem as a linear program
    from scipy.optimize import linprog
    c = np.zeros(dim_subspace + 1)
    c[0] = 1
    A_ub_withoutone = np.concatenate( (-W, W), axis=0)
    A_ub = np.concatenate( (-np.ones((2*dim_space, 1)), A_ub_withoutone), axis=1)
    b_ub = np.concatenate((v, -v), axis=0)

    #print ("A_ub has shape ", A_ub.shape)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,A_eq=None, b_eq=None,  bounds=None, method='highs')
    best_point = np.dot(W,res.x[1:])+v
    best_point_on_unit_cube = best_point / np.max(np.abs(best_point))
    
    print ("for a vector of norm ", norm_v, " the minimum maximum norm is ", res.fun)
    #print ("best point : ", best_point)
    #print ("best point on the unit cube is : ", best_point_on_unit_cube)
    print ("orthogonality of sol of LP is verified : ", np.dot(best_point - v, v) < 1e-10)
    # hand calculation
    k = np.linalg.norm(v)**2 / np.linalg.norm(v, ord=1)
    hand_vec = k * np.sign(v)
    hand_x = np.linalg.lstsq(W, hand_vec-v, rcond=None)[0]
    #print ("able to invert : ", np.all(np.abs(np.dot(W, hand_x) - (hand_vec-v)) < 1e-10))
    hand_x_withone = np.concatenate((np.array([k+1e-8]), hand_x), axis=0)

    #print ("orthogonality of hand calculation is verified : ", np.dot(hand_vec - v, v) < 1e-10) 
    #print ("a hand calculation yields best cost to be ", k)

    print ("we find that hand_x_withone is a feasable point : ", np.all(np.dot(A_ub, hand_x_withone) <= b_ub))
    print ("this point has cost ", np.dot(c, hand_x_withone))
    
    return res.fun

# Example usage
n=10
k=9
print (affine_max_norm_min(n, k))


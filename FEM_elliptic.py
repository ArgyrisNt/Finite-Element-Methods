import numpy as np
from skfem import *
from skfem import accum
from skfem import LinearForm
from skfem.helpers import grad
from skfem.models.poisson import laplace, unit_load
from skfem.visuals.matplotlib import draw_mesh2d, plot, show, draw


node = np.array([[-1.0,-1.0],
                 [-0.5,-1.0],
                 [0.0, -1.0],
                 [0.0, -0.5],
                 [0.0,  0.0],
                 [0.5,  0.0],
                 [1.0,  0.0],
                 [1.0,  0.5],
                 [1.0,  1.0],
                 [0.5,  1.0],
                 [0.0,  1.0],
                 [-0.5, 1.0],
                 [-1.0, 1.0],
                 [-1.0, 0.5],
                 [-1.0, 0.0],
                 [-1.0,-0.5],
                 [-0.5,-0.5],
                 [-0.5, 0.0],
                 [-0.5, 0.5],
                 [0.0,  0.5],
                 [0.5,  0.5]])
    

connectivity = np.array([[ 1,15, 0],
                         [15, 1,16],
                         [ 2,16, 1],
                         [16, 2, 3],
                         [16,14,15],
                         [14,16,17],
                         [ 3,17,16],
                         [17, 3, 4],
                         [18,14,17],
                         [14,18,13],
                         [19,17, 4],
                         [17,19,18],
                         [11,13,18],
                         [13,11,12],
                         [10,18,19],
                         [18,10,11],
                         [ 5,19, 4],
                         [19, 5,20],
                         [ 6,20, 5],
                         [20, 6, 7],
                         [20,10,19],
                         [10,20,9],
                         [ 7,9,20],
                         [9, 7, 8]])

# Assembling local stiffness matrix
def localstiff(p):
    Alocal = np.zeros((3,3));
    # Define matrix B, which exists in the map x=F(ξ)=Bξ+x(3)
    B = np.matrix([p[0,:]-p[2,:],p[1,:]-p[2,:]]);
    # Find the transpose of matrix B
    BT = np.transpose(B);
    # Find the inverse of the transpose of matrix B
    BTI = np.linalg.inv(BT);
    # Store gradients of basis function of the reference triangle
    G = np.matrix([[1,0,-1],[0,1,-1]]);
    area = 0.5*abs(np.linalg.det(B));    
    # Calculate the local stiffness matrix
    for i in range(0,3):
        for j in range(0,3):
            BG = BTI*G[:,j];
            BGT = np.transpose(BTI*G[:,i]);
            Alocal[i,j] = area*BGT*BG;
    return Alocal

# Assembling stiffness matrix.
def stiff(node,connectivity):
    N = np.size(node,0);
    NT = np.size(connectivity,0);
    A = np.zeros((N,N));
    # Call another funtion, in order to calculate the local stiffness matrix
    for k in range(0,NT):
        Alocal = localstiff(node[connectivity[k,:],:]);
        # Calculate the stiffness matrix
        for i in range(0,3):
            for j in range(0,3):
                A[connectivity[k,i],connectivity[k,j]] = A[connectivity[k,i],connectivity[k,j]] + Alocal[i,j];
    return A

# Find right hand side F, using 3 points quadrature rule
N = np.size(node,0);
NT = np.size(connectivity,0);
ve = np.zeros((NT,2,3));
F = np.zeros((N,1));
ve[:,:,2] = node[connectivity[:,1],:]-node[connectivity[:,0],:];
ve[:,:,0] = node[connectivity[:,2],:]-node[connectivity[:,1],:];
ve[:,:,1] = node[connectivity[:,0],:]-node[connectivity[:,2],:];
areaall = 0.5*abs(-ve[:,0,2]*ve[:,1,1]+ve[:,1,2]*ve[:,0,1]);
mid1 = (node[connectivity[:,1],:]+node[connectivity[:,2],:])/2;
mid2 = (node[connectivity[:,2],:]+node[connectivity[:,0],:])/2;
mid3 = (node[connectivity[:,0],:]+node[connectivity[:,1],:])/2;
bt1 = areaall*10/3;
bt = np.transpose(np.array([bt1,bt1,bt1]));
F = accum.accum(connectivity,bt)

# Draw the triangular mesh
A = stiff(node,connectivity);
m = MeshTri(np.transpose(node), np.transpose(connectivity))
draw_mesh2d(m)

# Solve Ax=F
x = solve(*condense(A, F, I=m.interior_nodes()))
plot(m, x)
show()

############################### Refinement ###################################

e = ElementTriP1()  
basis = InteriorBasis(m, e)

@LinearForm
def load(v, w):
    return 10 * v

# Interior residual at element w
@Functional
def res(w):
    # Diameter of triangle
    h = w.h
    ηT = h ** 2 * 10 ** 2
    return ηT

# Calculate the jump of edge between elements u1,u2
@Functional
def jump(w):
    # Diameter of triangle
    h = w.h
    # Normal vector on edge between elements u1,u2
    n = w.n
    ηE = h * ((grad(w['u1'])[0] - grad(w['u2'])[0]) * n[0] +(grad(w['u1'])[1] - grad(w['u2'])[1]) * n[1]) ** 2
    return ηE
        
for itr in range(2):
    if itr > 0:
        ηT = res.elemental(basis, w=basis.interpolate(u))
        fbasis = [FacetBasis(m, e, side=i) for i in [0, 1]]   
        w = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}    
        ηE = jump.elemental(fbasis[0], **w)    
        tmp = np.zeros(m.facets.shape[1])
        np.add.at(tmp, fbasis[0].find, ηE)
        ηE = np.sum(0.5*tmp[m.t2f], axis=0)    
        
        # Final error value at each element
        η = ηT+ηE
        
        # Do the refinement
        m.refine(adaptive_theta(η))     
    
    # Define the new stiffness matrix and right hand side    
    basis = InteriorBasis(m, e)    
    K = asm(laplace, basis)
    f = asm(load, basis)    
    I = m.interior_nodes()
    
    # Solve the new system Ku=f
    u = solve(*condense(K, f, I=I))
    
draw(m)
plot(m,u)

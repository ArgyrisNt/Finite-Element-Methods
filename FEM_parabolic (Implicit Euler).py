import numpy as np
from skfem import *
from skfem import accum
from skfem import LinearForm
from skfem.helpers import grad
from skfem.models.poisson import laplace, unit_load, mass
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
    Mlocal = np.zeros((3,3));
    # Define matrix B, which exists in the map x=F(ξ)=Bξ+x(3)
    B = np.matrix([p[0,:]-p[2,:],p[1,:]-p[2,:]]);
    # Find the transpose of matrix B
    BT = np.transpose(B);
    # Find the inverse of the transpose of matrix B
    BTI = np.linalg.inv(BT);
    # Store gradients of basis function of the reference triangle
    G = np.matrix([[1,0,-1],[0,1,-1]]);
    area = 0.5*abs(np.linalg.det(B));    
    for i in range(0,3):
        # Calculate diagonal elements of local mass matrix 
        Mlocal[i,i] = 2*area/12;
        for j in range(0,3):
            # Calculate non-diagonal elements of local mass matrix 
            Mlocal[i,j] = 1*area/12;           
            BG = BTI*G[:,j];
            BGT = np.transpose(BTI*G[:,i]);
            # Calculate the local stiffness matrix 
            Alocal[i,j] = area*BGT*BG;
    return [Alocal,Mlocal]
    
# Assembling stiffness matrix.
def stiff(node,connectivity):
    N = np.size(node,0);
    NT = np.size(connectivity,0);
    A = np.zeros((N,N));
    M = np.zeros((N,N));
    # Call another funtion, in order to calculate the local stiffness matrix
    for k in range(0,NT):
        [Alocal,Mlocal] = localstiff(node[connectivity[k,:],:]);
        for i in range(0,3):
            for j in range(0,3):
                # Calculate the stiffness matrix
                A[connectivity[k,i],connectivity[k,j]] = A[connectivity[k,i],connectivity[k,j]] + Alocal[i,j];
                # Calculate the mass matrix   
                M[connectivity[k,i],connectivity[k,j]] = M[connectivity[k,i],connectivity[k,j]] + Mlocal[i,j];
    return [A,M]
    
# Find right hand side F, using mid point quadrature rule
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
bt1 = areaall/3;
bt = np.transpose(np.array([bt1,bt1,bt1]));
F = accum.accum(connectivity,bt)

# Create the triangular mesh
[A,M] = stiff(node,connectivity);
m = MeshTri(np.transpose(node), np.transpose(connectivity))

# Implicit Euler Method
Nt=200;
T0=0; Tf=1;
tau=Tf/Nt;
U=np.zeros((N,Nt));
U[:,0] = 0;
for n in range(1,Nt):
    U[:,n] = np.matmul(np.linalg.inv(tau*A+M),(tau*F+np.matmul(M,U[:,n-1])));

draw_mesh2d(m)
plot(m, U[:,199])

################################## REFINEMENT ################################
from scipy.sparse import csr_matrix, csc_matrix
from Exercise1 import mm

@LinearForm
def loading(v, w):
    return(10 * w.x[0] * v)

e = ElementTriP1()
basis = InteriorBasis(mm, e)
A2 = asm(laplace, basis)
M2 = asm(mass, basis)
s = asm(loading, basis)
A2 = csr_matrix.todense(A2)
M2 = csr_matrix.todense(M2)
U2=np.zeros((len(A2),Nt));

U2[:,0] = 0;
for n in range(1,Nt):
    firstterm = np.linalg.inv(tau*A2+M2);
    secondterm = tau*s+np.matmul(M2,U2[:,n-1]);
    U2[:,n] = np.transpose(np.matmul(firstterm,np.transpose(secondterm)));
    
draw_mesh2d(mm)
plot(mm, U2[:,199])
### SET UP OF MEMBRANES
# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse.linalg as sla
from pymrm import construct_grad, construct_div, update_csc_array_indices, construct_convflux_upwind, construct_interface_matrices

# Define physical parameters
L = 0.2  # Length of the membrane module (m)
D = 1e-4  # Diffusion coefficient (m^2/s)
P = 0.002  # Permeability of the membrane (m/s)
c_in = 1.0  # Inlet concentration (mol/m^3)
v_ret = 1e-1  # Retentate velocity (m/s)
v_perm = 1e-1  # Permeate velocity (m/s)
R_ret = 5e-3  # Inner radius of the retentate side (m)
R_perm = R_ret + 5e-4  # Inner radius of the permeate side (m)
R_out = 7e-3  # Outer radius of the permeate side (m)

# Define discretization parameters
num_z = 100  # Number of grid points in the axial direction
num_r_perm = 30  # Number of grid points in the radial direction (permeate)
num_r_ret = 30  # Number of grid points in the radial direction (retentate)
dz = L/num_z  # Axial grid spacing

# Generate grid points
z_f = np.linspace(0, L, num_z+1)  # Axial face grid points
z_c = 0.5*(z_f[1:] + z_f[:-1])  # Axial cell-centered grid points
r_f_ret = np.linspace(0, R_ret, num_r_ret+1)  # Radial face grid points (retentate)
r_c_ret = 0.5*(r_f_ret[1:] + r_f_ret[:-1])  # Radial cell-centered grid points (retentate)
r_f_perm = np.linspace(R_perm, R_out, num_r_perm+1)  # Radial face grid points (permeate)
r_c_perm = 0.5*(r_f_perm[1:] + r_f_perm[:-1])  # Radial cell-centered grid points (permeate)

# Define boundary conditions
bc_neumann = {'a': 1, 'b': 0, 'd': 0}  # Neumann boundary condition
bc_dirichlet = {'a': 0, 'b': 1, 'd': 1}  # Dirichlet boundary condition

# Define shapes for matrices
shape_ret = (num_z, num_r_ret)  # Shape of the retentate domain
shape_perm = (num_z, num_r_perm)  # Shape of the permeate domain
shape_d = (num_z, 1)  # Shape for boundary condition vectors

# Construct divergence matrices for retentate and permeate domains
div_ret_mat_z = construct_div(shape_ret, z_f, axis=0, nu=0)  # Axial divergence (retentate)
div_ret_mat_r = construct_div(shape_ret, r_f_ret, axis=1, nu=1)  # Radial divergence (retentate)
div_perm_mat_z = construct_div(shape_perm, z_f, axis=0, nu=0)  # Axial divergence (permeate)
div_perm_mat_r = construct_div(shape_perm, r_f_perm, axis=1, nu=1)  # Radial divergence (permeate)

# Construct convection flux matrices for retentate and permeate domains
conv_ret_mat, conv_ret_bc = construct_convflux_upwind(shape_ret, z_f, z_c, (bc_dirichlet, bc_neumann), v_ret, axis=0)
conv_ret_bc *= c_in  # Apply inlet concentration to boundary condition
conv_perm_mat, conv_perm_bc = construct_convflux_upwind(shape_perm, z_f, z_c, (bc_neumann, bc_dirichlet), -v_perm, axis=0)
conv_perm_bc *= 0.0  # No inlet concentration for permeate

### FUNCTION TO PRINT AND COMPUTE MASS BALANCES
from pymrm import compute_boundary_values

def print_balances(c_ret, c_perm, c_b_ret, c_b_perm):
    # Compute boundary values for retentate and permeate sides
    c_in_ret, _, c_out_ret, _  = compute_boundary_values(c_ret, z_f, z_c, ({'a':0, 'b': 1, 'd':c_in}, bc_neumann), axis=0)
    c_out_perm, _, c_in_perm, _  = compute_boundary_values(c_perm, z_f, z_c, (bc_neumann, {'a':0, 'b': 1, 'd':0}), axis=0)

    # Compute cross-sectional areas for radial and axial directions
    dA_r_ret = np.pi*(r_f_ret[1:]**2 - r_f_ret[:-1]**2).reshape((1,-1))
    dA_r_perm = np.pi*(r_f_perm[1:]**2 - r_f_perm[:-1]**2).reshape((1,-1))
    dA_z_ret = 2*np.pi*R_ret*(z_f[1:] - z_f[:-1]).reshape((-1,1))
    dA_z_perm = 2*np.pi*R_perm*(z_f[1:] - z_f[:-1]).reshape((-1,1))

    # Compute gradients at the membrane interface
    _, _, _, grad_mem_ret  = compute_boundary_values(c_ret, r_f_ret, r_c_ret, (bc_neumann, {'a':0, 'b': 1, 'd':c_b_ret}), axis=1)
    _ , grad_mem_perm, _, _  = compute_boundary_values(c_perm, r_f_perm, r_c_perm, ({'a':0, 'b': 1, 'd':c_b_perm}, bc_neumann), axis=1)

    # Compute flow rates
    flow_in_ret = v_ret*np.sum(c_in_ret*dA_r_ret, axis=1).reshape(())
    flow_out_ret = v_ret*np.sum(c_out_ret*dA_r_ret, axis=1).reshape(())
    flow_in_perm = v_perm*np.sum(c_in_perm*dA_r_perm, axis=1).reshape(())
    flow_out_perm = v_perm*np.sum(c_out_perm*dA_r_perm, axis=1).reshape(())
    flow_mem_ret = -D*np.sum(grad_mem_ret*dA_z_ret, axis=0).reshape(())
    flow_mem_perm = -D*np.sum(grad_mem_perm*dA_z_perm, axis=0).reshape(())

    # Print mass balances
    print(f"retentate side: inlet molar flow = {flow_in_ret:.5e}, membrane molar flow out = {flow_mem_ret:.5e}, outlet molar flow = {flow_out_ret:.5e}")
    print(f"retentate balance: {flow_in_ret - flow_mem_ret - flow_out_ret:.5e}")
    print(f"permeate side:  inlet molar flow = {flow_in_perm:.5e}, membrane molar flow in  = {flow_mem_perm:.5e}, outlet molar flow = {flow_out_perm:.5e}")
    print(f"permeate balance: {flow_in_perm + flow_mem_perm - flow_out_perm:.5e}")
    print(f"overall balance: {flow_in_ret + flow_in_perm - flow_out_ret - flow_out_perm:.5e}")

if __name__ == "__main__":
    # Explicit coupling
    from pymrm import compute_boundary_values
    import copy

    num_iter = 20  # Number of iterations for the explicit coupling
    c_ret = np.zeros(shape_ret)
    c_perm = np.zeros(shape_perm)
    c_b_ret = np.zeros((num_z, 1))
    c_b_perm = np.zeros((num_z, 1))

    bc_mem_ret = {'a': D, 'b': P, 'd': P}
    bc_r_ret = (bc_neumann, bc_mem_ret)
    bc_mem_perm = {'a': R_perm/R_ret*D, 'b': P, 'd': P}
    bc_r_perm = (bc_mem_perm, bc_neumann)

    grad_ret_mat, _, grad_ret_bc = construct_grad(shape_ret, r_f_ret, r_c_ret, bc_r_ret, axis = 1, shapes_d = (None, shape_d))
    grad_perm_mat, grad_perm_bc, _ = construct_grad(shape_perm, r_f_perm, r_c_perm, bc_r_perm, axis = 1, shapes_d = (shape_d, None))

    jac_ret_mat = div_ret_mat_z @ conv_ret_mat - D*div_ret_mat_r @ grad_ret_mat
    jac_ret_lu = sla.splu(jac_ret_mat)
    jac_perm_mat = div_perm_mat_z @ conv_perm_mat - D*div_perm_mat_r @ grad_perm_mat
    jac_perm_lu = sla.splu(jac_perm_mat)

    rhs_ret_bc_const = -div_ret_mat_z @ conv_ret_bc
    rhs_perm_bc_const = -div_perm_mat_z @ conv_perm_bc
    jac_ret_bc = D*div_ret_mat_r @ grad_ret_bc
    jac_perm_bc = D*div_perm_mat_r @ grad_perm_bc
    bc_r_ret_interp = copy.deepcopy(bc_r_ret)
    bc_r_perm_interp = copy.deepcopy(bc_r_perm)

    for i in range(num_iter):
        bc_r_perm_interp[0]['d'] = bc_r_perm[0]['d']*c_b_ret
        c_b_perm, _, _, _ = compute_boundary_values(c_perm, r_f_perm, r_c_perm, bc_r_perm_interp, axis=1)
        rhs_ret_bc = rhs_ret_bc_const + jac_ret_bc @ c_b_perm.reshape(-1,1)
        c_ret[:]   = jac_ret_lu.solve(rhs_ret_bc).reshape(shape_ret)
        
        bc_r_ret_interp[1]['d'] = bc_r_ret[1]['d']*c_b_perm
        _, _, c_b_ret, _  = compute_boundary_values(c_ret, r_f_ret, r_c_ret, bc_r_ret_interp, axis=1)
        rhs_perm_bc= rhs_perm_bc_const +  jac_perm_bc @ c_b_ret.reshape(-1,1)
        c_perm[:]  = jac_perm_lu.solve(rhs_perm_bc).reshape(shape_perm)
        

    plt.pcolormesh(z_f, r_f_ret, c_ret.transpose(), shading='flat', cmap='viridis', vmin=0, vmax=c_in)
    plt.pcolormesh(z_f, r_f_perm, c_perm.transpose(), shading='flat', cmap='viridis', vmin=0, vmax=c_in)

    plt.colorbar()
    plt.show()

    _, _, c_b_ret, _  = compute_boundary_values(c_ret, r_f_ret, r_c_ret, bc_r_ret_interp, axis=1)
    c_b_perm, _, _, _ = compute_boundary_values(c_perm, r_f_perm, r_c_perm, bc_r_perm_interp, axis=1)
    print_balances(c_ret, c_perm, c_b_ret, c_b_perm)
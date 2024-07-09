import numpy as np
from skfem.assembly import BilinearForm
from skfem.helpers import ddot, trace, sym_grad, eye

def linear_stress_anisotropic(C):
    """
    Define the linear-elastic stress-strain relationship for anisotropic materials.

    Parameters
    ----------
    C : numpy.ndarray
        The 6x6 stiffness matrix in Voigt notation.

    Returns
    -------
    function
        A function that computes the stress tensor from the strain tensor.
    """
    def stress_strain_relation(T):
        # Convert 3x3 strain tensor to 6x1 strain vector (Voigt notation)
        strain_vector = np.array([
            T[0, 0],
            T[1, 1],
            T[2, 2],
            2 * T[1, 2],
            2 * T[0, 2],
            2 * T[0, 1]
        ])
        # Compute stress vector
        stress_vector = C @ strain_vector
        # Convert 6x1 stress vector back to 3x3 stress tensor
        stress_tensor = np.array([
            [stress_vector[0], stress_vector[5], stress_vector[4]],
            [stress_vector[5], stress_vector[1], stress_vector[3]],
            [stress_vector[4], stress_vector[3], stress_vector[2]]
        ])
        return stress_tensor
    return stress_strain_relation

def linear_elasticity_anisotropic(C):
    """
    Weak form of the linear elasticity operator for anisotropic materials.

    Parameters
    ----------
    C : numpy.ndarray
        The 6x6 stiffness matrix in Voigt notation.

    Returns
    -------
    function
        The weak form function for the anisotropic material.
    """
    stress_function = linear_stress_anisotropic(C)

    @BilinearForm
    def weakform(u, v, w):
        return ddot(stress_function(sym_grad(u)), sym_grad(v))

    return weakform

# Example usage:
# Define the stiffness matrix for an anisotropic material (example values)
C = np.array([
    [1, 0.2, 0.3, 0, 0, 0],
    [0.2, 1, 0.3, 0, 0, 0],
    [0.3, 0.3, 1, 0, 0, 0],
    [0, 0, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0, 0.5]
])

# Create the weak form function for anisotropic elasticity
weakform_anisotropic = linear_elasticity_anisotropic(C)

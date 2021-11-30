"""Parameters for the Newns-Anderson model from LCAO calculations."""
from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass
class NewnsAndersonLCAO:
    """ Get the parameters for the Newns-Anderson model with LCAO. """
    H_MM: list
    S_MM: list
    adsorbate_basis_index: list
    metal_basis_index: list
    cutoff_range: list 
    pad_range: list
    broadening_width: float

    def __post_init__(self):
        """Post initialization method."""
        self.get_chemisorption_function()
        self.order_chemsorption_function_by_energy()
        self.cutoff_delta()
        self.pad_with_zeros()
        self.smoothen_chemisorption_function()
        self.get_adsorbate_density_of_states()

    def _subdiagonalize(self, index):
        """Subdiagonalize the Hamiltonian. """
        H = self.H_MM.take(index, axis=0).take(index, axis=1) 
        S = self.S_MM.take(index, axis=0).take(index, axis=1) 

        # diagonalize the sub-matrix
        eigenval, eigenvec = np.linalg.eig(np.linalg.solve(S, H))

        # normalise the eigenvectors by taking into account the overlap
        for col in eigenvec.T:
            col /= np.sqrt(np.dot(col.conj(), np.dot(S, col)))

        # translation matrix to change H
        translation = np.identity(self.H_MM.shape[0], dtype=complex)
        for i in range(len(index)):
            for j in range(len(index)):
                translation[index[i], index[j]] = eigenvec[i, j]
        
        # Unitary transform to convert the overall H
        # Trans^T* . H . Trans
        self.H_MM = np.dot(np.transpose(np.conj(translation)), np.dot(self.H_MM, translation)) 
        self.S_MM = np.dot(np.transpose(np.conj(translation)), np.dot(self.S_MM, translation))

        return eigenval

    def get_chemisorption_function(self):
        """Get the Chemisorption function for the Newns-Anderson Model
           Delta = Sigma_{k} = | e_k * s_{ak} - v_{ak} | ^2"""
        eigenval_ads = self._subdiagonalize(self.adsorbate_basis_index)
        eigenval_metal = self._subdiagonalize(self.metal_basis_index)

        self.delta = np.zeros((len(self.adsorbate_basis_index), len(self.metal_basis_index)))
        # Store the Vak for each adsorbate basis index
        self.Vak = np.zeros(len(self.adsorbate_basis_index))

        for i, a in enumerate(self.adsorbate_basis_index):
            self.Vak[i] = np.sqrt( np.sum([ vak**2 for vak in self.H_MM[a] ]) )
            for j, k in enumerate(self.metal_basis_index):

                eps_k = eigenval_metal[j]
                s_ak = self.S_MM[a, k]
                v_ak = self.H_MM[a, k]

                self.delta[i, j] = np.abs( eps_k * s_ak - v_ak )**2
        
        self.delta *= np.pi
        self.eigenval_ads = np.real(eigenval_ads)
        self.eigenval_metal = np.real(eigenval_metal)
    
    def cutoff_delta(self):
        """Cutoff the delta function at a certain distance away from the fermi energy."""
        accepted_index = [index for index in range(len(self.eigenval_metal)) 
                                if self.eigenval_metal[index] > self.cutoff_range[0] and
                                self.eigenval_metal[index] < self.cutoff_range[1]]
        self.delta = self.delta[:, accepted_index]
        self.eigenval_metal = self.eigenval_metal[accepted_index]
    
    def pad_with_zeros(self):
        """Pad the chemisorption function with zeros."""
        shape_delta = self.delta.shape + np.array([0, self.pad_range[0]+self.pad_range[1]])
        padded_delta = np.zeros(shape_delta)
        for i in range(len(self.delta)):
            padded_delta[i] = np.pad(self.delta[i], (self.pad_range[0], self.pad_range[1]), 'constant')
        self.delta = padded_delta
        # Extra energy values
        negative_energies = np.linspace(-10, 0, self.pad_range[0]) + self.eigenval_metal[0]
        positive_energies = np.linspace(0, 10, self.pad_range[1]) + self.eigenval_metal[-1]
        self.eigenval_metal = np.concatenate((negative_energies, self.eigenval_metal, positive_energies), axis=None)

    def order_chemsorption_function_by_energy(self):
        """Order the chemisorption function by energy."""
        for i in range(len(self.delta)):
            self.delta[i] = np.real(self.delta[i][np.argsort(self.eigenval_metal)])
        self.eigenval_metal = np.sort(self.eigenval_metal)
    
    def smoothen_chemisorption_function(self):
        """Smoothen the chemisorption function by broadering it with a Gaussian."""
        for i in range(len(self.delta)):
            for j in range(len(self.delta[i])):
                broading = np.exp( -(self.eigenval_metal - self.eigenval_metal[j])**2 
                                / (2 * self.broadening_width**2) )
                self.delta[i] += broading
    
    def get_adsorbate_density_of_states(self):
        """Get the adsorbate density of states from the Newns Anderson model."""
        self.Lambda = np.imag(signal.hilbert(self.delta)) 







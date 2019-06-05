import copy
import numpy as np
import george
from george.kernels import ExpSquaredKernel, Matern52Kernel, \
    ExpKernel, RationalQuadraticKernel, Matern32Kernel
import scipy.optimize as op


#Assert statements to guarantee the linter doesn't complain
assert ExpSquaredKernel
assert Matern52Kernel
assert ExpKernel
assert Matern32Kernel
assert RationalQuadraticKernel

class _pmesh_emulator(object):
    """
    An emulator for particle mesh simulations. The emulator is trained
    on a set of input power spectra at given locations in cosmological
    parameter space. The power spectra are evaluated over a set of
    redshifts and wavenumbers (h/Mpc com.).

    Args:
        parameters (array-like): locations in parameter space
            of the input power spectra.
        redshifts (float array-like): list of redshifts. 
            Can be a single number.
        k (array-like): wavenumbers of the input power spectra.
        power_spectra (array-like): 2D array of power spectra
            evaluated at each location in parameter space.
    """

    def __init__(self, parameters, redshifts, k, power_spectra,
                 number_of_principle_components=6, kernel=None):
        parameters = np.asarray(parameters)
        redshifts = np.asarray(redshifts)
        k = np.asarray(k)
        power_spectra = np.asarray(power_spectra)

        if parameters.ndim != 2:
            raise Exception("Parameters must be 2D array.")
        if power_spectra.ndim != 2:
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        if len(parameters) != len(power_spectra):
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        if len(redshifts)*len(k) != len(power_spectra[0]):
            raise Exception("Power spectra must be a 2D array of dimensions "+
                            "N_parameters x (N_k*N_z).")
        
        self.parameters = parameters
        self.redshifts = redshifts
        self.k = k
        self.power_spectra = power_spectra
        self.Npars = len(self.parameters[0])

        self.NPC = number_of_principle_components
        metric_guess = np.std(self.parameters, 0)
        if kernel is None:
            kernel = 1.*ExpSquaredKernel(metric=metric_guess, ndim=self.Npars)
        self.kernel = kernel

    def train(self):
        """Train the emulator.

        Args:
            None

        Return:
            None
        """
        zs = self.redshifts
        k = self.k
        p = self.power_spectra
        k2p = copy.deepcopy(p)
        Nk = len(k)
        Nz = len(zs)

        #Multiply each P(k) by k^2, but note the shapes
        #of the power spectra array we have to deal with
        for i in range(Nz):
            lo = i*Nk
            hi = (i+1)*Nk
            k2p[:, lo:hi] *= k**2

        #Take the log -- this reduces the dynamic range
        lnk2p = np.log(k2p)
        
        #Remove the mean and make it unit variance in each k bin
        lnk2p_mean = np.mean(lnk2p)
        lnk2p_std = np.std(lnk2p, 0)
        lnk2p = (lnk2p - lnk2p_mean)/lnk2p_std

        #Save what we have now
        self.lnk2p = lnk2p
        self.lnk2p_mean = lnk2p_mean
        self.lnk2p_std = lnk2p_std

        #Do SVD to pull out principle components
        u,s,v = np.linalg.svd(lnk2p, 0) #Do the PCA
        s = np.diag(s)
        N = len(s)
        P = np.dot(v.T, s)/np.sqrt(N)
        Npc = self.NPC #number of principle components
        phis = P.T[:Npc]
        ws = np.sqrt(N) * u.T[:Npc]
        #Save the weights and PCs
        self.ws = ws
        self.phis = phis

        #Create the GPs and save them
        gplist = []
        for i in range(Npc):
            ws = self.ws[i, :]
            kern = copy.deepcopy(self.kernel)
            gp = george.GP(kernel=kern, fit_kernel=True, mean=np.mean(ws))
            gp.compute(self.parameters)
            gplist.append(gp)
            continue
        self.gplist = gplist

        #Train the GPs
        for i, gp in enumerate(self.gplist):
            ws = self.ws[i, :]
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(ws, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(ws, quiet=True)
            p0 = gp.get_parameter_vector()
            result = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(result.x)
            continue
        
        self.trained=True
        return

    def predict(self, params):
        """Predict the power spectrum at a set of cosmological parameters.

        Args:
            params (float or array-like): parameters of the requested 
                power spectra

        Returns:
            (array-like): length (Nz x Nk) 1D array with the predicted
                power spectra for the requested cosmology

        """

        if not self.trained:
            raise Exception("Need to train the emulator first.")

        params = np.atleast_1d(params)
        if params.ndim > 1:
            raise Exception("'params' must be a single point in parameter "+
                            "space; a 1D array at most.")
        if len(params) != self.Npars:
            raise Exception("length of 'params' does not match training "+\
                            "parameters.")
        #For higher dimensional trianing data, george requires a 2D array...
        if len(params) > 1:
            params = np.atleast_2d(params)

        #Loop over d GPs and predict weights
        wp = np.array([gp.predict(ws, params)[0] for ws, gp in\
                       zip(self.ws, self.gplist)])
        
        #Multiply by the principle components to get predicted lnk2p
        lnk2p_pred = wp[0]*self.phis[0]
        for i in range(1, self.NPC):
            lnk2p_pred += wp[i]*self.phis[i]

        #Multiply on the stddev and add on the mean
        lnk2p_pred = lnk2p_pred *self.lnk2p_std + self.lnk2p_mean
        k2p_pred = np.exp(lnk2p_pred)
        k = self.k
        zs = self.redshifts
        Nk = len(k)
        Nz = len(zs)
        P_pred = k2p_pred
        #Multiply each P(k) by k^2, but note the shapes
        #of the power spectra array we have to deal with
        for i in range(Nz):
            lo = i*Nk
            hi = (i+1)*Nk
            P_pred[lo:hi] /= k**2
        return P_pred

class pmesh_emulator(object):
    def __init__(self, excluded_indices=None, number_of_principle_components=6):
        
        import os, inspect
        data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/"
        
        self.number_of_principle_components = number_of_principle_components
        self.params = np.loadtxt(data_path+"training_points.txt")
        self.sf = np.linspace(0.02, 1.0, 30) #30 Snapshots
        self.zs = 1./self.sf - 1.
        self.k = np.load(data_path+"k.npy")
        self.pkz = np.load(data_path+"pkz_data_Nsim_x_NkNz.npy")
        
        if np.any(self.pkz <= 0):
            raise Exception("problem: negative or 0 P(k,z)")

        if np.any(np.isnan(self.pkz)):
            raise Exception("problem: nan value in P(k,z)")

        if np.any(np.isinf(self.pkz)):
            raise Exception("problem: inf value in P(k,z)")

        if excluded_indices is not None:
            inds = np.arange(len(self.params))
            self.excluded_indices=excluded_indices
            self.excluded_params = self.params[excluded_indices]
            self.excluded_pkz = self.pkz[excluded_indices]
            self.params = np.delete(self.params, excluded_indices, axis=0)
            self.pkz = np.delete(self.pkz, excluded_indices, axis=0)

        self._emu = _pmesh_emulator(self.params, self.zs,
                                    self.k, self.pkz,
                                    number_of_principle_components)
        self._emu.train()
        
    def predict(self, params):
        return self._emu.predict(params)

if __name__ == "__main__":
    emu = pmesh_emulator()

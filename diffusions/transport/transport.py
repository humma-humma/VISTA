import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """
    Transport model class. It defines the path the data takes, 
    calculates training targets, and converts model outputs 
    into mathematical objects (Drift/Score) needed for sampling.
    Args:
    - model_type: ModelType enum specifying model prediction type i.e. what the neural network outputs (Velocity, Noise, or Score)
    - path_type: PathType enum specifying the path interpolant i.e. the geometry of the interpolating path between  data and noise (Linear, GVP, VP)
    - loss_type: WeightType enum specifying the loss weighting type i.e. how the loss is weighted during training (None, Velocity, or Likelihood)
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling    
    """

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {                                         ## Used to select different path interpolants
            PathType.LINEAR: path.ICPlan,                        ## Linear interpolation in score space, used for Rectified Flow / SiT
            PathType.GVP: path.GVPCPlan,                         ## Geometric Variance Preserving interpolation, standard diffusion / DDPM
            PathType.VP: path.VPCPlan,                           ## Variance Preserving interpolation, standard diffusion / DDPM
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()            ## Create the path sampler object, used to calculate the ground truth x_t and target velocity
        self.train_eps = train_eps                               
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        """
            Computes the log-probability of the latent variable z under a standard Gaussian distribution. 
            This is used for calculating likelihoods (bits per dimension)
            ## NOT FOR STANDARD IMAGE GENERATION PURPOSES ##
            
            Standard multivariate normal prior
            Assume z is batched
            Args:
                z: [batch, *dim] Tensor of latent variables
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.   ## Log probability formula for standard Gaussian    
        return th.vmap(_fn)(z)                                              ## Vectorized (v)map over the batch dimension
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        """
        Check & return the time interval [t0, t1] for training/sampling
        This is the valid interval where the SDE/ODE is solved
        Args:
        - train_eps: small epsilon for avoiding instability during training
        - sample_eps: small epsilon for avoiding instability during sampling
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - sde: whether solving SDE (True) or ODE (False)
        - reverse: whether solving in reverse time direction
        - eval: whether in evaluation
        - last_step_size: size of the last step during sampling
        """

        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):
            ## VP path has numerical issues at t=0 for both SDE and ODE, so singularity avoidance is always needed
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): 
            ## For IC/GVP path, singularity avoidance is only needed for SDE or non-velocity model, so t0 > 0, else t0 = 0
            ## avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            ## Reverse time direction
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """
        Sampling x0 & t based on shape of x1 (if needed)
        Prepares a batch for training by picking random times and random start points
        Args:
        - x1: datapoint; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)                                                       ## Sample x0 from standard Gaussian
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)                ## Get valid time interval for training
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0                                 ## Sample uniform times in [t0, t1]
        t = t.to(x1)
        return t, x0, x1                                                             ## Return sampled times t, noise x0, data x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1)                            ## Sample training batch to get times t, noise x0, data x1
        t, xt, ut = self.path_sampler.plan(t, x0, x1)          ## Compute x_t and target velocity u_t based on the path interpolant
        model_output = model(xt, t, **model_kwargs)            ## Model prediction at (x_t, t)
        B, *_, C = xt.shape                                    ## Batch size B, channel dim C
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output

        # Recover x_start (denoised sample) based on the prediction type
        if self.model_type == ModelType.VELOCITY:
            # Formula: x_0 = x_t - t * v
            # We use path.expand_t_like_x to broadcast 't' to match 'xt' dimensions
            t_expanded = path.expand_t_like_x(t, xt) 
            terms['pred_xstart'] = xt - t_expanded * model_output

        elif self.model_type == ModelType.NOISE:
             # Formula: x_0 = (x_t - sigma * eps) / alpha
             # Note: This requires alpha/sigma methods to be available on path_sampler
             sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
             alpha_t, _ = self.path_sampler.compute_alpha_t(path.expand_t_like_x(t, xt))
             terms['pred_xstart'] = (xt - sigma_t * model_output) / alpha_t
             
        else:
             # Fallback for Score or undefined types: use Ground Truth to prevent crashes
             terms['pred_xstart'] = x0

        if self.model_type == ModelType.VELOCITY:                                         ## Velocity prediction
            terms['loss'] = mean_flat(((model_output - ut) ** 2))                         ## MSE loss between predicted (model_output) and target velocity (ut)
        else:                                                                             ## Noise / Score prediction
            _, drift_var = self.path_sampler.compute_drift(xt, t)                         ## Compute drift variance at (x_t, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))   ## Compute variance (sigma_t) at (x_t, t)
            if self.loss_type in [WeightType.VELOCITY]:                                   ## Velocity weighted loss
                weight = (drift_var / sigma_t) ** 2                                       
            elif self.loss_type in [WeightType.LIKELIHOOD]:                               ## Likelihood weighted loss
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:                                          
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:                                         ## Noise prediction
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))             ## MSE loss between predicted (model_output) and true noise (x0)
            else:                                                                          ## Score prediction
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))   ## Note the score target is -x0 / sigma_t, so model_output * sigma_t + x0
                
        return terms
    

    def get_drift(self):
        """
        Returns a function body_fn(x, t) representing the "Drift" of the probability flow ODE. 
        This adapts the model's output (which might be noise, score, or velocity) into 
        a standard format that an ODE solver can digest.
        """
        def score_ode(x, t, model, **model_kwargs):
            """
            Convert score model output to drift function for ODE solver
             Args:
                x: [batch, *dim] Tensor at time t
                t: [batch,] Tensor of times
                model: backbone model; could be score, noise, or velocity
                model_kwargs: additional arguments for the model
             Returns:
                drift: [batch, *dim] Tensor representing the drift at (x, t)
            """
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            """
            Convert noise model output to drift function for ODE solver.
            Here, it is first converted to score, then via change of variable
            formula to calculate the equivalent drift.
            Args:
                x: [batch, *dim] Tensor at time t
                t: [batch,] Tensor of times
                model: backbone model; could be score, noise, or velocity
                model_kwargs: additional arguments for the model
            Returns:
                drift: [batch, *dim] Tensor representing the drift at (x, t)
            """
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            """
            Convert velocity model output to drift function for ODE solver.
            In this case, the drift is directly given by the model output.
            Args:
                x: [batch, *dim] Tensor at time t
                t: [batch,] Tensor of times
                model: backbone model; could be score, noise, or velocity
                model_kwargs: additional arguments for the model
            Returns:
                drift: [batch, *dim] Tensor representing the drift at (x, t)
            """
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):

            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """
        Returns a function that computes the Score of x_t = alpha_t * x + sigma_t * eps
        This adapts the model's output (which might be noise, score, or velocity) into 
        a standard format that can be used for SDE solvers, also can be used for CFG.
        """
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
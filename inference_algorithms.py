import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Importance
from pyro.optim import Adam

from search_inference import HashingMarginal, Search, BestFirstSearch

def run_exact_search(model, *args, **kwargs):
    """
    Runs exact inference using the Search algorithm.
    """
    inference = Search(model)
    posterior = inference.run(*args, **kwargs)
    return HashingMarginal(posterior)

def run_best_first_search(model, num_samples, *args, **kwargs):
    """
    Runs approximate inference using the BestFirstSearch algorithm.
    """
    inference = BestFirstSearch(model, num_samples=num_samples)
    posterior = inference.run(*args, **kwargs)
    return HashingMarginal(posterior)

def run_svi(model, guide, num_steps, *args, **kwargs):
    """
    Runs Stochastic Variational Inference.
    """
    svi = SVI(model, 
              guide, 
              Adam({"lr": 0.01}), 
              loss=Trace_ELBO())
    
    pyro.clear_param_store()
    for _ in range(num_steps):
        svi.step(*args, **kwargs)

    # Return a posterior distribution object
    return pyro.infer.EmpiricalMarginal(
        pyro.infer.Importance(guide, num_samples=1000).run(*args, **kwargs)
    )

def run_importance_sampling(model, num_samples, *args, **kwargs):
    """
    Runs approximate inference using Importance Sampling.
    """
    importance = Importance(model, num_samples=num_samples)
    posterior = importance.run(*args, **kwargs)
    return pyro.infer.EmpiricalMarginal(posterior) 

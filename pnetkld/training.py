from typing import Any

import jax
from jax import random, numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state
from clu import metrics
from ml_collections import ConfigDict

@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")
    regularization: metrics.Average.from_output("regularization")

class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict
    distance: int
    l: float

def create_train_state(module, key, tx, config, input_shape):
    """Creates the initial `TrainState`."""
    distance = 0 if config.DISTANCE=="kld" else 1
    variables = module.init(key, jnp.ones(input_shape))
    state, params = variables.pop('params')
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        distance=distance,
        l=config.LAMBDA,
        metrics=Metrics.empty()
    )

def pearson_correlation(vec1, vec2):
    vec1 = vec1.squeeze()
    vec2 = vec2.squeeze()
    vec1_mean = vec1.mean()
    vec2_mean = vec2.mean()
    num = vec1-vec1_mean
    num *= vec2-vec2_mean
    num = num.sum()
    denom = jnp.sqrt(jnp.sum((vec1-vec1_mean)**2))
    denom *= jnp.sqrt(jnp.sum((vec2-vec2_mean)**2))
    return num/denom

def kld(mean_p, logstd_p, mean_q, logstd_q, axis=(1,2,3)):
    """Assume diagonal covariance matrix and that the input is the logvariance."""
    std_p, std_q = jnp.exp(logstd_p), jnp.exp(logstd_q)
    # def safe_div(a, b): return a/b #jnp.where(a == b, 1, a/b)
    logdet_p = jnp.mean(logstd_p, axis=axis)
    logdet_q = jnp.mean(logstd_q, axis=axis)
    
    return (logdet_q - logdet_p) + jnp.mean((1/std_q)*(mean_p - mean_q)**2, axis=axis) + jnp.mean(std_p/std_q, axis=axis) - 1

def js(mean_p, logstd_p, mean_q, logstd_q, axis=(1,2,3)):
    return (1/2)*(kld(mean_p, logstd_p, mean_q, logstd_q, axis) + kld(mean_q, logstd_q, mean_p, logstd_p, axis))

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        (img_mean, img_logstd), updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
        (img_dist_mean, img_dist_logstd), updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
        ## Calculate the KLD
        dist = jax.lax.cond(state.distance == 0,
                            kld,
                            js,
                            img_mean, img_logstd, img_dist_mean, img_dist_logstd)

        regularization = (jnp.mean(jnp.exp(img_logstd)**2) + jnp.mean(jnp.exp(img_dist_logstd)**2))
        
        ## Calculate pearson correlation
        return pearson_correlation(dist, mos) + state.l*regularization, (updated_state, regularization)
    
    (loss, (updated_state, regularization)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss, regularization=regularization)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    state = state.replace(state=updated_state)
    return state


@jax.jit
def compute_metrics(state, batch):
    """Train for a single step."""
    img, img_dist, mos = batch
    def loss_fn(params):
        ## Forward pass through the model
        (img_mean, img_logstd), updated_state = state.apply_fn({"params": params, **state.state}, img, mutable=list(state.state.keys()), train=True)
        (img_dist_mean, img_dist_logstd), updated_state = state.apply_fn({"params": params, **state.state}, img_dist, mutable=list(state.state.keys()), train=True)
        ## Calculate the KLD
        dist = jax.lax.cond(state.distance == 0,
                            kld,
                            js,
                            img_mean, img_logstd, img_dist_mean, img_dist_logstd)

        regularization = (jnp.mean(jnp.exp(img_logstd)**2) + jnp.mean(jnp.exp(img_dist_logstd)**2))
        
        ## Calculate pearson correlation
        return pearson_correlation(dist, mos)
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params), regularization=None)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

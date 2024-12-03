
from typing import Any

from einops import reduce
from jax import numpy as jnp
import flax.linen as nn

from fxlayers.layers import GDN

class IndependentMeanStd(nn.Module):
    """Baseline (Original) PerceptNet model."""
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=True)(inputs)
        outputs = nn.Conv(features=3, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=6, kernel_size=(self.config.CS_KERNEL_SIZE,self.config.CS_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=self.config.N_GABORS, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        mean = GDN(kernel_size=self.config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        logstd = GDN(kernel_size=self.config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        return mean, logstd

class DependentStd(nn.Module):
    """IQA model inspired by the visual system."""
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=True)(inputs)
        outputs = nn.Conv(features=3, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=6, kernel_size=(self.config.CS_KERNEL_SIZE,self.config.CS_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=self.config.N_GABORS, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        mean = GDN(kernel_size=self.config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        logstd = nn.Conv(features=self.config.N_GABORS, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(mean)
        return mean, logstd

class FixedStd(nn.Module):
    """IQA model inspired by the visual system."""
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=True)(inputs)
        outputs = nn.Conv(features=3, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=6, kernel_size=(self.config.CS_KERNEL_SIZE,self.config.CS_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=self.config.N_GABORS, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        mean = GDN(kernel_size=self.config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        logstd = self.config.LOGSTD*jnp.ones_like(mean)
        return mean, logstd

class DependentSingleStd(nn.Module):
    """IQA model inspired by the visual system."""
    config: Any

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=True)(inputs)
        outputs = nn.Conv(features=3, kernel_size=(1,1), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=1, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=6, kernel_size=(self.config.CS_KERNEL_SIZE,self.config.CS_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        outputs = nn.max_pool(outputs, window_shape=(2,2), strides=(2,2))
        outputs = GDN(kernel_size=self.config.GDNGAUSSIAN_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        outputs = nn.Conv(features=self.config.N_GABORS, kernel_size=(self.config.GABOR_KERNEL_SIZE,self.config.GABOR_KERNEL_SIZE), strides=1, padding="SAME")(outputs)
        mean = GDN(kernel_size=self.config.GDNSPATIOFREQ_KERNEL_SIZE, strides=1, padding="SAME", apply_independently=False)(outputs)
        logstd = nn.Dense(features=1)(reduce(mean, "b h w c -> b c", "mean"))
        logstd = logstd[:,None,None,:]*jnp.ones_like(mean)
        return mean, logstd

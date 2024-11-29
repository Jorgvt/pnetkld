import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

import os
from typing import Any, Callable, Sequence, Union
import argparse

import jax
from jax import lax, random, numpy as jnp

import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax.training import orbax_utils

import optax
import orbax.checkpoint

from ml_collections import ConfigDict

import wandb
from iqadatasets.datasets import *
from JaxPlayground.utils.constraints import *
from JaxPlayground.utils.wandb import *

from pnetkld.training import create_train_state, train_step, compute_metrics

parser = argparse.ArgumentParser(description="Trainig a very simple model on TID08 and testing in TID13",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", choices=["independent", "dependent"], help="Choose from the available metrics: mse, kld and js.")
parser.add_argument("--distance", choices=["mse", "kld", "js"], help="Choose from the available metrics: mse, kld and js.")
parser.add_argument("--testing", action="store_true", help="Perform only one batch of training and one of validation.")
parser.add_argument("--wandb", default="disabled", help="WandB mode.")
parser.add_argument("--run_name", default=None, help="Name for the WandB run.")
parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of training epochs.")
parser.add_argument("-b", "--batch-size", type=int, default=16, help="Number of samples per batch.")
parser.add_argument("--lambda", type=float, default=0., help="Lambda coefficient to weight regularization.")
# parser.add_argument("--std-conv", action="store_true", help="Use a Conv layer to produce the logstd instead of the GDN.")

args = parser.parse_args()
args = vars(args)

# dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/databases/IQA//TID/TID2013/", exclude_imgs=[25])

img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

config = {
    "MODEL": args["model"],
    "BATCH_SIZE": args["batch_size"],
    "EPOCHS": args["epochs"],
    "LEARNING_RATE": 3e-4,
    "SEED": 42,
    "DISTANCE": args["distance"],
    "CS_KERNEL_SIZE": 5,
    "GDNGAUSSIAN_KERNEL_SIZE": 1,
    "N_GABORS": 128,
    "GABOR_KERNEL_SIZE": 5,
    "GDNSPATIOFREQ_KERNEL_SIZE": 1,
    "LAMBDA": args["lambda"],
    # "STD_CONV": args["std_conv"],
}
config = ConfigDict(config)

wandb.init(project="PerceptNet_KLD",
           name=args["run_name"],
           job_type="training",
           config=dict(config),
           mode=args["wandb"],
           )
print(config)

if config.MODEL == "independent":
    from pnetkld.models import IndependentMeanStd as Model
elif config.MODEL == "dependent":
    from pnetkld.models import DependentStd as Model

def resize(img, img_dist, mos):
    h, w = 384, 512
    img = tf.image.resize(img, (h//8, w//8))
    img_dist = tf.image.resize(img_dist, (h//8, w//8))
    return img, img_dist, mos


dst_train_rdy = dst_train.dataset.shuffle(buffer_size=100,
                                      reshuffle_each_iteration=True,
                                      seed=config.SEED)\
                                 .batch(config.BATCH_SIZE, drop_remainder=True)\
                                #  .map(resize)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)\
                            #  .map(resize)


state = create_train_state(Model(config), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), config, input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

def check_trainable(path):
    return False

trainable_tree = freeze(flax.traverse_util.path_aware_map(lambda path, v: "non_trainable" if check_trainable(path)  else "trainable", state.params))

optimizers = {
    "trainable": optax.adam(learning_rate=config.LEARNING_RATE),
    "non_trainable": optax.set_to_zero(),
}

tx = optax.multi_transform(optimizers, trainable_tree)

state = create_train_state(Model(config), random.PRNGKey(config.SEED), tx, config, input_shape=(1,384,512,3))
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_param_count = sum([w.size if t=="trainable" else 0 for w, t in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(trainable_tree))])
print(f"Total params: {param_count} | Trainable params: {trainable_param_count}")

wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_param_count

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

metrics_history = {
    "train_loss": [],
    "train_regularization": [],
    "val_loss": [],
    "val_regularization": [],
}

for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state = train_step(state, batch)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        if args["testing"]: break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)
    
    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        if args["testing"]: break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())
    
    ## Checkpointing
    if not args["testing"]:
        if metrics_history["val_loss"][-1] <= max(metrics_history["val_loss"]):
            orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-best"), state, save_args=save_args, force=True) # force=True means allow overwritting.

    wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()}, commit=False)
    wandb.log({"epoch": epoch+1, **{name:values[-1] for name, values in metrics_history.items()}})
    print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}')
    if args["testing"]: break

if not args["testing"]: orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args)

wandb.finish()

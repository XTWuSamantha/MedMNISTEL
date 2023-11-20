# %%
import os
os.environ["USL_MODE"] = "USL"

import numpy as np
import tqdm
import torch
import models.resnet_cifar_cld as resnet_cifar_cld
import models.resnet_medmnist as resnet_medmnist
import utils
from utils import cfg, logger, print_b
import medmnist
from medmnist import INFO, Evaluator

utils.init(default_config_file="configs/BloodMNIST_usl.yaml")

logger.info(cfg)

# %%
print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)

model = resnet_medmnist.__dict__[cfg.MODEL.ARCH]().cuda()

#medmnist
state_dict = utils.single_model(checkpoint["train_model"])
for k in list(state_dict.keys()):
    if k.startswith('fc'):
        del state_dict[k]
# state_dict = utils.single_model(checkpoint["state_dict"])
# for k in list(state_dict.keys()):
#     if k.startswith('encoder_k'):
#         del state_dict[k]
# state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
# state_dict = {k.replace("fc.2.", "fc.1."): v for k, v in state_dict.items()}

mismatch = model.load_state_dict(state_dict, strict=False)

logger.warning(
    f"Key mismatches: {mismatch} (extra contrastive keys are intended)")

model.eval()

# %%
print_b("Loading dataset")
info = INFO[cfg.DATASET.NAME]
num_classes = len(info['label'])


train_memory_dataset, train_memory_loader = utils.train_memory_medmnist(
    dataname = cfg.DATASET.NAME,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME)

target = []
for i in train_memory_loader:
    temp = i[1]
    n = temp.numpy()
    for t in range(len(n)):
        target = np.append(target,n[t])
targetnp = target.astype(int)
# np.save('DermaMNIST_target.npy', target)
target = torch.tensor(targetnp)


# %%
print_b("Loading feat list")
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist

# %%
num_centroids, final_sample_num = utils.get_sample_info_cifar(
    chosen_sample_num=cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
for kMeans_seed in cfg.USL.SEEDS:
    print_b(f"Running k-Means with seed {kMeans_seed}")
    if final_sample_num <= 40:
        # This is for better reproducibility, but has low memory usage efficiency.
        force_no_lazy_tensor = True
    else:
        force_no_lazy_tensor = False

    # This has side-effect: it calls torch.manual_seed to ensure the seed in k-Means is set.
    # Note: NaN in centroids happens when there is no corresponding sample which belongs to the centroid
    cluster_labels, centroids = utils.run_kMeans(feats_list, num_centroids, final_sample_num, Niter=cfg.USL.K_MEANS_NITERS,
                                                 recompute=recompute_num_dependent, seed=kMeans_seed, force_no_lazy_tensor=force_no_lazy_tensor)

    print_b("Getting selections with regularization")
    selected_inds = utils.get_selection(utils.get_selection_with_reg, feats_list, neighbors_dist, cluster_labels, num_centroids, final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS, w=cfg.USL.REG.W,
                                        momentum=cfg.USL.REG.MOMENTUM, horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA, verbose=True, seed=kMeans_seed, recompute=recompute_num_dependent, save=True)

    counts = np.bincount(targetnp[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))
    
    # 为了获取spice中训练时相同的reliable label格式
    fine_tune = np.full(len(targetnp), -100)
    fine_tune[selected_inds] = targetnp[selected_inds]
    save_filename = "reliable_bloodmnist_{}_{}.npy".format(final_sample_num,kMeans_seed)
    np.save(save_filename, fine_tune)

import numpy as np


def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1
    )
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# Load the data
data = np.load("tiny_nerf_data.npz")
images = data["images"]
poses = data["poses"]
focal = data["focal"]
H, W = images.shape[1:3]

flat_data = np.zeros((images.shape[0] * H * W, 9))
for i in range(images.shape[0]):
    curr_data = np.zeros((H * W, 9))
    rays_o, rays_d = get_rays(H, W, focal, poses[i])
    curr_data[:, :3] = rays_o.reshape(-1, 3)
    curr_data[:, 3:6] = rays_d.reshape(-1, 3)
    curr_data[:, 6:9] = images[i].reshape(-1, 3)
    flat_data[i * H * W : (i + 1) * H * W] = curr_data

np.savez("training_data.npz", data=flat_data)

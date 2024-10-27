import os
import time
from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import imageio
import nerfview
import viser


@torch.no_grad()
def test(hn, hf, dataset, epoch, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.

    Returns:
        None: None
    """
    ray_origins = dataset[img_index * H * W : (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W : (img_index + 1) * H * W, 3:6]
    ground_truth_px_values = dataset[img_index * H * W : (img_index + 1) * H * W, 6:]

    data = []  # list of regenerated pixel values
    img_loss = 0
    for i in range(int(np.ceil(H / chunk_size))):  # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(
            device
        )
        ray_directions_ = ray_directions[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(device)
        ground_truth_px_values_ = ground_truth_px_values[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(device)
        regenerated_px_values = render_rays(
            model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins
        )
        data.append(regenerated_px_values)
        img_loss += ((ground_truth_px_values_ - regenerated_px_values) ** 2).sum()
    img_loss /= int(np.ceil(H / chunk_size))
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(
        f"./novel_views/img_epoch_{epoch}_idx_{img_index}.png", bbox_inches="tight"
    )
    plt.close()
    return img_loss


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # density estimation
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        # color estimation
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(
            o, self.embedding_dim_pos
        )  # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(
            d, self.embedding_dim_direction
        )  # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x)  # h: [batch_size, hidden_dim]
        tmp = self.block2(
            torch.cat((h, emb_x), dim=1)
        )  # tmp: [batch_size, hidden_dim + 1]
        h, sigma = (
            tmp[:, :-1],
            self.relu(tmp[:, -1]),
        )  # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(
            torch.cat((h, emb_d), dim=1)
        )  # h: [batch_size, hidden_dim // 2]
        c = self.block4(h)  # c: [batch_size, 3]
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device

    t = torch.linspace(hn, hf, nb_bins, device=device).expand(
        ray_origins.shape[0], nb_bins
    )
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(
        1
    )  # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(
        nb_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def _viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    chunk_size = 10
    curr_device = "cuda"
    W, H = img_wh
    W = min(W, 400)
    H = min(H, 400)
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(curr_device)
    K = torch.from_numpy(K).float().to(curr_device)

    ray_origins = torch.zeros(H * W, 3, device=curr_device)
    ray_directions = torch.zeros(H * W, 3, device=curr_device)

    for i in range(H):
        for j in range(W):
            ray_origins[i * W + j, :] = c2w[:3, 3]
            ray_directions[i * W + j, :] = torch.matmul(
                torch.inverse(K), torch.tensor([j, i, 1.0], device=curr_device)
            )
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1).unsqueeze(1)
    print(ray_origins.shape, ray_directions.shape)
    print(ray_origins[0], ray_directions[0])
    data = []  # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):  # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size : (i + 1) * W * chunk_size].to(
            curr_device
        )
        ray_directions_ = ray_directions[
            i * W * chunk_size : (i + 1) * W * chunk_size
        ].to(curr_device)
        regenerated_px_values = render_rays(
            model, ray_origins_, ray_directions_, hn=2, hf=6, nb_bins=192
        )
        data.append(regenerated_px_values)
    img = torch.cat(data)
    # Rescale to 0-255
    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255
    print(img.shape)
    print(img.max(), img.min())
    return img.cpu().numpy().reshape(H, W, 3).astype(np.uint8)


def train(
    nerf_model,
    optimizer,
    scheduler,
    data_loader,
    device="cpu",
    hn=0,
    hf=1,
    nb_epochs=int(1e5),
    nb_bins=192,
    H=400,
    W=400,
    viewer=None,
):
    training_loss = []
    log_interval = 100
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.show(block=False)
    # total_steps = nb_epochs * len(data_loader)
    curr_step = 0
    for epoch in tqdm(range(nb_epochs)):
        for batch_idx, batch in enumerate(data_loader):
            while viewer.state.status == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            # print(ray_origins[0], ray_directions[0], ground_truth_px_values[0])

            regenerated_px_values = render_rays(
                nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            curr_step += 1
            # viewer.render_fn = functools.partial(
            #     _viewer_render_fn, local_nerf_model=nerf_model
            # )
            viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = nb_bins * num_train_steps_per_sec
            # Update the viewer state.
            viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            viewer.update(curr_step, nb_bins)
            if batch_idx % log_interval == 0:
                print(f"Step: {batch_idx}, Loss: {loss.item()}")
                # Plot the training loss
                ax.clear()
                ax.plot(training_loss, label="Training Loss")
                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                fig.canvas.draw()
                fig.canvas.flush_events()

        scheduler.step()
        test_loss = 0
        with imageio.get_writer(
            f"./novel_views/testing_epoch_{epoch}.gif", mode="I"
        ) as writer:
            for img_index in tqdm(range(200)):
                test_img_loss = test(
                    hn,
                    hf,
                    testing_dataset,
                    epoch,
                    chunk_size=20,
                    img_index=img_index,
                    nb_bins=nb_bins,
                    H=H,
                    W=W,
                )
                writer.append_data(
                    imageio.imread(
                        f"./novel_views/img_epoch_{epoch}_idx_{img_index}.png"
                    )
                )
                test_loss += test_img_loss
        test_loss /= 200
        test_loss = test_loss.item()
        print(f"--- Epoch: {epoch}, Test Loss: {test_loss}")
        ax.plot(
            [batch_idx + epoch * len(data_loader)] * 2,
            [test_loss, test_loss],
            label="Test Loss",
        )
        ax.legend()
        fig.canvas.draw()
    fig.savefig("./training_loss.png")
    plt.close(fig)

    return training_loss


if __name__ == "__main__":
    device = "cuda"
    os.makedirs("./training_views", exist_ok=True)
    os.makedirs("./testing_views", exist_ok=True)
    os.makedirs("./novel_views", exist_ok=True)
    training_dataset = torch.from_numpy(
        np.load("./nerf_datasets/training_data.pkl", allow_pickle=True)
    )
    testing_dataset = torch.from_numpy(
        np.load("./nerf_datasets/testing_data.pkl", allow_pickle=True)
    )
    nb_train_frames = training_dataset.shape[0] // 400 // 400
    nb_test_frames = testing_dataset.shape[0] // 400 // 400
    print(f"Number of training frames: {nb_train_frames}")
    print(f"Number of testing frames: {nb_test_frames}")
    # Save training images
    for i in range(nb_train_frames):
        plt.imsave(
            f"./training_views/img_idx_{i}.png",
            training_dataset[i * 400 * 400 : (i + 1) * 400 * 400, 6:]
            .reshape(400, 400, 3)
            .cpu()
            .numpy(),
        )
    # Save training gif
    with imageio.get_writer(
        "./training_views/training.gif", mode="I", fps=10
    ) as writer:
        for i in range(nb_train_frames):
            writer.append_data(imageio.imread(f"./training_views/img_idx_{i}.png"))

    # Save testing images
    for i in range(nb_test_frames):
        plt.imsave(
            f"./testing_views/img_idx_{i}.png",
            testing_dataset[i * 400 * 400 : (i + 1) * 400 * 400, 6:]
            .reshape(400, 400, 3)
            .cpu()
            .numpy(),
        )
    # Save testing gif
    with imageio.get_writer("./testing_views/testing.gif", mode="I", fps=10) as writer:
        for i in range(nb_test_frames):
            writer.append_data(imageio.imread(f"./testing_views/img_idx_{i}.png"))

    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[2, 4, 8], gamma=0.5
    )
    data_loader = DataLoader(
        training_dataset, batch_size=2048, shuffle=True, num_workers=4
    )
    # render_fn = functools.partial(_viewer_render_fn, local_nerf_model=model)
    render_fn = _viewer_render_fn
    SERVER = viser.ViserServer(port=8080)
    SERVER.scene.world_axes.visible = True
    SERVER.scene.set_up_direction("-z")
    for i in range(nb_train_frames):
        img_view_position = training_dataset[i * 400 * 400, :3].cpu().numpy()
        SERVER.scene.add_image(
            f"training_img_{i}",
            imageio.imread(f"./training_views/img_idx_{i}.png"),
            render_height=4.0,
            render_width=4.0,
            format="png",
            position=img_view_position,
            visible=False,
        )
    VIEWER = nerfview.viewer.Viewer(
        SERVER,
        render_fn=render_fn,
        mode="training",
    )
    # HN: near plane distance, HF: far plane distance
    # 2 and 6 are the values used for this dataset
    train(
        model,
        model_optimizer,
        scheduler,
        data_loader,
        nb_epochs=16,
        device=device,
        hn=2,
        hf=6,
        nb_bins=192,
        H=400,
        W=400,
        viewer=VIEWER,
    )

"""Grad-CAM saliency maps (Selvaraju et al., 2017).

Accuracy cannot distinguish a model reading leaf lesions from one reading the
studio backdrop -- both score well on PlantVillage. This shows which pixels the
predicted class score actually depends on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for a single convolutional layer.

    Usage::

        with GradCAM(model, layer) as cam:
            heatmap = cam(images, class_indices)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "GradCAM":
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        # Fires with the gradient w.r.t. this layer's output -- the dY/dA term.
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activation(self, _module, _inputs, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Return `(heatmaps, logits)`.

        Heatmaps come back as a float array in [0, 1] with shape
        (batch, H, W), already upsampled to the input resolution.
        """
        was_training = self.model.training
        self.model.eval()

        # Gradients are required here, so no autocast/no_grad even though the
        # surrounding evaluation code runs under AMP.
        images = images.clone().requires_grad_(False)
        logits = self.model(images)

        if class_indices is None:
            class_indices = logits.argmax(dim=1)

        # Each sample's score depends only on its own activations, so summing
        # gives per-sample gradients from a single backward pass.
        selected = logits.gather(1, class_indices.view(-1, 1)).sum()

        self.model.zero_grad(set_to_none=True)
        selected.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check the target layer.")

        # Channel weights are the spatially averaged gradients. ReLU keeps only
        # evidence for the class; negative contributions argue for other ones.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))

        cam = F.interpolate(
            cam, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze(1)

        # Normalise per sample so maps are comparable across images.
        flat = cam.flatten(1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if was_training:
            self.model.train()

        return cam.detach().cpu().numpy(), logits.detach()


def background_mass(heatmap: np.ndarray, border_fraction: float = 0.25) -> float:
    """Fraction of Grad-CAM mass in the image border region.

    A crude proxy for background reliance: the PlantVillage leaf is always
    centred, so attention in the outer frame means the model is keying on the
    backdrop. Gives the qualitative figures a number to sit beside.
    """
    h, w = heatmap.shape
    bh, bw = int(h * border_fraction), int(w * border_fraction)

    interior = heatmap[bh : h - bh, bw : w - bw]
    total = heatmap.sum()
    if total <= 0:
        return float("nan")
    return float((total - interior.sum()) / total)

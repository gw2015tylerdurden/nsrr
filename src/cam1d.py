import pytorch_grad_cam
import torch
import numpy as np
from typing import Callable, List, Tuple, Optional
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM1D(pytorch_grad_cam.base_cam.BaseCAM):
    def __init__(self, *args, **kwargs):
        super(BaseCAM1D, self).__init__(*args, **kwargs)
        self.alpha = []

    def get_target_width_height(self,
                                input_tensor: torch.Tensor):
        # None(target_size) does not resize 1D-cam in utils.image.scale_cam_image()
        return None

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None] * activations  # modified for 1D
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam


class GradCAM1D(BaseCAM1D):
    def __init__(self, *args, **kwargs):
        super(GradCAM1D, self).__init__(*args, **kwargs)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # modified for 1D
        weights = np.mean(grads, axis=2)
        self.alpha = weights
        return weights

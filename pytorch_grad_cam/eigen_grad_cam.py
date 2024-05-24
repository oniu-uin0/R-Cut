from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.base_cam import ncut


# Like Eigen CAM: https://arxiv.org/abs/2008.00299
# But multiply the activations x gradients


class EigenGradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenGradCAM, self).__init__(model, target_layers, use_cuda,
                                           reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        weighted_activations = grads * activations
        cam = get_2d_projection(weighted_activations)

        save_map = {}
        save_map['base_cam'] = cam
        """use normalized_cut"""
        n_out = ncut(activations, tau=0.05)
        save_map["ncut"] = n_out
        # _, cam = cv2.threshold(cam[0, :],
        #                        cam[0, :].max() * 0.45, 1,
        #                        cv2.THRESH_TOZERO)
        # cam = progress_mask(cam)
        # cam = ncut(cam[None, :, :], tau=0.1)
        # cam = ncut(cam, tau=0.2)
        cam = ncut(weighted_activations, tau=0.05)
        save_map["cut_cam"] = cam

        return cam, save_map

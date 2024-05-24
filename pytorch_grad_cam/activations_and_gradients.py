import scipy
from matplotlib import pyplot as plt
import torch.nn.functional as F


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            # qkv
            # self.handles.append(
            #     target_layer.register_forward_hook(
            #         self.save_qkv))
            # norm1
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compitability with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_qkv(self, module, input, output):
        qkv = output
        nb_im = 1
        nb_tokens = 197
        nh = 12
        qkv = (
            qkv
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        activation = q
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)

        self.activations.append(activation.cpu().detach())

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
            """show map"""
            # feats_map = output[0][1:].mean(1).reshape(14, 14)
            # feats_map = feats_map.detach().cpu().numpy()
            # # feats_map = scipy.ndimage.zoom(feats_map, [16, 16], order=0, mode='nearest')
            # # map = cv2.resize(att, (img.shape[-1], img.shape[-2]))
            # plt.imshow(feats_map)
            # plt.show()
        self.activations.append(activation.cpu().detach())
        # self.activations.append(output[0, 1:, :].cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

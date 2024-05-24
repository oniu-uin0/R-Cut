import cv2
import os
import numpy as np
import scipy.special
import torch
import torch.nn.functional as F
import ttach as tta
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def closest(mylist, Number):
    answer = []
    mylist = np.sort(mylist, axis=None)
    for i in mylist:
        answer.append(abs(Number - i))
    return answer.index(min(answer)) / len(mylist)


def set_tau(mylist, rate):
    mylist = np.sort(mylist, axis=None)
    return mylist[int(rate * len(mylist))]


def my_norm(x, axis=1):
    # xmean = x.mean(axis=axis, keepdims=True)
    # xstd = np.std(x, axis=axis, keepdims=True)
    xp = np.linalg.norm(x, axis=1, keepdims=True)
    zscore = x / xp
    return zscore


def ncut(cam, tau=0.2):
    """use normalized cut"""
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cam[0].reshape(-1, 196).transpose()
    cam = torch.tensor(cam)
    cam = F.normalize(cam, p=2)
    # cam = my_norm(cam)
    # tau = set_tau(cam, 0.51)
    # near_rate = closest(cam, tau)

    A = (cam @ cam.transpose(1, 0))
    A = A.cpu().numpy()
    A = A > tau
    A = np.where(A.astype(float) == 0, 1e-5, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    # Print second and third smallest eigenvector
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])
    # Using average point to compute bipartition
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(14, 14).astype(float)
    '''if run '''
    # eigenvec = eigenvec * -1
    return eigenvec.reshape(14, 14)[None, :, :]
    # return bipartition[None, :, :]


def progress_mask(mask_pred):
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)

    return mask


class BaseCAM:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        targets,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      targets,
                      activations,
                      grads,
                      eigen_smooth=True):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       targets, activations, grads)
        # weights(1,768) activations(1,768,14,14)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            # cam(1,14,14)
            cam = weighted_activations.sum(axis=1)

        save_map = {}
        save_map["base_cam"] = cam
        """use normalized_cut"""
        n_out = ncut(activations, tau=0.25)
        save_map["ncut"] = n_out
        # _, cam = cv2.threshold(cam[0, :],
        #                        cam[0, :].max() * 0.45, 1,
        #                        cv2.THRESH_TOZERO)
        # cam = progress_mask(cam)
        # cam = ncut(cam[None, :, :], tau=0.1)
        # cam = ncut(cam, tau=0.2)
        cam = ncut(weighted_activations, tau=0.2)
        save_map["cut_cam"] = cam
        # plt.imshow(cam[0])
        # plt.show()
        # cam is numpy
        return cam, save_map

    def forward(self, input_tensor, targets=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        # if isinstance(target_category, int):
        #     target_category = [target_category] * input_tensor.size(0)

        # if target_category is None:
        #     target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        # else:
        #     assert (len(target_category) == input_tensor.size(0))
        pre_target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        pre_class = pre_target_categories[0]
        '''get the class_confidence'''
        class_confidence = scipy.special.softmax(outputs.cpu().data.numpy())[0][pre_class]
        if targets is None:
            # print(target_categories)
            targets = [ClassifierOutputTarget(
                category) for category in pre_target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            # loss = self.get_loss(outputs, target_category)
            loss = sum([target(output)
                        for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer, original_cam, class_confidence, pre_class \
            = self.compute_cam_per_layer(input_tensor,
                                         targets,
                                         eigen_smooth,
                                         class_confidence,
                                         pre_class)
        return self.aggregate_multi_layers(cam_per_layer, original_cam, class_confidence, pre_class)

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            targets,
            eigen_smooth,
            class_confidence,
            pre_class):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        # for target_layer, layer_activations, layer_grads in \
        #         zip(self.target_layers, activations_list, grads_list):
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            cam, original_cam = self.get_cam_image(input_tensor,
                                                   target_layer,
                                                   targets,
                                                   layer_activations,
                                                   layer_grads,
                                                   eigen_smooth)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer, original_cam, class_confidence, pre_class

    def aggregate_multi_layers(self, cam_per_target_layer, original_cam, class_confidence, pre_class):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result), original_cam, class_confidence, pre_class

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       targets=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 targets=None,
                 aug_smooth=False,
                 eigen_smooth=False):

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM

import cv2
import numpy as np
import torch
import ttach as tta
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.mul(cam_ss, R_ss)
    return R_ss_addition


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
                        target_layer,
                        targets,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            # activations(1,768,14,14)
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()
            # (1,768, 14,14)->(1,768,224,224)
            upsampled = upsample(activation_tensor)
            # (1,768,224,224) -> (1,768)
            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            # (1,768,1,1)
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)
            # input_tensor(1,3,224,224)   upsampled(1,768,224,224) ->(1,768, 3,224,224)
            input_tensors = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size") and self.batch_size > 16:
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 12

            scores = []
            # tensor (768,3,224,224)
            for target, tensor in zip(targets, input_tensors):
            # for batch_index, tensor in enumerate(input_tensors):
            #     category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    # outputs = self.model(batch).cpu().numpy()[:, category]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)  # list:768
            scores = scores.view(activations.shape[0], activations.shape[1])
            # (1,768)
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

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
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       targets, activations, grads)
        # weights(1,768) activations(1,768,14,14)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, targets=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        # if isinstance(target_category, int):
        #     target_category = [target_category] * input_tensor.size(0)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]
        # if target_category is None:
        #     target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        # else:
        #     assert (len(target_category) == input_tensor.size(0))

        if self.uses_gradients:
            self.model.zero_grad()
            # loss = self.get_loss(output, target_category)
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
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   target_category,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
        # return self.my_aggregate_multi_layers(input_tensor, target_category, cam_per_layer, eigen_smooth=eigen_smooth)
        # ......if use rollout please change.........
        # return cam_per_layer

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            targets,
            eigen_smooth):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []

        # Loop over the saliency image from every layer
        # R = torch.ones(1, 14, 14).cuda()

        # reverse relevance
        # self.target_layers.reverse()
        # activations_list.reverse()
        # grads_list.reverse()
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

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            # (1,14,14)->(1,224,224)
            scaled = self.scale_cam_image(cam, target_size=None)
            # R += apply_self_attention_rules(R.cuda(), torch.from_numpy(scaled).cuda())
            cam_per_target_layer.append(scaled[:, None, :])
        # (12,1,1,224,224)
        return cam_per_target_layer
        # return R

    def aggregate_multi_layers(self, cam_per_target_layer):
        # (12,1,1,224,224) -> (1,12,224,224)
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # (1, 12, 224, 224) compare the value with 0.
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # ->(1, 224, 224)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def my_aggregate_multi_layers(self,
                                  input_tensor,
                                  targets,
                                  cam_per_target_layer,
                                  eigen_smooth):
        # (12,1,1,224,224) -> (1,12,224,224)
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # (1, 12, 224, 224)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # (1, 224, 224)
        # result = np.mean(cam_per_target_layer, axis=1)
        result = self.get_cam_image(input_tensor,
                                    target_layer=None,
                                    targets=targets,
                                    activations=cam_per_target_layer,
                                    grads=None,
                                    eigen_smooth=eigen_smooth)
        return self.scale_cam_image(result)

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


class ScoreOut(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreOut, self).__init__(model, target_layers, use_cuda,
                                       reshape_transform=reshape_transform)

        if len(target_layers) > 0:
            print("Warning: You are using ScoreCAM with target layers, "
                  "however ScoreCAM will ignore them.")

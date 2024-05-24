import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model, target_layers, use_cuda,
                                       reshape_transform=reshape_transform)

        if len(target_layers) > 0:
            print("Warning: You are using ScoreCAM with target layers, "
                  "however ScoreCAM will ignore them.")

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

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            # tensor (768,3,224,224)
            # for batch_index, tensor in enumerate(input_tensors):
            #     category = target_category[batch_index]
            for target, tensor in zip(targets, input_tensors):
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

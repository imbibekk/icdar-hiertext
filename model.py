import torch
from segmentation_models_pytorch import Unet

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.segmenter = Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=6,
        )
        # if self.cfg.pretrained_path is not None:
        #     ckpt = torch.load(self.cfg.pretrained_path, map_location='cpu')
        #     state_dict = {k: v for k, v in  ckpt['state_dict'].items() if 'segmentation_head' not in k}
        #     self.segmenter.encoder.load_state_dict(state_dict, strict=True)

            # state_dict = ckpt['state_dict']
            # # create new OrderedDict that does not contain `module.`
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k.replace('model.segmenter.', '') # remove `module.`
            #     new_state_dict[name] = v
            # # load params
            # self.segmenter.load_state_dict(new_state_dict)

    def forward(self, pixel_values, **kwargs):
        segmenter_output = self.segmenter(pixel_values)
        return segmenter_output

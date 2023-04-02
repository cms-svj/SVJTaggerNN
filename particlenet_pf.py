import torch
import torch.nn as nn
from ParticleNet import ParticleNet, FeatureConv


class ParticleNetTagger1Path(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger1Path, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    # def forward(self, pf_points, pf_features):
    #     return self.pn(pf_points, self.pf_conv(pf_features))

    def forward(self, points, features, mask):
        if self.pf_input_dropout is not None:
          mask = (self.pf_input_dropout(mask) != 0).float()
          points *= mask
          features *= mask

        points = points
        features = self.pf_conv(features * mask)
        mask = mask

        # print("points")
        # print(points)
        # print("features")
        # print(features)
        # print("mask")
        # print(mask)

        return self.pn(points, features, mask)

def get_model(inputFeatureVars,**kwargs):
    # conv_params = [
    #     (16, (64, 64, 64)),
    #     (16, (128, 128, 128)),
    #     (16, (256, 256, 256)),
    #     ]
    num_of_k_nearest = kwargs.get('num_of_k_nearest', 16)
    num_of_edgeConv_dim = kwargs.get('num_of_edgeConv_dim', [64,128,256])
    num_of_edgeConv_convLayers = kwargs.get('num_of_edgeConv_convLayers', 3)
    num_of_fc_layers = kwargs.get('num_of_fc_layers', 1)
    num_of_fc_nodes, fc_dropout = kwargs.get('num_of_fc_nodes', 128), kwargs.get('fc_dropout', 0.1)
    conv_params = []
    for ec_dim_i in num_of_edgeConv_dim:
        conv_param = []
        for i in range(num_of_edgeConv_convLayers):
            conv_param.append(ec_dim_i)
        conv_params.append([num_of_k_nearest,conv_param])
    fc_params = []
    for i in range(num_of_fc_layers):
        fc_params.append([num_of_fc_nodes, fc_dropout])
    use_fusion = True

    pf_features_dims = len(inputFeatureVars)
    num_classes = kwargs.get('num_classes', 2)
    model = ParticleNetTagger1Path(pf_features_dims, num_classes,
                                   conv_params, fc_params,
                                   use_fusion=use_fusion,
                                   use_fts_bn=kwargs.get('use_fts_bn', False),
                                   use_counts=kwargs.get('use_counts', True),
                                   pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                   for_inference=kwargs.get('for_inference', False)
                                   )
    # model_info = {
    #     'input_names':list(data_config.input_names),
    #     'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
    #     'output_names':['softmax'],
    #     'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
    #     }

    return model


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()

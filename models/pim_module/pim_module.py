import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy

## Stage 2 ##

# Instance whitening
class InstanceWhitening(nn.Module):
    
    def __init__(self, dim):
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w


def instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov):
    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * mask_matrix

    off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) - margin # B X 1 X 1
    loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
    loss = torch.sum(loss) / B

    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B

# Semantic concept modeling for swin
class SCMModule(nn.Module):

    def __init__(self,
                 outs: dict,
                 fine_num_classes: int,
                 coarse_num_classes: int,
                 mid_feature_num: int):
        
        super(SCMModule, self).__init__()
        
        if len(outs['layer4'].shape) == 4:
            input_feature_num = outs['layer4'].shape[-1] * outs['layer4'].shape[-2]
        elif len(outs['layer4'].shape) == 3:
            input_feature_num = outs['layer4'].shape[1]
        self.branch1_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU()) # swin-144
        self.branch1_linear2 = nn.Linear(mid_feature_num, coarse_num_classes)
        self.branch1_iw = InstanceWhitening(coarse_num_classes)
        
        self.branch2_linear1 = nn.Sequential(nn.Linear(input_feature_num, mid_feature_num), nn.ReLU()) # swin-144
        self.branch2_linear2 = nn.Linear(mid_feature_num, fine_num_classes)
        self.branch21_linear = nn.Linear(fine_num_classes, coarse_num_classes)
        self.branch21_iw = InstanceWhitening(coarse_num_classes)
        self.branch22_linear = nn.Linear(fine_num_classes, fine_num_classes)
        self.branch22_iw = InstanceWhitening(fine_num_classes)
        
        self.constraint = nn.MSELoss()
        
    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view((B, C, H*W))
        else:
            x = x.transpose(1, 2).contiguous()
        branch1 = self.branch1_linear1(x)
        branch1 = self.branch1_linear2(branch1)
        branch1 = branch1.transpose(1, 2).contiguous()
        branch1 = branch1.unsqueeze(3)
        branch1, _ = self.branch1_iw(branch1)
        branch1 = branch1.squeeze(3)
        branch2 = self.branch2_linear1(x)
        branch2 = self.branch2_linear2(branch2)
        branch21 = self.branch21_linear(branch2)
        branch21 = branch21.transpose(1, 2).contiguous()
        branch21 = branch21.unsqueeze(3)
        branch21, _ = self.branch21_iw(branch21)
        branch21 = branch21.squeeze(3)
        branch22 = self.branch22_linear(branch2)
        branch22 = branch22.transpose(1, 2).contiguous()
        branch22 = branch22.unsqueeze(3)
        output, _ = self.branch22_iw(branch22)
        output = output.squeeze(3)
        constraint = self.constraint(branch1, branch21)
        
        return output, constraint
        
# Semantic concept embedding
class SCEModule(nn.Module):
    
    def __init__(self):
        
        super(SCEModule, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.linear_b1 = nn.Linear(3, 1)
        self.linear_b2 = nn.Linear(3, 1)
        self.linear_b3 = nn.Linear(3, 1)
        self.linear_b4 = nn.Linear(3, 1)
        
    def forward(self, x, out1):
        output = {}
        mask1 = self.sigmoid(out1.sum(dim=1))
        mask1 = mask1.unsqueeze(1)
        
        for name in x:
            if len(x[name].shape) == 4:
                B, C, H, W = x[name].shape
                x[name] = x[name].view((B, C, H*W))
                x[name] = x[name].transpose(1, 2).contiguous()

        g_b1 = self.relu(torch.mul(x['layer1'], mask1))
        g_b2 = self.relu(torch.mul(x['layer2'], mask1))
        g_b3 = self.relu(torch.mul(x['layer3'], mask1))
        g_b4 = self.relu(torch.mul(x['layer4'], mask1))
        
        mask_avg_b1 = torch.mean(x['layer1'], dim=1, keepdim=True)
        mask_max_b1, _ = torch.max(x['layer1'], dim=1, keepdim=True)
        mask_avg_b2 = torch.mean(x['layer2'], dim=1, keepdim=True)
        mask_max_b2, _ = torch.max(x['layer2'], dim=1, keepdim=True)
        mask_avg_b3 = torch.mean(x['layer3'], dim=1, keepdim=True)
        mask_max_b3, _ = torch.max(x['layer3'], dim=1, keepdim=True)
        mask_avg_b4 = torch.mean(x['layer4'], dim=1, keepdim=True)
        mask_max_b4, _ = torch.max(x['layer4'], dim=1, keepdim=True)
        
        mask2_b1 = torch.cat([mask_max_b1, mask_avg_b1, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b2 = torch.cat([mask_max_b2, mask_avg_b2, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b3 = torch.cat([mask_max_b3, mask_avg_b3, mask1], dim=1).transpose(1, 2).contiguous()
        mask2_b4 = torch.cat([mask_max_b4, mask_avg_b4, mask1], dim=1).transpose(1, 2).contiguous()
        
        mask3_b1 = self.sigmoid(self.linear_b1(mask2_b1)).transpose(1, 2).contiguous()
        mask3_b2 = self.sigmoid(self.linear_b2(mask2_b2)).transpose(1, 2).contiguous()
        mask3_b3 = self.sigmoid(self.linear_b3(mask2_b3)).transpose(1, 2).contiguous()
        mask3_b4 = self.sigmoid(self.linear_b4(mask2_b4)).transpose(1, 2).contiguous()
        
        output['layer1'] = torch.mul(g_b1, mask3_b1)
        output['layer2'] = torch.mul(g_b2, mask3_b2)
        output['layer3'] = torch.mul(g_b3, mask3_b3)
        output['layer4'] = torch.mul(g_b4, mask3_b4)
        
        return output
    
# Semantic constraint combiner
class SCCombiner(nn.Module):
    
    def __init__(self, 
                 outs: dict,
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(SCCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        # branch_size = [2048, 512, 128, 32]
        # branch_size = [2304, 576, 144, 144]
        if len(outs['layer1'].shape) == 4:
            branch_size = [outs['layer1'].shape[-1]**2, outs['layer2'].shape[-1]**2, outs['layer3'].shape[-1]**2, outs['layer4'].shape[-1]**2]
        else:
            branch_size = [outs['layer1'].shape[-2], outs['layer2'].shape[-2], outs['layer3'].shape[-2], outs['layer4'].shape[-2]]
            
        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### merge information
        self.pool0_b1 = nn.Sequential(nn.Linear(branch_size[0], 1), nn.ReLU())
        self.pool0_b2 = nn.Sequential(nn.Linear(branch_size[1], 1), nn.ReLU())
        self.pool0_b3 = nn.Sequential(nn.Linear(branch_size[2], 1), nn.ReLU())
        self.pool0_b4 = nn.Sequential(nn.Linear(branch_size[3], 1), nn.ReLU())

        self.pool1_b1 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b2 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b3 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b4 = nn.Linear(self.proj_size, num_classes)

        self.norm = nn.Sigmoid()
        self.constraint = nn.MSELoss()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        for i in range(len(hs)):
            hs[i] = hs[i].transpose(1, 2).contiguous()
        hs[0] = self.pool0_b1(hs[0])
        hs[1] = self.pool0_b2(hs[1])
        hs[2] = self.pool0_b3(hs[2])
        hs[3] = self.pool0_b4(hs[3])
        for i in range(len(hs)):
            hs[i] = hs[i].flatten(1)
        hs[0] = self.pool1_b1(hs[0])
        hs[1] = self.pool1_b2(hs[1])
        hs[2] = self.pool1_b3(hs[2])
        hs[3] = self.pool1_b4(hs[3])
        
        ## Gram matrix for constraint
        if len(hs[0].shape) == 2:
            gram4 = torch.matmul(hs[0].transpose(0, 1).contiguous(), hs[0])
            gram3 = torch.matmul(hs[1].transpose(0, 1).contiguous(), hs[1])
            gram2 = torch.matmul(hs[2].transpose(0, 1).contiguous(), hs[2])
            gram1 = torch.matmul(hs[3].transpose(0, 1).contiguous(), hs[3])
        elif len(hs[0].shape) == 3:
            gram4 = torch.matmul(hs[0].transpose(1, 2).contiguous(), hs[0])
            gram3 = torch.matmul(hs[1].transpose(1, 2).contiguous(), hs[1])
            gram2 = torch.matmul(hs[2].transpose(1, 2).contiguous(), hs[2])
            gram1 = torch.matmul(hs[3].transpose(1, 2).contiguous(), hs[3])
        constraint_loss = torch.abs(self.constraint(gram2, gram1))
        constraint_loss += torch.abs(self.constraint(gram3, gram1))
        constraint_loss += torch.abs(self.constraint(gram4, gram1))
        
        final_h = hs[0] + hs[1] + hs[2] + hs[3]
        final_h = self.norm(final_h)

        return final_h, constraint_loss



## Origin
class NEWCombiner(nn.Module):
    
    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(NEWCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        branch_size = [2048, 512, 128, 32]
        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### merge information
        self.pool0_b1 = nn.Sequential(nn.Linear(branch_size[0], 1), nn.ReLU())
        self.pool0_b2 = nn.Sequential(nn.Linear(branch_size[1], 1), nn.ReLU())
        self.pool0_b3 = nn.Sequential(nn.Linear(branch_size[2], 1), nn.ReLU())
        self.pool0_b4 = nn.Sequential(nn.Linear(branch_size[3], 1), nn.ReLU())

        self.pool1_b1 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b2 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b3 = nn.Linear(self.proj_size, num_classes)
        self.pool1_b4 = nn.Linear(self.proj_size, num_classes)

        self.norm = nn.Sigmoid()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        for i in range(len(hs)):
            hs[i] = hs[i].transpose(1, 2).contiguous()
        hs[0] = self.pool0_b1(hs[0])
        hs[1] = self.pool0_b2(hs[1])
        hs[2] = self.pool0_b3(hs[2])
        hs[3] = self.pool0_b4(hs[3])
        for i in range(len(hs)):
            hs[i] = hs[i].flatten(1)
        hs[0] = self.pool1_b1(hs[0])
        hs[1] = self.pool1_b2(hs[1])
        hs[2] = self.pool1_b3(hs[2])
        hs[3] = self.pool1_b4(hs[3])
        final_h = hs[0] + hs[1] + hs[2] + hs[3]
        final_h = self.norm(final_h)

        return final_h

class GCNCombiner(nn.Module):

    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 32

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        
        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)
        
        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        hs = self.param_pool0(hs)
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs

class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        self.fpn_size = fpn_size
        ### build classifier
        if self.fpn_size is None:
            self.num_classes = num_classes
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                m = nn.Linear(in_size, num_classes)
                self.add_module("classifier_l_"+name, m)

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

    def forward(self, x, logits=None):
        """
        x : 
            dictionary contain the features maps which 
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        if self.fpn_size is None:
            logits = {}
        selections = {}
        preds_1 = {}
        preds_0 = {}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])
            # s_ids, not_s_ids = self.select(logits[name], name)
            ### torch.gather may cause unsatble training
            # selections[name] = torch.gather(x[name], 1, s_ids.repeat(1, 1, C), sparse_grad=True)
            # logits["select_"+name] = torch.gather(logits[name], 1, s_ids.repeat(1, 1, logits[name].size(-1)), sparse_grad=True)
            # logits["drop_"+name] = torch.gather(logits[name], 1, not_s_ids.repeat(1, 1, logits[name].size(-1)), sparse_grad=True) # sparse_grad=True
            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1[name] = []
            preds_0[name] = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                selections[name].append(sf) # [num_selected, C]
                preds_1[name].append(logits[name][bi][ranks[:num_select]])
                preds_0[name].append(logits[name][bi][ranks[num_select:]])
            
            selections[name] = torch.stack(selections[name])
            preds_1[name] = torch.stack(preds_1[name])
            preds_0[name] = torch.stack(preds_0[name])
            
        return selections


class FPN(nn.Module):
    
    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn 
        proj_type: 
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'. 
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        assert proj_type in ["Conv", "Linear"], \
            "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], \
            "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(proj_type)

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_"+node_name, m)

            ### upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3 # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i-1]].size(1)
                if in_dim != out_dim:
                    m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                else:
                    m = nn.Identity()
                self.add_module("Up_"+node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            x1 = getattr(self, "Up_"+x1_name)(x1)
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []
        for i, name in enumerate(x):
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            x[x0_name] = self.upsample_add(x[x0_name], 
                                           x[x1_name], 
                                           x1_name)
        return x


class PluginMoodel(nn.Module):

    def __init__(self, 
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 coarse_num_classes: int,
                 num_selects: dict, 
                 use_combiner: bool,
                 comb_proj_size: Union[int, None]
                 ):
        """
        * backbone: 
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes: 
            tuple or list contain features map size of each layers. 
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn: 
            boolean, use features pyramid network or not
        * fpn_size: 
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }

        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which 
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer 
        could be different).
        """
        super(PluginMoodel, self).__init__()
        
        ### = = = = = Backbone = = = = =
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        else:
            self.backbone = backbone
        
        ### get hidden feartues size
        rand_in = torch.randn(1, 3, img_size, img_size)
        # outs = self.backbone(rand_in)
        out = self.backbone(rand_in)
        if type(out) == list:
            outs = {}
            outs['layer1'] = out[0]
            outs['layer2'] = out[1]
            outs['layer3'] = out[2]
            outs['layer4'] = out[3]
        else:
            outs = out

        ### just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size[-1]
                elif len(fs_size) == 4:
                    out_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

        ### = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = self.fpn_size if self.use_fpn else None # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = outs, comb_proj_size # redundant, fix in future
                gcn_proj_size = 1536
            total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
            # self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)
            self.combiner = NEWCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)

        ### Stage2
        self.scm_module = SCMModule(outs, num_classes, coarse_num_classes, 64)
        self.sce_module = SCEModule()
        if self.use_combiner:
            self.combiner = SCCombiner(outs, total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)
     
    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, num_classes, 1)
                )
            self.add_module("fpn_classifier_"+name, m)

    def forward_backbone(self, x):

        out = self.backbone(x)
        if type(out) == list:
            outs = {}
            outs['layer1'] = out[0]
            outs['layer2'] = out[1]
            outs['layer3'] = out[2]
            outs['layer4'] = out[3]
        else:
            outs = out

        # return self.backbone(x)
        return outs

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_"+name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose

    def forward(self, x: torch.Tensor):

        logits = {}

        x = self.forward_backbone(x)
        # print('Backbone Feature Size')
        # for k,v in x.items():
        #     print(k)
        #     print(v.shape)
        # print('\n')
        # exit()

        if self.use_fpn:
            x = self.fpn(x)
            self.fpn_predict(x, logits)
        #     print('FPN Feature x Size')
        #     for k,v in x.items():
        #         print(k)
        #         print(v.shape)
        #     print('FPN Feature logits Size')
        #     for k,v in logits.items():
        #         print(k)
        #         print(v.shape)
        #     print('\n')
        # exit()
        
            # Semantic concept modeling
            output1, constraint1 = self.scm_module(x['layer4'])
        
            # Semantic concept embedding
            output2 = self.sce_module(x, output1)

        # if self.use_selection:
        #     # selects = self.selector(x, logits)
        #     selects = self.selector(output2, logits)
        #     # print('Selection Feature Size')
        #     # for k,v in selects.items():
        #     #     print(k)
        #     #     print(v.shape)
        #     # print('\n')
        #     # exit()

        if self.use_combiner:
            # comb_outs, constraint2 = self.combiner(selects)
            comb_outs, constraint2 = self.combiner(output2)
            logits['comb_outs'] = comb_outs
            logits['constraint1'] = constraint1
            logits['constraint2'] = constraint2
            # print('Combiner Feature comb_outs Size')
            # print(comb_outs.shape)
            # print('Combiner Final logits Size')
            # for k,v in logits.items():
            #     print(k)
            #     print(v.shape)
            # print('\n')
            # exit()
            return logits
        
        if self.use_selection or self.use_fpn:
            return logits

        ### original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits['ori_out'] = out

        return logits

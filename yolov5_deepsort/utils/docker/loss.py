# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F
from utils.metrics import box_iou
from utils.general import xywh2xyxy



def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # 平衡正负样本 y=1时为正样本，alpha_factor=0.25 y=0为负样本
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        # 控制容易分类和难分类样本的权重，容易分类权重较小 难分类权重较大
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        """
        定义分类损失和置信度损失为带sigmoid的二值交叉熵损失，
        即会先将输入进行sigmoid再计算BinaryCrossEntropyLoss(BCELoss)。
        pos_weight：参数是正样本损失的权重参数。避免由于正负样本样本不均衡带来的问题
        """
        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))   # pos_weight是设置正负样本
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        '''
        对标签做平滑,eps=0就代表不做标签平滑,那么默认cp=1,cn=0
        后续对正类别赋值cp，负类别赋值cn
        '''
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # (0,1) --->(0.05,0.95)
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        '''
        超参设置g>0则计算FocalLoss
        '''
        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        '''
        使用1x1卷积获取detect层 输出的值与类别数量有关
        '''
        m = de_parallel(model).model[-1]  # Detect() module
        '''
        每一层预测值所占的权重比，分别代表浅层到深层，小特征到大特征，4.0对应着P3，1.0对应P4,0.4对应P5。
        如果__len__ = {int} 3是自己设置的输出不是3层，则返回[4.0, 1.0, 0.25, 0.06, 0.02]，可对应1-5个输出层P3-P7的情况。
        如果输出是3层，则 4.0*小目标权重 1.0*中目标权重 0.25*大目标权重
        '''
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        '''
        autobalance 默认为 False，yolov5中目前也没有使用 ssi = 0即可
        '''
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors   每个网格有3个anchor
        self.nc = m.nc  # number of classes   数据集类别个数
        self.nl = m.nl  # number of layers    检测层数量
        self.anchors = m.anchors  # [3 3 2] 代表有3个不同的检测层(以416输入为例 分别为 13×13 26×26 52×52) 每个检测层有三个不同尺度的anchor 2 anchor的w和h值
        self.device = device

    def __call__(self, p, targets):  # predictions, targets  p:预测值 targets:真实值
        # 1. 初始化损失函数
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        # 2. tcsl:目标类别 tbox：真实框（x,y,w,h） indices:目标中心所在左网格坐标 indices = [image, anchor, gridy, gridx]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions  p为检测层
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # 将所有anchor初始化为负样本
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # (x,y,w,h,conf,cls1,cls2)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # iou = bbox_iou(pbox, tbox[i], WIoU=True, scale=True)
                # if type(iou) is tuple:
                #     if len(iou) == 2:
                #         lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
                #         iou = iou[0].squeeze()
                #     else:
                #         lbox += (iou[0] * iou[1]).mean()
                #         iou = iou[2].squeeze()
                # else:
                #     lbox += (1.0 - iou.squeeze()).mean()  # iou loss
                #     iou = iou.squeeze()

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # 如果检测目标比较密集 使sort_obj_iou=True效果较好
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # 为compute_loss()构建目标，输入目标(image,class,x,y,w,h)
        '''
        na = 3,表示每个预测层anchors的个数
        targets 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h
        targets = [[image1,class1,x1,y1,w1,h1],
                   [image2,class2,x2,y2,w2,h2],
                   ...
                   [imageN,classN,xN,yN,wN,hN]]
        nt为一个batch中所有标签的数量
        '''
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        '''
        gain是为了最终将坐标所属grid坐标限制在坐标系内，不要超出范围,
        其中7是为了对应: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny,
        nx和ny为当前输出层的grid大小。
        '''
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        '''
        ai.shape = [na,nt]
        ai = [[0,0,0,.....],
              [1,1,1,...],
              [2,2,2,...]]
        这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引         ai ：是为了便于所引哪个检测层
        '''
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],.....],
                            [[1],[1],[1],...],
                              [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7]
        targets = [[[image1,class1,x1,y1,w1,h1,0],
                    [image2,class2,x2,y2,w2,h2,0],
                    ...
                    [imageN,classN,xN,yN,wN,hN,0]],
                    [[image1,class1,x1,y1,w1,h1,1],
                     [image2,class2,x2,y2,w2,h2,1],
                    ...],
                    [[image1,class1,x1,y1,w1,h1,2],
                     [image2,class2,x2,y2,w2,h2,2],
                    ...]]
        这么做是为了纪录每个label对应的anchor。
        '''
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        '''
        定义每个grid偏移量，会根据标签在grid中的相对位置来进行偏移
        '''
        g = 0.5  # bias
        '''
        [0, 0]代表中间,
        [1, 0] * g = [0.5, 0]代表往左偏移半个grid， [0, 1]*0.5 = [0, 0.5]代表往上偏移半个grid，与后面代码的j,k对应
        [-1, 0] * g = [-0.5, 0]代代表往右偏移半个grid， [0, -1]*0.5 = [0, -0.5]代表往下偏移半个grid，与后面代码的l,m对应
        '''
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            '''
            原本yaml中加载的anchors.shape = [3,6],但在yolo.py的Detect中已经通过代码
            a = torch.tensor(anchors).float().view(self.nl, -1, 2)
            self.register_buffer('anchors', a) 
            将anchors进行了reshape。
            self.anchors.shape = [3,3,2]
            anchors.shape = [3,2]
            '''
            anchors, shape = self.anchors[i], p[i].shape
            '''
            p.shape = [nl,bs,na,nx,ny,no]
            p[i].shape = [bs,na,nx,ny,no]
            gain = [1,1,nx,ny,nx,ny,1]
            '''
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            '''
            因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
            需要将其映射到当前输出层w = nx, h = ny的坐标系中。即输出的特征图上
            '''
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                '''
                t[:, :, 4:6].shape = [na,nt,2] = [3,nt,2],存放的是标签的w和h
                anchor[:,None] = [3,1,2]
                r.shape = [3,nt,2],存放的是标签和当前层anchor的长宽比
                '''
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                '''
                torch.max(r, 1. / r)求出最大的宽比和最大的长比，shape = [3,nt,2]
                再max(2)求出同一标签中宽比和长比较大的一个，shape = [2，3,nt],之所以第一个维度变成2，
                因为torch.max如果不是比较两个tensor的大小，而是比较1个tensor某一维度的大小，则会返回values和indices：
                    torch.return_types.max(
                        values=tensor([...]),
                        indices=tensor([...]))
                所以还需要加上索引0获取values，
                torch.max(r, 1. / r).max(2)[0].shape = [3,nt],
                将其和hyp.yaml中的anchor_t超参比较，小于该值则认为标签属于当前输出层的anchor
                j = [[bool,bool,....],[bool,bool,...],[bool,bool,...]]
                j.shape = [3,nt]                # 真实标签与对使用聚类方法生成的anchors长宽比最大值不超过 '4' 或者 '0.25' # 超过该值则认为该物体不在当前的gt中
                '''
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                '''
                t.shape = [na,nt,7] 
                j.shape = [3,nt]
                假设j中有NTrue个True值，则
                t[j].shape = [NTrue,7]
                返回的是na*nt的标签中，所有属于当前层anchor的标签。
                '''
                t = t[j]  # filter

                # Offsets
                '''
                下面这段代码和注释可以配合代码后的图片进行理解。
                t.shape = [NTrue,7] 
                7:image,class,x,y,h,w,ai
                gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的记录
                gxi.shape = [NTrue,2] 存放的是w-x,h-y,相当于测量坐标到坐标系右边框和下边框的距离
                '''
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                '''
                因为grid单位为1，共nx*ny个gird
                gxy % 1相当于求得标签在第gxy.long()个grid中以grid左上角为原点的相对坐标，
                gxi % 1相当于求得标签在第gxy.long()个grid中以grid右下角为原点的相对坐标，
                下面这两行代码作用在于
                筛选中心坐标 左、上方偏移量小于0.5,并且中心点大于1的标签
                筛选中心坐标 右、下方偏移量小于0.5,并且中心点大于1的标签          
                j.shape = [NTrue], j = [bool,bool,...]
                k.shape = [NTrue], k = [bool,bool,...]
                l.shape = [NTrue], l = [bool,bool,...]
                m.shape = [NTrue], m = [bool,bool,...]   判断离那块近 得到最近的三个块
                '''
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                '''
                j.shape = [5,NTrue]
                t.repeat之后shape为[5,NTrue,7], 
                通过索引j后t.shape = [NOff,7],NOff表示NTrue + (j,k,l,m中True的总数量)
                torch.zeros_like(gxy)[None].shape = [1,NTrue,2]
                off[:, None].shape = [5,1,2]
                相加之和shape = [5,NTrue,2]
                通过索引j后offsets.shape = [NOff,2]
                这段代码的表示当标签在grid左侧半部分时，会将标签往左偏移0.5个grid，上下右同理。
                '''
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            # Define
            '''
            t.shape = [NOff,7],(image,class,x,y,w,h,ai)
            '''
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            '''
            offsets.shape = [NOff,2]
            gxy - offsets为gxy偏移后的坐标，
            gxi通过long()得到偏移后坐标所在的grid坐标
            '''
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices
            # Append
            '''
            a:所有anchor的索引 shape = [NOff]
            b:标签所属image的索引 shape = [NOff]
            gj.clamp_(0, gain[3] - 1)将标签所在grid的y限定在0到ny-1之间
            gi.clamp_(0, gain[2] - 1)将标签所在grid的x限定在0到nx-1之间
            indices = [image, anchor, gridy, gridx] 最终shape = [nl,4,NOff]
            tbox存放的是标签在所在grid内的相对坐标，∈[0,1] 最终shape = [nl,NOff]
            anch存放的是anchors 最终shape = [nl,NOff,2]
            tcls存放的是标签的分类 最终shape = [nl,NOff]
            '''
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox, selected_tbox, CIoU=True)  # iou(prediction, target)
                if type(iou) is tuple:
                    lbox += (iou[1].detach() * (1 - iou[0])).mean()
                    iou = iou[0]
                else:
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets, imgs):
        indices, anch = self.find_3_positive(p, targets)
        device = torch.device(targets.device)
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost, device=device)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
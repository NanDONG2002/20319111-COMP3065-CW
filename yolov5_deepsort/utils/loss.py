# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


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
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
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

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        # 初始化损失函数
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
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
        """
        Args:
            p: p.shape = [nl,bs,na,nx,ny,no] = [输出层个数，Batch_size, 每个输出层对应的anchors的个数，\
             输出层x, 输出层y, 要预测的参数包括（x,y,w,h,clc）]
            targets: 为一个batch中所有的标签，包括标签所属的image，以及class,x,y,w,h

        Returns:
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
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
        tcls, tbox, indices, anch = [], [], [], []
        '''
        gain是为了最终将坐标所属grid坐标限制在坐标系内，不要超出范围,
        其中7是为了对应: image class x y w h ai,
        但后续代码只对x y w h赋值，x,y,w,h = nx,ny,nx,ny,
        nx和ny为当前输出层的grid大小。
        '''
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain  原标签 + 框ID
        '''
        ai.shape = [na,nt]
        ai = [[0,0,0,.....],
              [1,1,1,...],
              [2,2,2,...]]
        这么做的目的是为了给targets增加一个属性，即当前标签所属的anchor索引
        '''
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        '''
        targets.repeat(na, 1, 1).shape = [na,nt,6]
        ai[:, :, None].shape = [na,nt,1](None在list中的作用就是在插入维度1)
        ai[:, :, None] = [[[0],[0],[0],.....],
                            [[1],[1],[1],...],
                              [[2],[2],[2],...]]
        cat之后：
        targets.shape = [na,nt,7] = [3, nt, 7]
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
        具体原理在代码后讲述
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

        for i in range(self.nl):    # nl表示输出层的个数
            """
            self.anchors.shape = [3, 3, 2] First 3：代表检测层的数量  Second 3：代表anchor数量  2：长和宽
            anchors.shape = [3,2]
            p.shape = [nl,bs,na,nx,ny,no]
            p[i]：表示第几个输出层 
            p[i].shape = [bs,na,nx,ny,no]
            """
            anchors, shape = self.anchors[i], p[i].shape
            """
            gain = [1,1,nx,ny,nx,ny,1]  防止出界
            """
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            '''
            因为targets进行了归一化，默认在w = 1, h =1 的坐标系中，
            需要将其映射到当前输出层w = nx, h = ny的坐标系中。
            '''
            t = targets * gain  # shape(3,n,7)
            if nt:    # 标签的数量不为0
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
                j.shape = [3,nt]
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
                t.shape = [NTrue,7] 
                7:image,class,x,y,h,w,ai
                gxy.shape = [NTrue,2] 存放的是x,y,相当于坐标到坐标系左边框和上边框的距离
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
                m.shape = [NTrue], m = [bool,bool,...]
                '''
                j, k = ((gxy % 1 < g) & (gxy > 1)).T   # 挑选坐标到坐标系左边框和上边框的距离近的
                l, m = ((gxi % 1 < g) & (gxi > 1)).T   # 挑选坐标到坐标系右边框和下边框的距离近的
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

import torch
import torch.nn as nn

#############
# classes
#############

class LabelLoss(nn.Module):

    def __init__(self, loss_name='LSGAN', soft_label=True):
        super(LabelLoss, self).__init__()
        self.soft_label = soft_label
        if loss_name == 'LSGAN':
            self.loss = nn.MSELoss()
        elif loss_name == 'GAN':
            self.loss = nn.BCELoss()
            self.soft_label = False
            print('WARNING! Soft label not supported in BCE loss.')
        else:
            raise NotImplementedError('Objective [%s] is not implemented' % loss_name)

    def get_label(self, logits, isreal, soft_range):
        if isreal:
            out = torch.ones_like(logits)
        else:
            out = torch.zeros_like(logits)

        return out.uniform_(*soft_range) if self.soft_label else out

    def __call__(self):
        pass

class G_Label_Loss(LabelLoss):
    
    def __init__(self, loss_name='LSGAN', soft_label=True):
        super(G_Label_Loss, self).__init__(loss_name=loss_name, soft_label=soft_label)

    def __call__(self, logits, soft_real=[0.78, 1.12]):
        label = self.get_label(logits=logits, isreal=True, soft_range=soft_real)
        return self.loss(logits, label)

class D_Label_Loss(LabelLoss):
    
    def __init__(self, loss_name='LSGAN', soft_label=True):
        super(D_Label_Loss, self).__init__(loss_name=loss_name, soft_label=soft_label)

    def __call__(self, logits_real, logits_fake, soft_real=[0.78, 1.12], soft_fake=[0.0, 0.24]):
        label_real = self.get_label(logits=logits_real, isreal=True, soft_range=soft_real)
        label_fake = self.get_label(logits=logits_fake, isreal=False, soft_range=soft_fake)
        return self.loss(logits_real, label_real) + self.loss(logits_fake, label_fake)
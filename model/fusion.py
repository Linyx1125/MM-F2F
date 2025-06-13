import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self):
        super(LMF, self).__init__()

        self.hidden_dim = 256
        self.output_dim = 3
        self.rank = 16
        self.use_softmax = False

        self.post_fusion_prob = 0.1

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.factor_1 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.factor_2 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.factor_3 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.factor_1)
        xavier_normal_(self.factor_2)
        xavier_normal_(self.factor_3)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, text_x, audio_x, video_x):
        temp_x = text_x if text_x is not None else audio_x
        batch_size = temp_x.data.shape[0]
        if temp_x.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        if text_x is not None:
            _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_x), dim=1)
            fusion_text = torch.matmul(_text_h, self.factor_1)
        if audio_x is not None:
            _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_x), dim=1)
            fusion_audio = torch.matmul(_audio_h, self.factor_2)
        if video_x is not None:
            _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_x), dim=1)
            fusion_video = torch.matmul(_video_h, self.factor_3)

        if text_x is None:
            fusion_zy = fusion_audio * fusion_video
        elif audio_x is None:
            fusion_zy = fusion_text * fusion_video
        elif video_x is None:
            fusion_zy = fusion_audio * fusion_text
        else:
            fusion_zy = fusion_audio * fusion_video * fusion_text

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output


if __name__ == "__main__":
    lmf = LMF()

    batch_size = 16
    Xt = torch.randn(batch_size, 256)
    Xa = torch.randn(batch_size, 256)
    Xv = torch.randn(batch_size, 256)

    print(lmf(Xa, Xv, Xt).shape)
    print(lmf(None, Xv, Xt).shape)
    print(lmf(Xa, None, Xt).shape)
    print(lmf(Xa, Xv, None).shape)
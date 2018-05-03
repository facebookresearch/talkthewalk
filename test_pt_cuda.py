import torch

from torch.autograd import Variable

a = Variable(torch.FloatTensor(10, 10, 10)).cuda()
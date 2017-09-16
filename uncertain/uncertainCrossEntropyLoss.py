import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss, CrossEntropyLoss
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad
import torch

class UncertainCrossEntropyLoss(_WeightedLoss):
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a classification problem with `n` classes.
    If provided, the optional argument `weights` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 2D `Tensor` of size `batch x n`.

    The `target` is expected to contain what probability each training example
    is of being each class. Each element should be a weight between 0 and 1
    such that the weights for each row sum to 1. Each index in each row
    should correspond to each classes index.

    `target` has to be a 2D 'Tensor' of floats of size `batch x n`

    The loss can be described as::

        loss(x, classes) = \sum_{c \in classes} (y[c] * -log(exp(x[c]) / (\sum_j exp(x[j]))))
                       =

    or in the case of the `weights` argument being specified::

        loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to False, the losses are
           instead summed for each minibatch.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`

    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, use_gpu, weight=None, size_average=True, ignore_index=-100):
        super(UncertainCrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.use_gpu = use_gpu

    def forward(self, input, trueDistributions):
        _assert_no_grad(trueDistributions)
        logSoftmaxesForAll = F.log_softmax(input)
        # elmentwise multiply every logsoft_max by the true distribution for its example to get
        # the cross-entropy
        unsumedCrossEntropy = logSoftmaxesForAll * trueDistributions
        return torch.sum(unsumedCrossEntropy) * -1 / input.size(0)
        # sum all, and let nll_loss make negative
        # cross entropy function just calls nll_loss(log_softmax(input), target, weight, size_average, ignore_index)
        # nll_loss just calls for matrices of size 2
        # _functions.thnn.NLLLoss.apply(input, target, weight, size_average, ignore_index)
        # use 0 for size as want to get a 0 for every row
        nonCudaTarget = autograd.Variable(torch.LongTensor([0] * input.size(0)))
        withRightCudaTarget = nonCudaTarget.cuda() if self.use_gpu else nonCudaTarget
        return F.nll_loss(torch.sum(unsumedCrossEntropy, dim = 1), withRightCudaTarget)

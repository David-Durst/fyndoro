import unittest
import torch
import torch.nn as nn
import torch.autograd as autograd
from uncertain import uncertainCrossEntropyLoss


class TestUncertainCrossEntropyLoss(unittest.TestCase):

    def forwardTestNExamples(self, numExamples, testName):
        input = autograd.Variable(torch.randn(numExamples, 2), requires_grad=True)
        all0target = autograd.Variable(torch.LongTensor([0] * numExamples))
        all1target = autograd.Variable(torch.LongTensor([1] * numExamples))
        all0probabilites = autograd.Variable(torch.FloatTensor([[1, 0]] * numExamples))
        all1probabilites = autograd.Variable(torch.FloatTensor([[0, 1]] * numExamples))
        halfhalfprobabilites = autograd.Variable(torch.FloatTensor([[0.5, 0.5]] * numExamples))
        certainCrossEntropyLoss = nn.CrossEntropyLoss()
        uncertainLoss = uncertainCrossEntropyLoss.UncertainCrossEntropyLoss(False)
        certainAll0 = certainCrossEntropyLoss(input, all0target).data[0]
        certainAll1 = certainCrossEntropyLoss(input, all1target).data[0]

        self.assertAlmostEqual(certainAll0, uncertainLoss(input, all0probabilites).data[0], places=3)
        self.assertAlmostEqual(certainAll1, uncertainLoss(input, all1probabilites).data[0], places=3)
        self.assertAlmostEqual((certainAll0 + certainAll1) / 2, uncertainLoss(input, halfhalfprobabilites).data[0], places=3)

    def test_forwardCorrectLossOneEx(self):
        self.forwardTestNExamples(1, "test_forwardCorrectLossOneEx")

    def test_forwardCorrectLossMultiEx(self):
        self.forwardTestNExamples(7, "test_forwardCorrectLossMultiEx")

if __name__ == '__main__':
    unittest.main()
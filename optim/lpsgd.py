from torch.optim.optimizer import Optimizer, required
import torch
from torch.autograd import Variable
import copy, logging
# import sys
# sys.path.append('..')
from .. import quantize


class LPSGD(torch.optim.SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, scale_factor=0.0001, bits=None):
        super(LPSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                    weight_decay=weight_decay, nesterov=nesterov)
        self._bits = bits
        self._param_list = []
        for group in self.param_groups:
            for p in group["params"]:
                self._param_list.append(p)
        if isinstance(scale_factor, list):
            self._scale_factor = scale_factor
            if len(self._scale_factor) != sum(1 for _ in self._param_list):
                raise Exception("# of (list style) scale factor need to be the same as the # of variables")
        else:
            self._scale_factor = [scale_factor] * sum(1 for _ in self._param_list)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        if self._bits is not None:
            for scale, param in zip(self._scale_factor, self._param_list):
                param.data.quantize_(scale, self._bits)
        return loss     






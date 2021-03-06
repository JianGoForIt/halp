from torch.optim.optimizer import Optimizer, required
import torch
from torch.autograd import Variable
import copy, logging
import math
import numpy as np

from .. import quantize

class BitCenterLMSGD(torch.optim.SGD):
    """Implements high-accuracy low-precision algorithm for linear models.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        T (int): number of iterations between the step to take the full grad/save w
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum (float, optional): momentum (default: 0)
        opt (torch.optim): optimizer to baseclass (default: SGD)
        mu (float, optional): mu hyperparameter for HALP algorithm (default: 0.1)
        bits (int, optional): number of bits to use for offset (default: 8)
        biased (bool, optional): type of rounding to use for quantization (default: unbiased)
    """

    def __init__(self, params, lr=required, T=required, data_loader=required,
                 weight_decay=0.0, momentum=0.0, opt=torch.optim.SGD, mu=1e-1, bits=8, 
                 biased=False, data_scale=0.01):

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Choose the baseclass dynamically
        self.__class__ = type(self.__class__.__name__,
                              (opt,object),
                              dict(self.__class__.__dict__))
        logging.info("Using base optimizer {} in HALP".format(opt))
        super(self.__class__, self).__init__(params, **defaults)

        if len(self.param_groups) != 1:
            raise ValueError("HALP doesn't support per-parameter options "
                             "(parameter groups)")

        if bits <= 1:
            raise ValueError("HALP requires > 1 bit.")

        params = self.param_groups[0]['params']
        self._params = params

        self._curr_w = [p.data for p in params]
        self._z = [p.data.clone() for p in params]
        self._prev_w = [p.data.clone() for p in params]

        # Gradients are lazily allocated and don't exist yet. However, gradients are
        # the same shape as the weights so we can still allocate buffers here
        self._curr_grad = [p.data.clone() for p in params]
        self._prev_grad = [p.data.clone() for p in params]
        self._full_grad = None

        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T # Needed to trigger full gradient
        logging.info("Data Loader has {} with batch {}".format(len(self.data_loader),
                                                               self.data_loader.batch_size))
        # record the scale of input feature data
        self._data_scale = data_scale
        # Separate scale factor for each layer
        self._scale_factors = [1 for p in params]
        self._scale_factors_i = [1 for p in params]
        self._scale_factors_s = [1 for p in params]
        self._bits = bits
        self._mu = mu
        self._biased = biased

    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.data.zero_()

    def _set_weights_grad(self,ws,gs):
        """ Set the pointers in params to ws and gs for p.data and p.grad.data
        respectively. This allows us to avoid copying data in and out of parameters.
        """
        for idx, p in enumerate(self._params):
            if ws is not None: p.data = ws[idx]
            if gs is not None and p.grad is not None:
                p.grad.data = gs[idx]
                assert (p.grad.data.data_ptr() == gs[idx].data_ptr())

    def _rescale(self):
        """Update scale factors for z."""
        div_factor = math.pow(2.0, self._bits-1) - 1
        for i, fg in enumerate(self._full_grad):
            self._scale_factors[i] = fg.norm() / (self._mu * div_factor)
            self._scale_factors_i[i] = self._scale_factors[i] / float(2**self._bits)
            self._scale_factors_s[i] = self._scale_factors[i] / float(2**self._bits) / self._data_scale

    def _reset_z(self):
        """Set z to zero."""
        for p in self._z:
            p.fill_(0)

    def _recenter(self, ws):
        """Add the values in self._z to ws."""
        for w, z in zip(ws, self._z):
            w.add_(z)

    def _compute_full_grad(self, closure):
        """ Call the closure function to compute the gradient
        over the entire dataset, and accumulate the gradient into
        self._full_grad.
        """

        # Set up pointers for the full gradient
        # Reset gradients before accumulating them
        self._set_weights_grad(self._prev_w, self._full_grad)
        self._zero_grad()

        # Accumulate gradients
        for i, (data, target) in enumerate(self.data_loader):
            closure(data, target)

        # Adjust summed gradients by num_iterations accumulated over
        # Assumes loss size average argument is true
        for p in self._params:
            if p.grad is not None:
                p.grad.data /= len(self.data_loader)

        # Since p.grad is dynamically allocated, the pointers to the gradients won't
        # be set before backward is called the first time
        if self._full_grad is None:
            self._full_grad = [p.grad.data.clone() for p in self._params]

    def _quantize_full_grad(self):
        self._alpha_full_grad = []
        for p, sf in zip(self._full_grad, self._scale_factors_i):
            lr = self.param_groups[0]['lr']
            self._alpha_full_grad.append(lr * p.clone() )
            # print("full gradient quant ", sf, 2 * self._bits)
            self._alpha_full_grad[-1].quantize_(sf, 2 * self._bits, biased=self._biased) 

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and     returns the loss.
        """
        assert len(self.param_groups) == 1

        # Calculate full gradient
        if self.state['t_iters'] == self.T:
            self._compute_full_grad(closure)
            self._rescale()
            self._quantize_full_grad()
            self._reset_z()
            # Reset t
            self.state['t_iters'] = 0
            # TODO: Update the other couple of delta values

        # Calculate gradient of prev_w
        self._set_weights_grad(self._prev_w, self._prev_grad)
        self._zero_grad()

        # Get the intermediate gradient out
        # X is the RFF
        _, X, l_prime_prev = closure()

        # Calculate the current curr_w (which equals prev_w + z)
        self._set_weights_grad(self._curr_w, self._curr_grad)
        self._zero_grad()
        # Get the intermediate gradient out
        # We pass in the quantized RFF in the closure call for model prev
        # so that the quantized feature are sync in the two gradient calculation
        loss, X, l_prime_curr = closure(feat=X)

        # Adjust the current gradient using the previous gradient and the full gradient.
        # TODO: change the way the gradient is calculated, and adapt to low precision
        # for i, p in enumerate(self._params):
        #     # Adjust gradient in-place
        #     if p.grad is not None:
        #         # gradient_update = curr_grad - prev_grad + full_grad
        #         p.grad.data -= (self._prev_grad[i] - self._full_grad[i])

        for i, p in enumerate(self._params):
            # Adjust gradient in-place
            # print "before ", p.grad.data
            if p.grad is not None:
                lr = self.param_groups[0]['lr']
                beta = lr * l_prime_curr
                #beta = lr * (l_prime_curr - l_prime_prev)
                # the next line below can be used to test whether the intermediate grad based 
                # gradient calculation is correct or not
                # beta = lr * l_prime_curr
                # TODO quantize gamma
                gamma = beta.data.clone()
                if self.state['t_iters'] == 0:
                    print("data quant ", self._data_scale, self._bits)
                    print("beta quant ", self._scale_factors_s[i], self._bits)
                gamma.quantize_(self._scale_factors_s[i], self._bits, biased=self._biased)
                if len(list(p.data.size() ) ) == 2:
                    #assert np.allclose(torch.mm(l_prime_curr.data.transpose(0, 1), X.data).cpu().numpy(), p.grad.data.cpu().numpy() )
                    #assert np.allclose(torch.mm(l_prime_prev.data.transpose(0, 1), X.data).cpu().numpy(), self._prev_grad[i].cpu().numpy() )
                    # this is the weight matrix
                    p.grad.data.copy_(torch.mm(gamma.transpose(0, 1), X.data) )
                    #p.grad.data.copy_(torch.mm(gamma.transpose(0, 1), X.data) + self._alpha_full_grad[i] )
                    # the next line below can be used to test whether the intermediate grad based 
                    # gradient calculation is correct or not
                    # p.grad.data.copy_(torch.mm(gamma.data.transpose(0, 1), X.data) )
                else:
                    # this is the bias vector
                    n_sample = gamma.size(0)
                    #assert np.allclose(torch.squeeze(torch.mm(l_prime_curr.data.transpose(0, 1), 
                    #    torch.ones(n_sample, 1).type(type(l_prime_curr.data) ) ) ).cpu().numpy(), p.grad.data.cpu().numpy() )
                    #assert np.allclose(torch.squeeze(torch.mm(l_prime_prev.data.transpose(0, 1), 
                    #    torch.ones(n_sample, 1).type(type(l_prime_prev.data) ) ) ).cpu().numpy(), self._prev_grad[i].cpu().numpy() )
                    p.grad.data.copy_(torch.squeeze(torch.mm(gamma.transpose(0, 1),
                        torch.ones(n_sample, 1).type(type(gamma) ) ) ) )
                    #p.grad.data.copy_(torch.squeeze(torch.mm(gamma.transpose(0, 1), 
                    #    torch.ones(n_sample, 1).type(type(gamma) ) ) ) + self._alpha_full_grad[i] )
                    # the next line below can be used to test whether the intermediate grad based 
                    # gradient calculation is correct or not
                    # p.grad.data.copy_(torch.squeeze(torch.mm(gamma.data.transpose(0, 1), 
                    #     torch.ones(n_sample, 1).type(type(gamma.data) ) ) ) )
                # to adapt the quantized update to the operation of optimizer,
                # we divide the grad by lr. It is multiplied back in optimizer step.
                p.grad.data /= lr
                # gradient_update = curr_grad - prev_grad + full_grad
                # p.grad.data -= (self._prev_grad[i] - self._full_grad[i] )

        # Set the param pointers to z to update z with step
        self._set_weights_grad(self._z, None)
        # Call optimizer update step
        super(self.__class__, self).step()

        # Quantize z in place
        for p, sf in zip(self._z, self._scale_factors):
            if self.state['t_iters'] == 0:
                print("z quant ", sf, self._bits)
            p.quantize_(sf, self._bits, biased=self._biased)

        # Increment "inner loop" counter
        self.state['t_iters'] += 1

        # Set curr_w to prev_w + z
        for p, p0 in zip(self._curr_w, self._prev_w):
            p.copy_(p0)
        self._recenter(self._curr_w)
        # Update param pointers to curr_w for user access
        self._set_weights_grad(self._curr_w, self._curr_grad)

        # Update prev_w to prev_w + z after the "inner loop" has finished
        if self.state['t_iters'] == self.T:
            self._recenter(self._prev_w)

        return loss

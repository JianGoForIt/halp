High-Accuracy Low-Precision Training
====================================
This repo contains a PyTorch implementation of the HALP optimizer from the paper [High-Accuracy Low-Precision Training](https://arxiv.org/abs/1803.03383) and a SVRG optimizer.

This repo is duplicated from the source code of PyTorch-based HALP repo from [Megan Leszczynski](https://github.com/mleszczy) under permission. This repo is only used for the Low-precision Random Fourier Feature project.

For HALP repo with up-to-date maintainance, please refer to [HazyResearch/torchhalp](https://github.com/HazyResearch/torchhalp).

<!--### Getting Started

To run tests, run `cd test && pytest -v`.

To add the optimizers to your existing PyTorch code:

1. Import the optimizer
`from optim import HALP`
2. Change the optimizer to `optimizer = HALP(model.parameters(), lr=args.lr, T=T, data_loader=train_loader)`
3. Add a closure method which takes a datapoint and target, and recomputes the gradient.

```
def closure(data=data, target=target):
	data = Variable(data, requires_grad=False)
	target = Variable(target, requires_grad=False)
    if cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    return loss
 ```

###  Notes

* The SVRG and HALP optimizers take two additional arguments as compared to the SGD optimizer, `T` and `data_loader`. `T` indicates how often the full gradient over the entire dataset, a key step in the SVRG algorithm, is taken, where `T` is the number of weight updates in between updating the full gradient. The `data_loader` argument requires a PyTorch [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), such that the gradient over the full dataset can be initiated internally in the optimizer. The HALP optimizer has the additional arguments of `mu`, `bits`, and `unbiased` which affect the quantization, where `mu` contributes to the dynamic rescaling, `bits` is the number of bits used for the quantized numbers, and `unbiased` indicates stochastic rounding is used.

* Currently, the optimizer doesn’t support multiple per-parameter options and parameter groups.

* Due to the structure of the closure method and the optimizers self-containing the copying structure of SVRG and HALP, stateful LSTMs in which we reuse the hidden layer across batches are not currently supported. However, we can still use learned hidden layers or stateless LSTMs.
-->

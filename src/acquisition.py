
import math
import warnings
import gpytorch
import torch
import numpy as np
from torch import Tensor
from typing import Optional

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform



warnings.filterwarnings("ignore")

class qUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

        self.sampler = sampler
        self.beta = beta
        self.weights = weights

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        mean = posterior.mean  # b x q x o
        if self.weights is None:
            scalarized_samples = samples.mean(dim=-1)  # n x b x q
            scalarized_mean = mean.mean(dim=-1)  # b x q
        else:
            scalarized_samples = samples.matmul(self.weights)  # n x b x q
            scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            + math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


class qLowerConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

        self.sampler = sampler
        self.beta = beta
        self.weights = weights

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        mean = posterior.mean  # b x q x o
        if self.weights is None:
            scalarized_samples = samples.mean(dim=-1)  # n x b x q
            scalarized_mean = mean.mean(dim=-1)  # b x q
        else:
            scalarized_samples = samples.matmul(self.weights)  # n x b x q
            scalarized_mean = mean.matmul(self.weights)  # b x q
        lcb_samples = (
            scalarized_mean
            - math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return lcb_samples.min(dim=-1)[0].mean(dim=0)

class qConstrainedCI_m_UCB(MCAcquisitionFunction):

    def __init__(
        self,
        model_list: Model,
        threshold_list: Tensor,
        return_UCB: bool,
        beta: Tensor,
        filter_beta: Tensor = None,
        weights: Tensor  = None,
        sampler: Optional[MCSampler] = None,
        constrained: bool = True,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model_list[0])
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.model_list = model_list # the first model is the main model, the rest are the constraint models
        self.threshold_list = threshold_list
        self.beta = beta
        self.filter_beta = filter_beta if filter_beta is not None else beta
        self.weights = weights
        self.return_UCB = return_UCB
        self.constrained = constrained


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        # print(X.shape)
        lcb_list, ucb_list, filter_ucb_list, filter_lcb_list, ci_list = [], [], [], [], []
        for model in self.model_list:
            posterior = model.posterior(X)
            samples = self.get_posterior_samples(posterior)  # n x b x q x o
            mean = posterior.mean  # b x q x o
            if self.weights is None: # for multi-output models
                scalarized_samples = samples.mean(dim=-1)  # n x b x q
                scalarized_mean = mean.mean(dim=-1)  # b x q
            else:
                scalarized_samples = samples.matmul(self.weights)  # n x b x q
                scalarized_mean = mean.matmul(self.weights)  # b x q

            ucb_list.append(
                scalarized_mean
                + math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            lcb_list.append(
                scalarized_mean
                - math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

            filter_ucb_list.append(
                scalarized_mean
                + math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            filter_lcb_list.append(
                scalarized_mean
                - math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

        # constrained UCB
        if self.return_UCB:
            ucb = ucb_list[0].max(dim=-1)[0].mean(dim=0) - self.threshold_list[0]
            if self.constrained:
                _roi = torch.ones(ucb.shape)
                for idx in range(0, len(lcb_list)):
                    _roi = _roi * ((-filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0) + self.threshold_list[idx])>0)
                ucb = ucb * _roi 
        
        # constrained CI
        else:
            ci = torch.zeros(ucb_list[0].max(dim=-1)[0].mean(dim=0).shape)
            _roi = torch.ones(ci.shape)
            for idx in range(1, len(ucb_list)):
                ci, _ = (torch.max(torch.cat([ci.reshape(1,-1), self.threshold_list[idx] - lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1)],dim=0), dim=0))
                _roi = _roi * ((filter_ucb_list[idx].max(dim=-1)[0].mean(dim=0).reshape(1,-1) - self.threshold_list[idx])>0) * ((self.threshold_list[idx] - filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1))>0)
            if self.constrained:
                ci = ci * _roi
        
        if self.return_UCB:
            return ucb.squeeze()
        else:
            return ci.squeeze()

class qConfidenceInterval(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor = torch.ones(1),
        weights: Tensor = None,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

        self.sampler = sampler
        self.beta = beta
        self.weights = weights

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        mean = posterior.mean  # b x q x o
        if self.weights is None:
            scalarized_samples = samples.mean(dim=-1)  # n x b x q
            scalarized_mean = mean.mean(dim=-1)  # b x q
        else:
            scalarized_samples = samples.matmul(self.weights)  # n x b x q
            scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            - math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        lcb_samples = (
            scalarized_mean
            - math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return ucb_samples.max(dim=-1)[0].mean(dim=0) - lcb_samples.min(dim=-1)[0].mean(dim=0)

class qConstrainedCI_m_UCB(MCAcquisitionFunction):

    def __init__(
        self,
        model_list: Model,
        threshold_list: Tensor,
        return_UCB: bool,
        beta: Tensor,
        filter_beta: Tensor = None,
        weights: Tensor  = None,
        sampler: Optional[MCSampler] = None,
        constrained: bool = True,
        sample_num: int = 512,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model_list[0])
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([sample_num]))
        self.sampler = sampler
        self.model_list = model_list # the first model is the main model, the rest are the constraint models
        self.threshold_list = threshold_list
        self.beta = beta
        self.filter_beta = filter_beta if filter_beta is not None else beta
        self.weights = weights
        self.return_UCB = return_UCB
        self.constrained = constrained


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        # print(X.shape)
        lcb_list, ucb_list, filter_ucb_list, filter_lcb_list, ci_list = [], [], [], [], []
        for model in self.model_list:
            posterior = model.posterior(X)
            samples = self.get_posterior_samples(posterior)  # n x b x q x o
            mean = posterior.mean  # b x q x o
            if self.weights is None: # for multi-output models
                scalarized_samples = samples.mean(dim=-1)  # n x b x q
                scalarized_mean = mean.mean(dim=-1)  # b x q
            else:
                scalarized_samples = samples.matmul(self.weights)  # n x b x q
                scalarized_mean = mean.matmul(self.weights)  # b x q

            ucb_list.append(
                scalarized_mean
                + math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            lcb_list.append(
                scalarized_mean
                - math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

            filter_ucb_list.append(
                scalarized_mean
                + math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            filter_lcb_list.append(
                scalarized_mean
                - math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

        # constrained UCB
        if self.return_UCB:
            ucb = ucb_list[0].max(dim=-1)[0].mean(dim=0) - self.threshold_list[0]
            if self.constrained:
                _roi = torch.ones(ucb.shape)
                for idx in range(0, len(lcb_list)):
                    _roi = _roi * ((-filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0) + self.threshold_list[idx])>0)
                ucb = ucb * _roi 
        
        # constrained CI
        else:
            ci = torch.zeros(ucb_list[0].max(dim=-1)[0].mean(dim=0).shape)
            _roi = torch.ones(ci.shape)
            for idx in range(1, len(ucb_list)):
                ci, _ = (torch.max(torch.cat([ci.reshape(1,-1), self.threshold_list[idx] - lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1)],dim=0), dim=0))
                _roi = _roi * ((filter_ucb_list[idx].max(dim=-1)[0].mean(dim=0).reshape(1,-1) - self.threshold_list[idx])>0) * ((self.threshold_list[idx] - filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1))>0)
            if self.constrained:
                ci = ci * _roi
        
        if self.return_UCB:
            return ucb.squeeze()
        else:
            return ci.squeeze()

class MultiFidelityTS(MCAcquisitionFunction):
    '''
    Multi-fidelity TS, extending the TS to multi-fidelity setting
    '''
    def __init__(
        self,
        model_list: Model,
        threshold_list: Tensor,
        return_UCB: bool,
        beta: Tensor,
        filter_beta: Tensor = None,
        weights: Tensor  = None,
        sampler: Optional[MCSampler] = None,
        constrained: bool = True,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model_list[0])
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.model_list = model_list # the first model is the main model, the rest are the constraint models
        self.threshold_list = threshold_list
        self.beta = beta
        self.filter_beta = filter_beta if filter_beta is not None else beta
        self.weights = weights
        self.return_UCB = return_UCB
        self.constrained = constrained


    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        # print(X.shape)
        lcb_list, ucb_list, filter_ucb_list, filter_lcb_list, ci_list = [], [], [], [], []
        for model in self.model_list:
            posterior = model.posterior(X)
            samples = self.get_posterior_samples(posterior)  # n x b x q x o
            mean = posterior.mean  # b x q x o
            if self.weights is None: # for multi-output models
                scalarized_samples = samples.mean(dim=-1)  # n x b x q
                scalarized_mean = mean.mean(dim=-1)  # b x q
            else:
                scalarized_samples = samples.matmul(self.weights)  # n x b x q
                scalarized_mean = mean.matmul(self.weights)  # b x q

            ucb_list.append(
                scalarized_mean
                + math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            lcb_list.append(
                scalarized_mean
                - math.sqrt(self.beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

            filter_ucb_list.append(
                scalarized_mean
                + math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )
            filter_lcb_list.append(
                scalarized_mean
                - math.sqrt(self.filter_beta * math.pi / 2)
                * (scalarized_samples - scalarized_mean).abs()
            )

        # constrained UCB
        if self.return_UCB:
            ucb = ucb_list[0].max(dim=-1)[0].mean(dim=0) - self.threshold_list[0]
            if self.constrained:
                _roi = torch.ones(ucb.shape)
                for idx in range(0, len(lcb_list)):
                    _roi = _roi * ((-filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0) + self.threshold_list[idx])>0)
                ucb = ucb * _roi 
        
        # constrained CI
        else:
            ci = torch.zeros(ucb_list[0].max(dim=-1)[0].mean(dim=0).shape)
            _roi = torch.ones(ci.shape)
            for idx in range(1, len(ucb_list)):
                ci, _ = (torch.max(torch.cat([ci.reshape(1,-1), self.threshold_list[idx] - lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1)],dim=0), dim=0))
                _roi = _roi * ((filter_ucb_list[idx].max(dim=-1)[0].mean(dim=0).reshape(1,-1) - self.threshold_list[idx])>0) * ((self.threshold_list[idx] - filter_lcb_list[idx].min(dim=-1)[0].mean(dim=0).reshape(1,-1))>0)
            if self.constrained:
                ci = ci * _roi
        
        if self.return_UCB:
            return ucb.squeeze()
        else:
            return ci.squeeze()

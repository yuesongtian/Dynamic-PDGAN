#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np
import dnnlib

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model_G, model_D, criterion, data=None, dataloader=None, training_set=None, batch_size=None, cuda=True, total_nimg=70000):
        """
        model_G: the generator that needs Hessain information
        model_D: the discriminator that needs Hessian information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model_G = model_G.eval()  # make model is in evaluation model
        self.model_D = model_D.eval()
        self.criterion = criterion
        self.total_nimg = total_nimg

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.training_set = training_set
            self.batch_size= batch_size
            self.batch_gpu = batch_size // 1
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        self.model_G.requires_grad_(True)
        self.model_D.requires_grad_(True)
        params_G, gradsH_G = get_params_grad(self.model_G)
        self.params_G = params_G
        self.gradsH_G = gradsH_G  # gradient used for Hessian computation
        params_D, gradsH_D = get_params_grad(self.model_D)
        self.params_D, self.gradsH_D = params_D, gradsH_D

    def dataloader_hv_product(self, v, mode):
        if mode == 'G':
            model = self.model_G
            params = self.params_G
        elif mode == 'D':
            model = self.model_D
            params = self.params_D

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in params
              ]  # accumulate result
        phases = [mode+'main']
        cur_nimg = 0
        while cur_nimg < self.total_nimg:
            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                phase_real_img, phase_real_c = next(self.data)
                phase_real_img = phase_real_img.to(device).to(torch.float32) / 127.5 - 1
                phase_real_c = phase_real_c.to(device)
                all_gen_z = torch.randn([self.batch_size, self.model_G.z_dim], device=device)
                all_gen_c = [self.training_set.get_label(np.random.randint(len(self.training_set))) for _ in range(self.batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                #all_gen_c = [phase_gen_c.split(self.batch_gpu) for phase_gen_c in all_gen_c.split(self.batch_size)]
            
            # Execute training phases.
            model.zero_grad()
            model.requires_grad_(True)
            #self.model_D.requires_grad_(True)
            params, _ = get_params_grad(model)
            # Accumulate gradients.
            loss = self.criterion.accumulate_gradients_main(phase=phases[0], real_img=phase_real_img, real_c=phase_real_c, gen_z=all_gen_z, gen_c=all_gen_c, gain=1, cur_nimg=cur_nimg)
            dl_dparam = torch.autograd.grad(loss.mean(), params, create_graph=True)
            #model.requires_grad_(False)
            #self.model_D.requires_grad_(False)

            tmp_num_data = phase_real_img.size(0) if mode== 'D' else all_gen_z.size(0)
            #params, gradsH = get_params_grad(model)
            #model.zero_grad()
            #for grad, param in zip(dl_dparam, params):
            #    grad.retain_grad()
            #    param.retain_grad()
            Hv = torch.autograd.grad(dl_dparam,
                                    params,
                                    grad_outputs=v,
                                    create_graph=True)
            del dl_dparam
            del params
            THv = [
                THv1 + Hv1.detach() * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            del Hv
            num_data += float(tmp_num_data)
            cur_nimg += self.batch_size

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v)   #.cpu().item()
        return eigenvalue, THv

    def first_order(self, mode):
        if mode == 'G':
            model = self.model_G
            params = self.params_G
        elif mode == 'D':
            model = self.model_D
            params = self.params_D

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in params
              ]  # accumulate result
        phases = [mode+'main']
        cur_nimg = 0
        total_norm = 0.
        while cur_nimg < self.total_nimg:
            # Fetch training data.
            with torch.autograd.profiler.record_function('data_fetch'):
                phase_real_img, phase_real_c = next(self.data)
                phase_real_img = phase_real_img.to(device).to(torch.float32) / 127.5 - 1
                phase_real_c = phase_real_c.to(device)
                all_gen_z = torch.randn([self.batch_size, self.model_G.z_dim], device=device)
                all_gen_c = [self.training_set.get_label(np.random.randint(len(self.training_set))) for _ in range(self.batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                #all_gen_c = [phase_gen_c.split(self.batch_gpu) for phase_gen_c in all_gen_c.split(self.batch_size)]
            
            # Execute training phases.
            model.zero_grad()
            model.requires_grad_(True)
            #self.model_D.requires_grad_(True)
            params, _ = get_params_grad(model)
            # Accumulate gradients.
            loss = self.criterion.accumulate_gradients_main(phase=phases[0], real_img=phase_real_img, real_c=phase_real_c, gen_z=all_gen_z, gen_c=all_gen_c, gain=1, cur_nimg=cur_nimg)
            dl_dparam = torch.autograd.grad(loss.mean(), params, create_graph=True)
            print(f'dl_dparam is {type(dl_dparam)}, item is {(type(dl_dparam[0]))}')
            #model.requires_grad_(False)
            #self.model_D.requires_grad_(False)

            item = 0.
            for p in dl_dparam:
                param_norm = p.data.norm(2)
                item += param_norm.item() ** 2
            print(f'item is {item}')
            total_norm += item ** (1. / 2)

        return total_norm

    def eigenvalues(self, mode, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0
        params = self.params_G if mode == 'G' else self.params_D
        gradsH = self.gradsH_G if mode == 'G' else self.gradsH_D
        model = self.model_G if mode == 'G' else self.model_D

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector
            print(f'v is {len(v)}')

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v, mode)
                else:
                    Hv = hessian_vector_product(gradsH, params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)
                print(f'i is {i}, tmp_eigenvalue is {tmp_eigenvalue}')
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            print(f'{computed_dim}th eigenvalue is {eigenvalue}')
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, mode, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        model = self.model_G if mode=='G' else self.model_D
        params = self.params_G if mode == 'G' else self.params_D
        gradsH = self.gradsH_G if mode == 'G' else self.gradsH_D
        for i in range(maxIter):
            model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in params
            ]
            # generate Rasdemacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v, mode)
            else:
                Hv = hessian_vector_product(gradsH, params, v)
            trace_vhv.append(group_product(Hv, v).detach().cpu().item())
            print(f'trace_vhv is {trace_vhv}, trace_vhv is {type(trace_vhv)}')
            if abs(np.mean(np.array(trace_vhv)) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(np.array(trace_vhv))

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :]**2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full

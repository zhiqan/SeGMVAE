import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module2 import Encoder, Decoder

class SeGMVAE(nn.Module):
    def __init__(
        self,
        input_shape=[1024],
        unsupervised_em_iters=5,
        semisupervised_em_iters=5,
        fix_pi=False,
        hidden_size=64,
        component_size=20,   
        latent_size=64, 
        train_mc_sample_size=10,
        test_mc_sample_size=10
    ):
        super(SeGMVAE, self).__init__()
        
        self.input_shape = input_shape
        self.unsupervised_em_iters = unsupervised_em_iters
        self.semisupervised_em_iters = semisupervised_em_iters
        self.fix_pi = fix_pi
        self.hidden_size = hidden_size
        self.last_hidden_size =2*2*hidden_size
        self.component_size = component_size
        self.latent_size = latent_size
        self.train_mc_sample_size = train_mc_sample_size
        self.test_mc_sample_size = test_mc_sample_size

        self.encoder = Encoder(hidden_size=hidden_size)
   

        self.q_z_given_x_net = nn.Sequential(nn.Linear(self.last_hidden_size, self.hidden_size))


        self.decoder = Decoder(hidden_size=hidden_size)

        
        self.rec_criterion = nn.MSELoss(reduction='sum') ### 必须有reduction='sum'

        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * np.log(2 * np.pi)))
        self.register_buffer('uniform_pi', torch.ones(self.component_size)/self.component_size)
        
    def reparametrize(self, mean, logvar, S=1):
        mean = mean.unsqueeze(1).repeat(1, S, 1)
        logvar = logvar.unsqueeze(1).repeat(1, S, 1)
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(mean)
        return eps.mul(std).add(mean)

    def gaussian_log_prob(self, x, mean, logvar=None):
        if logvar is None:
            logvar = torch.zeros_like(mean)
        a = (x - mean).pow(2)
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p + self.log_norm_constant
        return log_p.sum(dim=-1)

    def get_unsupervised_params(self, X, psi):
        sample_size = X.shape[1]        
        pi, mean = psi

        
        log_likelihoods = self.gaussian_log_prob(
           
            X[:, :, None, :].repeat(1, 1, self.component_size, 1), 
            mean[:, None, :, :].repeat(1, sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, sample_size, 1))

        # batch_size, sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        # batch_size, component_size
        N = torch.sum(posteriors, dim=1)
        if not self.fix_pi:
            # batch_size, component_size
            pi = N / N.sum(dim=-1, keepdim=True)
        # batch_size, component_size, latent_size
        denominator = N[:, :, None].repeat(1, 1, self.latent_size)
        
        # batch_size, component_size, latent_size
        mean = torch.matmul(
            # batch_size, component_size, sample_size
            posteriors.permute(0, 2, 1).contiguous(),
            # batch_size, sample_size, latent_size
            X
        ) / denominator

        return pi, mean

    def get_semisupervised_params(self, unsupervised_X, supervised_X, y, psi):
        unsupervised_sample_size = unsupervised_X.shape[1]
        pi, mean, logvar = psi

        # batch_size, unsupervised_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, unsupervised_sample_size, component_size, latent_size
            unsupervised_X[:, :, None, :].repeat(1, 1, self.component_size, 1), 
            mean[:, None, :, :].repeat(1, unsupervised_sample_size, 1, 1)
        ) + torch.log(pi[:, None, :].repeat(1, unsupervised_sample_size, 1))

        # batch_size, unsupervised_sample_size, component_size
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )

        
        supervised_N = torch.sum(y, dim=1)

        unsupervised_N = torch.sum(posteriors, dim=1)        
        
        denominator = supervised_N[:, :, None].repeat(1, 1, self.latent_size) + unsupervised_N[:, :, None].repeat(1, 1, self.latent_size)

        
        supervised_mean = torch.matmul(
            
            y.permute(0, 2, 1).contiguous(),
         
            supervised_X
        )
        #
        unsupervised_mean = torch.matmul(
         
            posteriors.permute(0, 2, 1).contiguous(),
          
            unsupervised_X
        ) 
        mean = (supervised_mean + unsupervised_mean) / denominator

        supervised_X2 = torch.matmul(
            
            y.permute(0, 2, 1).contiguous(),
            
            supervised_X.pow(2.0)
        )

        supervised_X_mean = mean * torch.matmul(
            
            y.permute(0, 2, 1).contiguous(),
            
            supervised_X
        )
        
       
        supervised_mean2 = supervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)

        supervised_var = supervised_X2 - 2 * supervised_X_mean + supervised_mean2

        unsupervised_X2 = torch.matmul(
        
            posteriors.permute(0, 2, 1).contiguous(),
  
            unsupervised_X.pow(2.0)
        )

        unsupervised_X_mean = mean * torch.matmul(
        
            posteriors.permute(0, 2, 1).contiguous(),
        
            unsupervised_X
        )
        
     
        unsupervised_mean2 = unsupervised_N[:, :, None].repeat(1, 1, self.latent_size) * mean.pow(2.0)

        unsupervised_var = unsupervised_X2 - 2 * unsupervised_X_mean + unsupervised_mean2

        var = (supervised_var + unsupervised_var) / denominator
        logvar = torch.log(var)

        return pi, mean, logvar

    def get_posterior(self, H, mc_sample_size=10):
        batch_size, sample_size = H.shape[:2]

        ## q z ##
      
        q_z_given_x_mean= self.q_z_given_x_net(H)
        q_z_given_x_logvar = self.q_z_given_x_net(H)
        
        q_z_given_x = self.reparametrize(
            mean=q_z_given_x_mean.view(batch_size*sample_size, self.latent_size), 
            logvar=q_z_given_x_logvar.view(batch_size*sample_size, self.latent_size), 
            S=mc_sample_size
        ).view(batch_size, sample_size, mc_sample_size, self.latent_size)

        return q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x

    def get_unsupervised_prior(self, z):
        batch_size, sample_size = z.shape[0], z.shape[1]
        
        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        initial_mean = []
        for i in range(batch_size):
            idxs = torch.from_numpy(
                np.random.choice(sample_size, self.component_size, replace=False)
            ).to(z.device)
            # component_size, latent_size
            initial_mean.append(torch.index_select(z[i], dim=0, index=idxs))
       
        initial_mean = torch.stack(initial_mean, dim=0)
        
        psi = (initial_pi, initial_mean)
        for _ in range(self.unsupervised_em_iters):
            psi = self.get_unsupervised_params(
                X=z, 
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi

    def get_semisupervised_prior(self, unsupervised_z, supervised_z, y):
        batch_size = unsupervised_z.shape[0]

        # batch_size, component_size
        initial_pi = self.uniform_pi[None, :].repeat(batch_size, 1)
        # batch_size, component_size
        supervised_N = y.sum(dim=1)
        # batch_size, component_size, latent_size
        initial_mean = torch.matmul(
          
            y.permute(0, 2, 1).contiguous(),
          
            supervised_z
        ) / supervised_N[:, :, None].repeat(1, 1, self.latent_size)
       
        initial_logvar = torch.zeros_like(initial_mean)
        
        psi = (initial_pi, initial_mean, initial_logvar)
        for _ in range(self.semisupervised_em_iters):
            psi = self.get_semisupervised_params(
                unsupervised_X=unsupervised_z,
                supervised_X=supervised_z,
                y=y,
                psi=psi
            )
        psi = (param.detach() for param in psi)
        return psi
    
    def klloss(self, mean, logvar, laplace=False):
        if laplace:
            mean = mean.abs()
            logvar -= np.log(2)
            # return (-logvar - 1 + mean / logvar.exp() + (-mean / logvar.exp()).exp()).sum()
            return (-logvar - 1 + mean + (-mean / logvar.exp() + logvar).exp()).sum(-1)
        else:  # gaussian
            return -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp())

    def forward(self, X):
        batch_size, sample_size = X.shape[:2]

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, sample_size, self.last_hidden_size)

        # q_z
   
        q_z_given_x_mean, q_z_given_x_logvar, q_z_given_x = self.get_posterior(H, self.train_mc_sample_size)

        # p_z
        all_z = q_z_given_x.view(batch_size, -1, self.latent_size)
        p_z_given_psi = self.get_unsupervised_prior(z=all_z)

        p_y_given_psi_pi, p_z_given_y_psi_mean = p_z_given_psi

        ## decode ##
        
        H_rec = q_z_given_x.view(-1, self.latent_size)
       
        X_rec = self.decoder(
            H_rec
        ).view(batch_size, sample_size, self.train_mc_sample_size, *self.input_shape)

        ## rec loss ##
        rec_loss = self.rec_criterion(X_rec, X[:, :, None, :].repeat(1,1, self.train_mc_sample_size, 1))
       
       
        ## kl loss ##
        KLD=self.klloss(q_z_given_x_mean,q_z_given_x_logvar)
        
        # batch_size, sample_size, mc_sample_size
        log_qz = self.gaussian_log_prob(
           
            q_z_given_x,
            q_z_given_x_mean[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1),
            q_z_given_x_logvar[:, :, None, :].repeat(1, 1, self.train_mc_sample_size, 1)
        )
        
        #
        log_likelihoods = self.gaussian_log_prob(
          
            q_z_given_x[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1),
            p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, sample_size, self.train_mc_sample_size, 1, 1)
        ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, sample_size, self.train_mc_sample_size, 1))
        # batch_size, sample_size, mc_sample_size
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)

        #kl_loss = torch.mean(log_qz.mean(dim=-1) - log_pz.mean(dim=-1))
        kl_loss = torch.mean(log_qz - log_pz)

        return rec_loss, 0.005*KLD-kl_loss

    def prediction(self, X_tr, y_tr, X_te):
        batch_size, tr_sample_size = X_tr.shape[:2]
        te_sample_size = X_te.shape[1]

        
        X = torch.cat([X_tr, X_te], dim=1)

        # encode
        H = self.encoder(
            X.view(-1, *self.input_shape)
        ).view(batch_size, tr_sample_size+te_sample_size, self.last_hidden_size)

        # q_z
        
        _, _, q_z_given_x = self.get_posterior(H, self.test_mc_sample_size)
        
        ## p z ##
       
        unsupervised_z = q_z_given_x[:, tr_sample_size:(tr_sample_size+te_sample_size), :, :].view(batch_size, te_sample_size*self.test_mc_sample_size, self.latent_size)
        
        supervised_z = q_z_given_x[:, :tr_sample_size, :, :].view(batch_size, tr_sample_size*self.test_mc_sample_size, self.latent_size)        
       
        y = y_tr.view(batch_size*tr_sample_size)[:, None].repeat(1, self.test_mc_sample_size).view(batch_size, tr_sample_size*self.test_mc_sample_size)
       
        y = F.one_hot(y, self.component_size).float()
                
        # p_z
        p_z_given_psi = self.get_semisupervised_prior(
            unsupervised_z=unsupervised_z,
            supervised_z=supervised_z,
            y=y
        )

    
        p_y_given_psi_pi, p_z_given_y_psi_mean, p_z_given_y_psi_logvar = p_z_given_psi

        # batch_size, te_sample_size, mc_sample_size, component_size
        log_likelihoods = self.gaussian_log_prob(
            # batch_size, te_sample_size, mc_sample_size, component_size, latent_size
            unsupervised_z.view(batch_size, te_sample_size, self.test_mc_sample_size, self.latent_size)[:, :, :, None, :].repeat(1, 1, 1, self.component_size, 1), 
            p_z_given_y_psi_mean[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1),
            p_z_given_y_psi_logvar[:, None, None, :, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1, 1)
        ) + torch.log(p_y_given_psi_pi[:, None, None, :].repeat(1, te_sample_size, self.test_mc_sample_size, 1))

        
        posteriors = torch.exp(
            log_likelihoods - torch.logsumexp(log_likelihoods, dim=-1, keepdim=True)
        )
        # batch_size, sample_size
        y_te_pred = posteriors.mean(dim=-2).argmax(dim=-1)
        posteriors1=posteriors.mean(dim=-2)
        

        return y_te_pred,posteriors1,q_z_given_x[:, tr_sample_size:(tr_sample_size+te_sample_size), :, :].view(batch_size, te_sample_size*self.test_mc_sample_size, self.latent_size)


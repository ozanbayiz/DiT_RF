import torch
import torch.nn as nn

class RF:

    def __init__(self, model, distributed=False):
        self.model = model.module if distributed else model
        self.criterion = nn.MSELoss()


    def forward(self, x_0, y):
        b = x_0.shape[0]

        # sample time from [0, 1] using logit-normal sampling
        n_t = torch.randn((b,)).to(x_0.device)
        t = torch.sigmoid(n_t)
        
        # expand time to match input shape
        t_exp = t.view([b, *([1] * len(x_0.shape[1:]))])
        
        # add noise to input
        x_1 = torch.randn_like(x_0)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1

        # predict velocity
        v_theta = self.model(x_t, t, y)

        # compute loss
        loss = self.criterion(x_1 - x_0, v_theta)

        return loss
    
    @torch.no_grad()
    def sample(self, z, y, sample_steps=50, cfg_scale=2.0, return_all_steps=True):
        b = z.shape[0]

        # step size
        dt = 1.0 / sample_steps
        # # convert to tensor
        dt = torch.tensor([dt] * b, device=z.device).view([b, *([1] * len(z.shape[1:]))])

        # list of images from t=1 to t=0
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t]* b, device=z.device)
            
            # predict velocity
            if cfg_scale > 0:
                v_theta = self.model.forward_with_cfg(z, t, y, cfg_scale=cfg_scale)
            else:
                v_theta = self.model(z, t, y)
            
            # take step
            z = z - v_theta * dt

            # image at t_i
            images.append(z)
        
        if return_all_steps:
            return images
        else:
            return z
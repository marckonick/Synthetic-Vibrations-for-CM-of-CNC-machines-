

import numpy as np
import torch
import torch.utils.data as data
from DiffusionModel import *
from operator import itemgetter
import Functions_FeatureExtraction as FFE



class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        """Use EMA weights (for sampling / evaluation)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original training weights"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
        
        

class labeled_dataset(data.Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])
      
        
def producable_ops():

    y_data = np.load("Y_data_segment_filt_train.npy")


    embed_classes_unique = np.unique(y_data)
    embed_classes_unique = np.sort(embed_classes_unique) # od najmanjeg 

    cur_label = 0

    rel_dict = {}
    for i in range(15):
        if i in embed_classes_unique:
            rel_dict[i] = cur_label
            y_data[y_data == i] = cur_label
            cur_label += 1
        else:
            rel_dict[i] = -1    

    return embed_classes_unique, rel_dict 


@torch.no_grad()
def sample(model, y, T, beta_t, alpha_t, alpha_bar_t, device):
    model.eval()

    # start from pure noise
    N_sample = len(y)
    x = torch.randn((N_sample, 1, 4096), device=device)
    
    #y = torch.ones(N_sample, device=device)
    #y = y.long()

    for tt in reversed(range(T)):   # tt = T-1, ..., 0
        # timestep tensor for the batch
        t = torch.full((N_sample,), tt, device=device, dtype=torch.long)

        beta      = beta_t[tt]                      # scalar
        alpha     = alpha_t[tt]                     # scalar
        alpha_bar = alpha_bar_t[tt]                 # scalar

        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        sqrt_recip_alpha         = torch.sqrt(1.0 / alpha)

        # ε_θ(x_t, t)
        eps_theta = model(x, t, y)                     # (N_sample, 1, 256)

        # DDPM mean (Ho et al., Alg. 2)
        mean = sqrt_recip_alpha * (
            x - (beta / sqrt_one_minus_alpha_bar) * eps_theta
        )

        if tt > 0:
            # add noise except at last step
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * z
        else:
            # last step: deterministic
            x = mean

    return x

def sample_per_op(min_samples):

  synt_samples = []
  for c_ix in ops_2_synt: 
    
      if c_ix not in embed_classes_unique:
          print(f"The class {c_ix} cannot be reproduced by this model ")
          break
    
      N_current = sum(y_train_orig == c_ix)
        
      if N_current < min_samples: 
          N_2_synt = min_samples - N_current
      else:
          print(f"OP {c_ix} has more than {min_samples} samples already !")
          continue  
            
            
      ops_2_sample_real = np.ones(N_2_synt)# 0np.random.choice(embed_classes_unique, 4000)
      ops_2_sample_indexes = itemgetter(*ops_2_sample_real)(rel_dict)
      ops_2_sample_indexes = torch.tensor(ops_2_sample_indexes, device=device).long()
    
     
      synt_samples_temp = sample(model, ops_2_sample_indexes, T, beta_t, alpha_t, alpha_bar_t, device)

      synt_samples_temp = synt_samples_temp.detach().cpu().numpy()
      #synt_samples_temp = np.reshape(synt_samples_temp, (1,4096))
      synt_samples.append(synt_samples_temp)
      print(f"Synthesized samples from OP {c_ix}")

  
  if len(synt_samples) > 0:
      
    synt_samples = np.concatenate(synt_samples, axis=0)
    synt_samples = synt_samples[:,0,:]

    np.save(f"X_DIFF_AUG.npy", synt_samples)
    print("Synth data saved after - sample_per_op !!!")
  else:
    print("Nothing to save!!!")
    
  return synt_samples  
    
    
def sample_uniform(ratio_N, y_total):
    
    print("Uniform sampling here ...\n")
    
    ema.apply_shadow(model)
    N_total =  int(y_total*ratio_N)
    x_samples_all = []
    N_iter2s = 140 # num of samples to generate per 
    print(f"Generating {N_total} samples!")
    for i_iter in range(N_total//N_iter2s + 1):
     
     if i_iter == N_total//N_iter2s:
         N_iter2s =  N_total % N_iter2s    
        
     ops_2_sample_real = np.random.choice(ops_2_synt, N_iter2s)
     ops_2_sample_indexes = itemgetter(*ops_2_sample_real)(rel_dict)
     ops_2_sample_indexes = torch.tensor(ops_2_sample_indexes, device=device).long()

     x_samples = sample(model, ops_2_sample_indexes, T, beta_t, alpha_t, alpha_bar_t, device)
     x_samples = x_samples.detach().cpu().numpy()
     x_samples = x_samples[:,0,:]
     x_samples_all.append(x_samples)
    
     print(f"I have synth {N_iter2s} of new data samples...")
     #print(x_samples_all[-1].shape)
    
    np.save(f"X_DIFF_AUG_Unif.npy", np.concatenate(x_samples_all, 0))
    print("Data saved after - sample_uniform")
    
    ema.restore(model)

    return x_samples

# %%

device = 'mps' # cuda 

embed_classes_unique, rel_dict = producable_ops()
T = 400

model = UNet1D(in_channels=1, base_channels=32, time_emb_dim=128, num_ops=len(embed_classes_unique), op_emb_dim=32) # base_channels=64
model = model.to(device)
#model.load_state_dict(torch.load(f"diffusion_model_M2_M3_ALL_OPS_400_steps_320_epochs_1767030765.pth")) 
model.number_of_params()

# EMA STYLE 


ema = EMA(model, decay=0.999)


ckpt = torch.load("SavedDifussionModels/ckpt.pt", map_location=torch.device('mps')) 
model.load_state_dict(ckpt["diffusion_model_M2_M3_ALL_OPS_400_steps_401_epochs_1769045238"])
ema.shadow = ckpt["ema_400_steps_401_epochs_1769045238"]


# diffusion_model_M2_M3_ALL_OPS_200_steps_600_epochs - 48 base_channels
#  diffusion_model_M2_M3_ALL_OPS_220_steps_400_epochs

beta_t = torch.linspace(1e-4, 2e-2, T)
alpha_t = 1 - beta_t
alpha_bar_t = torch.cumprod(alpha_t, dim=0)# (T,)

# %%

N_ratio =  0.01
ops_2_synt = [-1] 
y_train_orig = np.load("Y_data_segment_filt_train.npy")


if ops_2_synt[0] == -1:
   ops_2_synt = embed_classes_unique
   y_total = len(y_train_orig)
else:
    y_total = 0
    for cc_ix in ops_2_synt:
        y_total += sum(y_train_orig == cc_ix)
   

# uniform - samples same numb of samples from each class
#samples_uniform = sample_uniform(N_ratio, y_total)

# per_op - za svaku operaciju koliko ima u originalu pa puta N_ratio 

min_samples = 400
samples_per_op = sample_per_op(min_samples)

# %%
kwarg_args = {'win_len':128, 'overlap_l':128, 'hop_l':64, "n_mels":64}

selected_feature = "MelLog"
X_features_extracted = FFE.ExtractSelectedFeatures_Synthetic(-1, "X_DIFF_AUG.npy", "", selected_feature, **kwarg_args) 
save_name_X_synth = f"x_samples_diffusion_{selected_feature}.npy"
np.save(save_name_X_synth, X_features_extracted)
print(f"Computed syntehtic samples! - {save_name_X_synth}, shape is {X_features_extracted.shape}")







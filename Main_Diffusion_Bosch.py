import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.nn as nn
import time
from DiffusionModel import *
import torchaudio
import pickle 


# works great, gives same results as numpy 
class MelLog(nn.Module):
    def __init__(self, fs=2000, n_fft=256, hop_length=64, win_length=128,
                 n_mels=64, f_min=0.0, f_max=1000.0, eps=1e-10, ref_power=None, top_db=80.0):
        super().__init__()
        self.eps = eps
        
        #ref_power (e.g. median of per-sample mel_pow max over the training set)
        self.ref_power = ref_power  # float or None, not in decibels
        self.top_db = top_db

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=fs,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,        # power spectrogram
            window_fn=torch.hann_window,
            normalized=False,
            center=True,
            pad_mode="constant",
            mel_scale="slaney",
            norm="slaney",
        )        

    def power_to_db_like_librosa(self, mel_pow, amin=1e-10):
        # mel_pow: (B, n_mels, T), power
        mel_pow = mel_pow.to(torch.float32)

        # librosa clamps by amin before log
        mel_pow = torch.clamp(mel_pow, min=amin)

        # ref=np.max -> per-example max over (mel, time)
        ref = mel_pow.amax(dim=(1, 2), keepdim=True)
        ref = torch.clamp(ref, min=amin)

        # 10 * log10(S) - 10 * log10(ref)
        mel_db = 10.0 * torch.log10(mel_pow) - 10.0 * torch.log10(ref)

        # librosa default top_db=80: clamp to (max - top_db).
        # after ref=np.max, max is 0, so this floors at -top_db
        if self.top_db is not None:
            mel_db = torch.clamp(mel_db, min=-float(self.top_db))
            
        return mel_db

    def power_to_db(self, mel_pow, amin=1e-10):
        
        mel_pow = torch.clamp(mel_pow.to(torch.float32), min=amin)

        if self.ref_power is None:
            # absolute dB (ref=1.0)
            mel_db = 10.0 * torch.log10(mel_pow)
        else:
            # This keeps values in a familiar numeric range without making them sample-dependent.
            ref = torch.tensor(self.ref_power, device=mel_pow.device, dtype=mel_pow.dtype)
            ref = torch.clamp(ref, min=amin)
            mel_db = 10.0 * torch.log10(mel_pow) - 10.0 * torch.log10(ref)

        # optional: clamp dynamic range per sample or globally
        if self.top_db is not None:
            # If you want "top_db relative to each sample max", you can still do it
            # without changing absolute reference, by subtracting sample max then flooring:
            peak = mel_db.amax(dim=(1,2), keepdim=True)
            mel_db = torch.clamp(mel_db, min=peak - float(self.top_db)) # OVO JE FLOORING SA (peak - top_db)

        return mel_db
    
    def forward(self, x):  # x: (B,1,L)
        # mel expects (B, L) or (B, C, L) depending; safest:
        x = x.squeeze(1)  # (B,L)
        mel_pow = self.mel(x)  # (B, n_mels, T) power
        #mel_db = self.power_to_db_like_librosa(mel_pow, amin=1e-10)
        mel_db = self.power_to_db(mel_pow, amin=1e-10)
        #mel_db = 10.0 * torch.log10(mel_pow + self.eps)
        return mel_db
    
    
def estimate_x0_from_eps(x_t, eps_hat, alpha_bar):
    """
    x_t: (B,1,L)
    eps_hat: (B,1,L)
    alpha_bar: (B,1,1)
    """
    return (x_t - torch.sqrt(1.0 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar + 1e-8)

def bandlimit_loss_fft(x0_hat, fs=2000.0, f_cut=800.0):
    """
    Computes energy above f_cut and divides with total energy 
    Penalize high-frequency energy above f_cut in x0_hat.
    x0_hat: (B,1,L)
    """
    B, C, L = x0_hat.shape
    # fft.rfft one dimensional Fourier transform of real-valued input
    X = torch.fft.rfft(x0_hat[:, 0], dim=-1)          # (B, n_freq), 64 x 2049
    P = (X.real**2 + X.imag**2) / L                   # power (B, n_freq), 64 x 2049 
    # freqs  - frekvencije 0 .. Fs/2
    freqs = torch.fft.rfftfreq(L, d=1.0/fs).to(x0_hat.device)  # (n_freq,) 2049
    mask = freqs >= f_cut #2049 - false false, ... true true 
    # normalize by total power so scale is stable
    hf = P[:, mask].mean() # srednja snaga frekvencija vecih od f_cut - jedan broj
    tot = P.mean() # srednja snaaga - jedan broj
    return hf / (tot + 1e-8)



# finding frejms with lowest energy in real data and compute 
# energy for them in the fake data 
def silence_time_masked_loss(x0_hat, x0, q=0.2, frame=128, hop=64):
    frames_real = x0[:, 0].unfold(-1, frame, hop)
    rms_real = torch.sqrt(frames_real.pow(2).mean(dim=-1) + 1e-8)

    # nadjemo frejmove koji imaju najmanju energiju
    k = max(1, int(rms_real.shape[1] * q))
    _, low_idx = torch.topk(rms_real, k=k, dim=1, largest=False)

    frames_hat = x0_hat[:, 0].unfold(-1, frame, hop)
    idx_exp = low_idx[..., None].expand(-1, -1, frame)
    quiet_hat = frames_hat.gather(dim=1, index=idx_exp)

    return quiet_hat.pow(2).mean()



def mel_silence_loss_torch(mel_db, y, db_tsh = 65):
    
    with open("freqs_minmax.pkl", "rb") as f: # Open in read binary mode ('rb')
        freqs_minmax = pickle.load(f)
        
    B, Nm, T = mel_db.shape[0], mel_db.shape[1], mel_db.shape[2]    
    y_real_ops = [labels_2_ops[i.item()] for i in y]
    suma = 0.0

    for i,xx in enumerate(mel_db):
        suma += torch.mean(torch.abs(xx[freqs_minmax[y_real_ops[i]][0:12]]))

    return -suma/B + db_tsh

def silence_mel_floor_loss(
    mel_db, 
    x0_hat,                  # (B,1,L)
    x0,
    y,                       # (B,)
    labels_2_ops,            # dict: model-label -> original op id
    freqs_minmax_t,
    q=0.20,                  # fraction of lowest-energy frames to treat as "quiet"
    n_low_bins=12,           # number of "rare" mel bins to penalize (e.g. 12/64)
    target_db=-45.0,         # silence floor (dB). We want mel_db <= target_db in quiet frames
    margin=2.6,
    frame=128,
    hop=64,
    use_softplus=True,
):
    """
    Penalize energy in selected mel bins, but ONLY in the lowest-energy time frames.
    Fully differentiable (no numpy, no detach).
    """
    B, C, L = x0_hat.shape
    assert C == 1

    # ---- 1) find quiet frames via time-domain RMS (aligned with win/hop)
    #frames = x0_hat[:, 0].unfold(dimension=-1, size=frame, step=hop)  # (B, n_frames, frame)
    frames = x0[:, 0].unfold(dimension=-1, size=frame, step=hop)  # (B, n_frames, frame)
    rms = torch.sqrt(frames.pow(2).mean(dim=-1) + 1e-8)               # (B, n_frames)

    n_frames = rms.shape[1]
    k = max(1, int(n_frames * q))
    _, low_t_idx = torch.topk(rms, k=k, dim=1, largest=False)         # (B, k), frame indices with lowest RMS

    # ---- 2) mel in dB (differentiable)
    # With your settings win=128 hop=64, T should match n_frames (or be off by 1 due to center=True).
    T = mel_db.shape[-1]
    # Guard small mismatch: clamp indices
    low_t_idx = low_t_idx.clamp_(0, T - 1)

    # ---- 3) build op ids for each sample
    # y contains model labels (0..N_ec-1). Convert to original op id keys used in freqs_minmax.
    ops = torch.as_tensor([labels_2_ops[int(i)] for i in y], device=x0_hat.device)

    # ---- 4) compute loss grouped by op (vectorized enough, avoids per-sample python loops over mel bins)
    loss = x0_hat.new_zeros(()) # ovo je nula
    n_groups = 0

    for op_id in ops.unique().tolist():
        mask = (ops == op_id)
        if mask.sum() == 0:
            continue

        mel_sub = mel_db[mask]                       # (Bop, n_mels, T) uzme samo one koji odgovaraju op_id i onda bude npr(5,64,65)
        t_idx_sub = low_t_idx[mask]                  # (Bop, k)

        # choose the "rare" mel bins (lowest average energy) for this op
        f_idx = freqs_minmax_t[op_id][:n_low_bins]   # (n_low_bins,)

        # gather mel at those freqs: (Bop, n_low_bins, T)
        mel_f = mel_sub.index_select(dim=1, index=f_idx) # npr 7,12,65  

        # gather at quiet time frames: want (Bop, n_low_bins, k)
        # Expand indices to match gather shape
        t_idx_exp = t_idx_sub[:, None, :].expand(-1, mel_f.shape[1], -1) # od npr 7,12 bude 7,12,12, gde imamo 7 matrica sa 12 istih redova 
        mel_fq = mel_f.gather(dim=2, index=t_idx_exp) # na kraju uzme i 12 vrem tacaka (tihih) tako da bude 7,12,12

        # Penalize if mel_fq is ABOVE target_db (i.e., less negative => too much energy)
        if use_softplus:
            # smooth hinge: softplus(x - target)
            pen = F.softplus(mel_fq - (target_db + margin))
        else:
            pen = torch.relu(mel_fq - (target_db + margin))
            #pen = torch.clamp(pen, max=10.0)   # in dB units

        loss = loss + pen.mean()
        n_groups += 1

    if n_groups > 0:
        loss = loss / n_groups # bude oko 17 na pocetku
    return loss

# fakticki ocekujemo da srednja vrednost bude nula 
# i onda pratimo pos - neg i kaznjavamo sto je vece 
def polarity_balance_loss(x0_hat):
    pos = torch.relu(x0_hat).mean()
    neg = torch.relu(-x0_hat).mean()
    return (pos - neg).abs()



class labeled_dataset(data.Dataset):

      def __init__(self,X, Y):
        self.data = X
        self.labels = Y

      def __len__(self):
          return len(self.data)

      def __getitem__(self,idx):
          return (self.data[idx], self.labels[idx])

def comp_val_loss():

  model.eval()
  val_loss = 0.0
    
  with torch.no_grad():
   for xt, yt in x_test:

    noyz = torch.randn((len(xt), 1,4096), device=device)    
    t  = torch.randint(0, T, (len(xt),), device=device)
    alpha_bar = alpha_bar_t[t].view(len(xt), 1, 1)    
        
    x0 = xt.float()    
    x0 = x0.to(device) 
    yt = yt.to(device)
    xt = x0*torch.sqrt(alpha_bar) + torch.sqrt(1-alpha_bar)*noyz
    
    y_pred = model(xt, t, yt)
    x0_hat = estimate_x0_from_eps(xt, y_pred, alpha_bar)    

    ######## LOSSES #########
    loss_v_eps = loss_fn(y_pred, noyz)
    
    #loss_sil_v = silence_time_masked_loss(x0_hat, x0, q=0.2)  #timesil
    #loss_hf_v  = bandlimit_loss_fft(x0_hat, fs=2000.0, f_cut=800.0) #hfloss
    if w_mel_eff > 0.0:
            mask = (t.float() / (T - 1) < 0.3)  # or 0.25–0.35< <- fakticki ovde selektujemo kad je mask=true biramo          
            if mask.any():
                mel_db = mel_extractor(x0_hat[mask])  # (B,n_mels,T)
                loss_mel_sil_v = silence_mel_floor_loss(mel_db, x0_hat[mask], x0[mask], yt[mask], labels_2_ops, freqs_minmax_t, q=q_most_sil, use_softplus=False)
    else:
            loss_mel_sil_v = loss_v_eps * 0.0  # keep graph-friendly zero
    
    #loss_polarity_bal_v = polarity_balance_loss(x0_hat)  # polarity
    
    # ---- Normalize auxiliary losses so scale doesn’t dominate ----
    #loss_sill_norm_v = loss_sil_v / loss_ema.norm("sil") #timesil
    loss_mel_norm_v = loss_mel_sil_v # / loss_ema.norm("mel")
    #loss_hf_norm_v = loss_hf_v / loss_ema.norm("hf")  #hfloss
    #loss_pol_norm_v = loss_polarity_bal_v / loss_ema.norm("pol")  # polarity
        
    loss_v = loss_v_eps + w_mel_eff*loss_mel_norm_v  #+ lambda_hf*loss_hf_norm_v + lambda_sil*loss_sill_norm_v  + lambda_pol * loss_pol_norm_v
    
    val_loss += loss_v.item()
 
  return val_loss/len(x_test)


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
        
class LossEMA:
    def __init__(self, decay=0.99, eps=1e-8):
        self.decay = decay
        self.eps = eps
        self.vals = {}

    def update(self, name, value_float):
        if name not in self.vals:
            self.vals[name] = value_float
        else:
            self.vals[name] = self.decay * self.vals[name] + (1.0 - self.decay) * value_float

    def norm(self, name):
        # divisor for normalization
        return self.vals.get(name, 1.0) + self.eps
        
        
def linear_ramp(epoch, start, length):
    """
    Returns 0 before 'start', then linearly increases to 1 over 'length' epochs,
    and stays at 1 afterwards.
    """
    if epoch < start:
        return 0.0
    if epoch >= start + length:
        return 1.0
    return (epoch - start) / float(length)
        
        
x_data = np.load("X_data_segment_filt_train.npy")
y_data = np.load("Y_data_segment_filt_train.npy")

x_test = np.load("X_data_segment_filt_test.npy")
y_test = np.load("Y_data_segment_filt_test.npy")

x_data = np.concatenate((x_data, x_test), 0)
y_data = np.concatenate((y_data, y_test))


embed_classes_unique = np.unique(y_data)
embed_classes_unique = np.sort(embed_classes_unique) # od najmanjeg 

N_ec = len(embed_classes_unique)
cur_label = 0

rel_dict = {}
for i in range(15):
    if i in embed_classes_unique:
        rel_dict[i] = cur_label
        y_data[y_data == i] = cur_label
        cur_label += 1
    else:
        rel_dict[i] = -1    

labels_2_ops = {v: k for k, v in rel_dict.items()}


x_data = x_data[:,:,0]
x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size=0.05, random_state=42)

x_train = x_train[:,None,:]
x_test = x_test[:,None,:]


print(x_train.shape)

x_train = labeled_dataset(x_train, y_train)
x_test = labeled_dataset(x_test, y_test)

batch_size = 64
x_train = data.DataLoader(x_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
x_test = data.DataLoader(x_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

# %%

device = 'mps' # cuda 

T = 400
#device = 'cpu'
#model = SimpleTimeSeriesDiffusionModel(T)
model = UNet1D(in_channels=1, base_channels=32, time_emb_dim=128, num_ops = N_ec, op_emb_dim=32) # base_channels=64
model = model.to(device)
ema = EMA(model, decay=0.999)


model.number_of_params()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

N_epochs = 401
beta_t = torch.linspace(1e-4, 2e-2, T) # 0.02
alpha_t = 1 - beta_t
alpha_bar_t = torch.cumprod(alpha_t, dim=0)# (T,)

beta_t = beta_t.to(device=device)
alpha_t = alpha_t.to(device=device)
alpha_bar_t = alpha_bar_t.to(device=device)

q_most_sil = 0.4


lambda_sil = 0.03
lambda_hf = 0.02
lambda_pol = 0.02

lambda_mel = 1e-2


mel_ramp_start = 80
mel_ramp_len   = 120   # reaches full strength at epoch 240


mel_extractor = MelLog(fs=2000, n_fft=128, hop_length=64, win_length=128,
                       n_mels=64, f_min=0.0, f_max=1000.0, top_db = 80).to(device)
losses = []

with open("freqs_minmax.pkl", "rb") as f: # Open in read binary mode ('rb')
   freqs_minmax = pickle.load(f)

freqs_minmax_t = {
        op: torch.as_tensor(idxs, dtype=torch.long, device=device)
        for op, idxs in freqs_minmax.items()
}
    
"""
what does lossEMA does is 

update: lambda * loss(t) + (1 - lambda) * loss(t-1)
norm: loss(t) / (lambda * loss(t) + (1 - lambda) * loss(t-1))
"""

loss_ema = LossEMA(decay=0.99)   # EMA over loss magnitudes

for epoch in range(N_epochs):
    
  cur_loss = 0.0

  cur_sil_loss = 0.0
  cur_hf_loss = 0.0  

  cur_mel_loss = 0.0
  cur_pol_loss = 0.0  
  cur_eps_loss = 0.0

  mel_r = linear_ramp(epoch, start=mel_ramp_start, length=mel_ramp_len)
  w_mel_eff = lambda_mel * mel_r

    
  for x, y in x_train:

      noyz = torch.randn((batch_size,1,4096), device=device)
      t  = torch.randint(0, T, (batch_size,), device=device)
      alpha_bar = alpha_bar_t[t].view(batch_size, 1, 1)
        
      x0 = x.float()      
      x0 = x0.to(device = device)
      y = y.to(device = device)
    
      x_t = x0*torch.sqrt(alpha_bar) + torch.sqrt(1 - alpha_bar) * noyz
      y_pred = model(x_t, t, y)     
      ####### LOSS COMPUTATION #######  
    
      loss_eps = loss_fn(y_pred, noyz)
      
      #additiona losses 
      x0_hat = estimate_x0_from_eps(x_t, y_pred, alpha_bar)
    
      #loss_sil = silence_time_masked_loss(x0_hat, x0, q=0.2) #timesil
      #loss_hf  = bandlimit_loss_fft(x0_hat, fs=2000.0, f_cut=800.0) #hfloss
      if w_mel_eff > 0.0:
            mask = (t.float() / (T - 1) < 0.3)  # or 0.25–0.35< <- fakticki ovde selektujemo kad je mask=true biramo 
            if mask.any():
                  mel_db = mel_extractor(x0_hat[mask])  # (B,n_mels,T)
                  loss_mel_sil = silence_mel_floor_loss(mel_db, x0_hat[mask], x0[mask], y[mask], labels_2_ops, freqs_minmax_t, q=q_most_sil, use_softplus=False)
      else:
            loss_mel_sil = loss_eps * 0.0  # keep graph-friendly zero
        
      #loss_polarity_bal = polarity_balance_loss(x0_hat)  #polarity
    
      """
      # These additionall losses are  defined but not used 
      
      # ---- Update EMA magnitudes (no grad) ---- 
      with torch.no_grad():
            loss_ema.update("eps", float(loss_eps.item()))
            #loss_ema.update("hf", float(loss_hf.item()))  #hfloss
            #loss_ema.update("sil", float(loss_sil.item()))  #timesil
            #loss_ema.update("pol", float(loss_polarity_bal.item()))  #polarity 
            
            #if w_mel_eff > 0.0:
            #    loss_ema.update("mel", float(loss_mel_sil.item()))            

      # ---- Normalize auxiliary losses so scale doesn’t dominate ----
      loss_mel_norm = loss_mel_sil / loss_ema.norm("mel")
      #loss_hf_norm = loss_hf / loss_ema.norm("hf") #hfloss
      #loss_sil_norm = loss_sil / loss_ema.norm("sil")  #timesil
      #loss_pol_norm = loss_polarity_bal / loss_ema.norm("pol")  #polarity
      """
      
      loss_mel_norm = loss_mel_sil
      loss = loss_eps + w_mel_eff*loss_mel_norm #+ lambda_hf*loss_hf_norm #+ lambda_sil*loss_sil_norm #+ lambda_pol*loss_pol_norm 
        
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      ema.update(model)   # ← THIS LINE

      cur_loss += loss.item()
      cur_eps_loss += loss_eps.item()  
      cur_mel_loss += (w_mel_eff * loss_mel_norm).item()
      
      
      #cur_sil_loss += (loss_sil_norm * loss_sil).item()   #timesil 
      #cur_hf_loss += (lambda_hf * loss_hf_norm).item() #hfloss
      #cur_pol_loss += (lambda_pol * loss_pol_norm).item() #polarity

  losses.append(cur_loss/len(x_train))
  if epoch % 20 == 0:
      vl = comp_val_loss()
      print(
            f"  Epoch: {epoch}, Train/Val: {cur_loss/len(x_train):.6f} / {vl:.6f}\n"
            f"  ramp={mel_r:.3f}, w_mel={w_mel_eff:.6f}\n"
            f"  eps={cur_eps_loss/len(x_train):.6f}, mel={cur_mel_loss/len(x_train):.6f}, hf={cur_hf_loss/len(x_train):.6f}, sil={cur_sil_loss/len(x_train):.6f}\n"
            f"  EMA mags: eps={loss_ema.vals.get('eps',0):.4f}, mel={loss_ema.vals.get('mel',0):.4f}, hf={loss_ema.vals.get('hf',0):.4f}, sil={loss_ema.vals.get('sil',0):.4f}\n"
        )
      model.train()

int_timestamp = int(time.time())

torch.save(model.state_dict(), f"diffusion_model_M2_M3_ALL_OPS_{T}_steps_{N_epochs}_epochs_{int_timestamp}.pth")   
torch.save(beta_t, f"beta_t_{int_timestamp}.npy")

torch.save({
    f"diffusion_model_M2_M3_ALL_OPS_{T}_steps_{N_epochs}_epochs_{int_timestamp}": model.state_dict(),
    f"ema_{T}_steps_{N_epochs}_epochs_{int_timestamp}": ema.shadow,
}, "ckpt.pt")

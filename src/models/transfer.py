import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdapter(nn.Module):
    """
    Projects domain-specific sensors into the Shared 32-dim Engine State Representation Space.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super(DomainAdapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Apply linear layers over the last dimension
        x = self.fc1(x)
        x = F.relu(x)
        x = self.ln(x)
        x = self.fc2(x)
        return x

class NCMAPSSAdapter(DomainAdapter):
    def __init__(self):
        # N-CMAPSS has 14 sensor inputs
        super(NCMAPSSAdapter, self).__init__(input_dim=14, hidden_dim=64, output_dim=32)

class TurbojetAdapter(DomainAdapter):
    def __init__(self):
        # Turbojet has 19 combined features (12 raw + 7 physics)
        super(TurbojetAdapter, self).__init__(input_dim=19, hidden_dim=64, output_dim=32)

class SharedEncoder(nn.Module):
    """
    The architecture-agnostic backbone. Processes the 32-dim latent sequence
    and outputs a single 32-dim summary vector for the sequence.
    """
    def __init__(self, input_dim=32, hidden_dim=32, dropout_rate=0.1):
        super(SharedEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        gru_out, h_n = self.gru(x)
        # h_n shape: (num_layers, batch, hidden_dim)
        x = h_n[-1] # take the last layer's hidden state -> (batch, hidden_dim)
        x = self.dropout(x)
        return x

class ContrastivePretrainingModel(nn.Module):
    """
    Bundles the N-CMAPSS Adapter and Shared Encoder for contrastive learning.
    """
    def __init__(self):
        super(ContrastivePretrainingModel, self).__init__()
        self.adapter = NCMAPSSAdapter()
        self.encoder = SharedEncoder()
        
    def forward(self, x):
        # x shape: (batch, seq_len, 14)
        z_seq = self.adapter(x) # (batch, seq_len, 32)
        h = self.encoder(z_seq) # (batch, 32)
        return h

class TransferredDigitalTwinModel(nn.Module):
    """
    The downstream Digital Twin model that uses the Turbojet Adapter 
    and the pretrained Shared Encoder.
    """
    def __init__(self, pretrained_encoder=None, dropout_rate=0.1):
        super(TransferredDigitalTwinModel, self).__init__()
        self.adapter = TurbojetAdapter()
        
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = SharedEncoder(dropout_rate=dropout_rate)
            
        self.mc_dropout = nn.Dropout(p=dropout_rate)
        self.post_gru_fc = nn.Linear(32, 32)
        
        # Output Heads (same as original Digital Twin)
        self.compressor_head = nn.Linear(32, 1)
        self.combustor_head = nn.Linear(32, 1)
        self.turbine_head = nn.Linear(32, 1)
        self.overall_health_head = nn.Linear(32, 1)
        self.thrust_head = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 19)
        z_seq = self.adapter(x)
        x = self.encoder(z_seq)
        
        x = F.relu(self.post_gru_fc(x))
        x = self.mc_dropout(x)
        
        comp_h = self.compressor_head(x)
        comb_h = self.combustor_head(x)
        turb_h = self.turbine_head(x)
        overall_h = self.overall_health_head(x)
        thrust = self.thrust_head(x)
        
        return comp_h, comb_h, turb_h, overall_h, thrust
        
    def predict_with_uncertainty(self, x, num_samples=30):
        self.train()
        preds_comp, preds_comb, preds_turb = [], [], []
        preds_overall, preds_thrust = [], []
        with torch.no_grad():
            for _ in range(num_samples):
                comp, comb, turb, overall, thrust = self.forward(x)
                preds_comp.append(comp)
                preds_comb.append(comb)
                preds_turb.append(turb)
                preds_overall.append(overall)
                preds_thrust.append(thrust)
        def get_stats(tensor_list):
            stacked = torch.stack(tensor_list)
            return stacked.mean(dim=0), stacked.std(dim=0)
        return (get_stats(preds_comp), get_stats(preds_comb), get_stats(preds_turb),
                get_stats(preds_overall), get_stats(preds_thrust))

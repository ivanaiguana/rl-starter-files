import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=True, use_text=True):
        super().__init__()

        self.use_text = use_text
        self.use_memory = use_memory

        # Image embedding layers
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)), nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Memory (LSTM) for image embeddings
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Text embedding and Transformer layers
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.word_embedding_size, nhead=4, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(
                self.transformer_encoder_layer, num_layers=2)
            self.text_embedding_size = self.word_embedding_size  # Assuming output size matches

        # Combined embedding size and actor-critic networks
        self.embedding_size = self.semi_memory_size if self.use_memory else self.image_embedding_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size if self.use_memory else 0

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x).view(x.size(0), -1)

        if self.use_memory:
            hidden = self.memory_rnn(x, (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:]))
            
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            memory = None

        if self.use_text:
            #text_embedded = self.word_embedding(obs.text)
            #text_encoded = self.transformer_encoder(text_embedded)
            #text_embedding = text_encoded.mean(dim=1)  # Combine sequence embeddings
            text_embedding = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, text_embedding), dim=1)

        actor_output = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(actor_output, dim=1))

        critic_output = self.critic(embedding)
        value = critic_output.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        text_embedded = self.word_embedding(text)
        text_encoded = self.transformer_encoder(text_embedded)
        return text_encoded.mean(dim=1)  # Return mean to condense sequence information
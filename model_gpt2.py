import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, p_dropout=0.5):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_size_per_head = n_embed // n_head

        # n_head * n_per_head_dim == n_embed
        assert n_embed % n_head == 0
        self.linear_k = nn.Linear(n_embed, n_embed, bias=False)
        self.linear_v = nn.Linear(n_embed, n_embed, bias=False)
        self.linear_q = nn.Linear(n_embed, n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.atten_dropout = nn.Dropout(p=p_dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        # linear projection
        key = self.linear_k(key).view(batch_size, self.n_head, seq_len, self.n_size_per_head)
        value = self.linear_v(value).view(batch_size, self.n_head, seq_len, self.n_size_per_head)
        query = self.linear_q(query).view(batch_size, self.n_head, seq_len, self.n_size_per_head)

        w = torch.matmul(query, key.transpose(2, 3))
        # scale
        w = w / (float(value.size(-1)) ** 0.5)

        if mask is not None:
            w = w.masked_fill(mask == 0, -128)

        w = self.atten_dropout(F.softmax(w, dim=-1))
        output = torch.matmul(w, value)

        # merge head
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.proj(output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, n_state, n_hidden, p_dropout=0.5):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.fc_1 = nn.Linear(n_state, n_hidden) # position-wise
        self.fc_2 = nn.Linear(n_hidden, n_state) # position-wise
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

        nn.init.normal_(self.fc_1.weight, std=0.02)
        nn.init.normal_(self.fc_2.weight, std=0.02)

    def forward(self, x):
        fc1_out = self.act(self.fc_1(x))
        fc2_out = self.fc_2(fc1_out)
        out = self.dropout(fc2_out)
        return out

class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_norm_epsilon, scale=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(n_embed, n_head)
        self.ln_2 = nn.LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.ffn = PositionwiseFeedForward(n_embed, 4 * n_embed)

    def forward(
        self, x, mask=None
    ):
        ln_1_x = self.ln_1(x)
        output_attn = self.attn(ln_1_x, ln_1_x, ln_1_x, mask=mask)

        x = x + output_attn
        m = self.ffn(self.ln_2(x))
        x = x + m

        return x

class GPT2Model(nn.Module):
    def __init__(self, config=None):
        super(GPT2Model, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = {
                'vocab_size': 30000,
                'n_embed': 256,
                'n_positions': 512,
                'n_layer': 8,
            }

        # word token embedding
        self.wte = nn.Embedding(self.config['vocab_size'], self.config['n_embed'])
        # word position embedding
        self.wpe = nn.Embedding(self.config['n_positions'], self.config['n_embed'])
        # dropout
        self.drop = nn.Dropout(self.config['dropout'])
        # self attentions blocks
        self.h = nn.ModuleList([
            Block(
                self.config['n_embed'],
                self.config['n_head'],
                self.config['layer_norm_epsilon']
                )
                for _ in range(self.config['n_layer'])
            ])
        self.ln_f = nn.LayerNorm(self.config['n_embed'], eps=config['layer_norm_epsilon'])


    def forward(self,
            input_ids=None,
            inputs_embeds=None,
            position_ids=None,
        ):
        '''
        input_ids: torch.FloatTensor, [batch_size, sequence_length]
        position_ids: [batch_size, sequence_length]
        '''
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_ids.shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states,
            )

            hidden_states = outputs

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        outputs = (hidden_states,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.n_embed = config['n_embed']
        self.vocab_size = config['vocab_size']
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids=None,
        labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) # + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

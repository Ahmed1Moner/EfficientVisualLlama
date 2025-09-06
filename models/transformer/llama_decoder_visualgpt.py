import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .configuration_llama_visualgpt import VisualLlamaConfig
from typing import Optional

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super(LlamaMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = gelu
        self.dropout = nn.Dropout(getattr(config, 'resid_dropout', 0.1))

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(CrossAttention, self).__init__()
        n_state = nx
        assert n_state % config.num_attention_heads == 0
        self.n_head = config.num_attention_heads
        self.split_size = n_state
        self.scale = scale
        
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        resid_dropout_value = getattr(config, 'resid_dropout', 0.1)
        self.resid_dropout = nn.Dropout(resid_dropout_value)
        
        self.visual_proj = Conv1D(n_state, config.visual_feature_size)
        
    def _attn(self, q, k, v, attention_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        
        if attention_mask is not None:
            if isinstance(attention_mask, torch.Tensor):
                w = w + attention_mask
            else:
                attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=w.device)
                attention_mask = attention_mask * -10000.0
                w = w + attention_mask
        
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)
    
    def forward(self, x, encoder_output, attention_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        encoder_output = self.visual_proj(encoder_output)
        
        enc_key = self.split_heads(encoder_output, k=True)
        enc_value = self.split_heads(encoder_output)
        
        a = self._attn(query, enc_key, enc_value, attention_mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a

class CustomLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self._init_rotary_embeddings()

    def _init_rotary_embeddings(self):
        dim = self.head_dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if query_states.shape[1] != self.num_heads:
            raise ValueError(f"Expected query_states dim 1 to be num_heads={self.num_heads}, got {query_states.shape[1]}")
        if key_states.shape[1] != self.num_heads:
            raise ValueError(f"Expected key_states dim 1 to be num_heads={self.num_heads}, got {key_states.shape[1]}")

        if position_embeddings is None:
            device = hidden_states.device
            position_ids = torch.arange(q_len, device=device).unsqueeze(0)
            t = position_ids.float()
            freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
            freqs = freqs.repeat(1, 1, 2)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            cos = cos.unsqueeze(1).expand(bsz, self.num_heads, q_len, self.head_dim)
            sin = sin.unsqueeze(1).expand(bsz, self.num_heads, q_len, self.head_dim)
            position_embeddings = (cos, sin)

        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, *position_embeddings)

        if past_key_values is not None:
            past_key, past_value = past_key_values
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        key_states_transposed = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states_transposed) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.size(0) != bsz:
                raise ValueError(f"Attention mask batch size {attention_mask.size(0)} does not match input batch size {bsz}")
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(bsz, self.num_heads, q_len, q_len)
            elif attention_mask.dim() == 4 and attention_mask.size(1) != self.num_heads:
                attention_mask = attention_mask.expand(bsz, self.num_heads, -1, -1)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if use_cache:
            past_key_values = (key_states, value_states)
        else:
            past_key_values = None
            
        return attn_output, attn_weights, past_key_values

class VisualLlamaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super(VisualLlamaBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = CustomLlamaAttention(config, layer_idx=layer_idx)
        self.cross_attention = CrossAttention(config.hidden_size, config.max_position_embeddings, config, scale=True)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc_alpha = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        encoder_output = kwargs.get("encoder_output", None)
        visual_attention_mask = kwargs.get("visual_attention_mask", None)
        mask_queries = kwargs.get("mask_queries", None)
        tau = kwargs.get("tau", 0.5)

        if position_ids is None:
            seq_length = hidden_states.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=None,
            past_key_values=past_key_values,
            cache_position=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache
        )

        self_attention_output = attn_outputs[0]
        if output_attentions:
            attentions = attn_outputs[1]
        if use_cache:
            past_key_values = attn_outputs[-1]

        hidden_states = residual + self_attention_output

        if encoder_output is not None:
            residual_cross = hidden_states
            cross_norm = self.post_attention_layernorm(hidden_states)

            cross_attn = self.cross_attention(
                cross_norm,
                encoder_output,
                visual_attention_mask
            )

            alpha_input = torch.cat([hidden_states, cross_attn], dim=-1)
            alpha = torch.sigmoid(self.fc_alpha(alpha_input))

            linguistics_mask = (alpha > tau).float()
            visual_mask = (alpha < (1 - tau)).float()

            hidden_states = (alpha * linguistics_mask * hidden_states +
                            (1 - alpha) * visual_mask * cross_attn)
            hidden_states = residual_cross + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (past_key_values,)

        return outputs

class VisualLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            VisualLlamaBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.post_init()

    def register_state(self, name, tensor, persistent=True):
        if hasattr(self, "register_buffer"):
            try:
                self.register_buffer(name, tensor, persistent=persistent)
            except TypeError:
                self.register_buffer(name, tensor)
        else:
            setattr(self, name, tensor)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_output=None,
        visual_attention_mask=None,
        mask_queries=None,
        tau=0.5
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        if attention_mask is not None:
            if isinstance(attention_mask, bool):
                attention_mask = torch.ones_like(input_ids, dtype=torch.float32) if attention_mask else torch.zeros_like(input_ids, dtype=torch.float32)
            attention_mask = attention_mask.view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if visual_attention_mask is not None:
            if isinstance(visual_attention_mask, bool):
                visual_attention_mask = torch.ones(encoder_output.size(0), encoder_output.size(1), dtype=torch.float32) if visual_attention_mask else torch.zeros(encoder_output.size(0), encoder_output.size(1), dtype=torch.float32)
            visual_attention_mask = visual_attention_mask[:, None, None, :]
            visual_attention_mask = (1.0 - visual_attention_mask) * torch.finfo(self.dtype).min

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
                encoder_output=encoder_output,
                visual_attention_mask=visual_attention_mask,
                mask_queries=mask_queries,
                tau=tau,
            )

            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)
            if use_cache:
                all_past_key_values = all_past_key_values + (outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions, all_past_key_values] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=all_past_key_values,
        )

class VisualLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = VisualLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_idx = config.pad_token_id if config.pad_token_id is not None else 0
        self.tau = config.tau
        self.model.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_output=None,
        visual_attention_mask=None
    ):
        # Debug prints
        print(f"Forward pass: input_ids shape={input_ids.shape if input_ids is not None else None}")
        print(f"Forward pass: attention_mask shape={attention_mask.shape if attention_mask is not None else None}")
        print(f"Forward pass: encoder_output is None: {encoder_output is None}")
        if encoder_output is not None:
            print(f"Forward pass: encoder_output shape={encoder_output.shape}")
        print(f"Forward pass: visual_attention_mask is None: {visual_attention_mask is None}")
        if visual_attention_mask is not None:
            print(f"Forward pass: visual_attention_mask shape={visual_attention_mask.shape}")
        print(f"Forward pass: past_key_values is None: {past_key_values is None}")
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        elif isinstance(attention_mask, bool):
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32) if attention_mask else torch.zeros_like(input_ids, dtype=torch.float32)
            
        if input_ids is not None:
            mask_queries = (input_ids != self.padding_idx)
            if isinstance(mask_queries, torch.Tensor):
                mask_queries = mask_queries.unsqueeze(-1).float()
            else:
                mask_queries = torch.tensor(mask_queries, device=input_ids.device).unsqueeze(-1).float()
        else:
            mask_queries = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], 1, 
                                   dtype=torch.float32, device=inputs_embeds.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_output=encoder_output,
            visual_attention_mask=visual_attention_mask,
            mask_queries=mask_queries,
            tau=self.tau
        )

        if return_dict:
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            hidden_states_all = outputs.hidden_states
            attentions = outputs.attentions
        else:
            hidden_states = outputs[0]
            hidden_states_all = outputs[1] if output_hidden_states else None
            attentions = outputs[2] if output_attentions else None
            past_key_values = outputs[3] if use_cache else None
        
        # Add this check to ensure past_key_values doesn't contain None
        if past_key_values is not None:
            for i, layer_past in enumerate(past_key_values):
                if layer_past is None or any(p is None for p in layer_past):
                    print(f"Warning: past_key_values contains None at layer {i}")
                    # Handle the None case appropriately, perhaps by not using cache
                    past_key_values = None
                    break
                    
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits, hidden_states_all, attentions, past_key_values)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states_all,
            attentions=attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]  # Only keep the last token
    
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
        
        # Always pass encoder_output and visual_attention_mask if they are in kwargs
        if "encoder_output" in kwargs:
            model_inputs["encoder_output"] = kwargs["encoder_output"]
        if "visual_attention_mask" in kwargs:
            model_inputs["visual_attention_mask"] = kwargs["visual_attention_mask"]
        
        return model_inputs
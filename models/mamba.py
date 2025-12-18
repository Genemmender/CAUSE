from typing import Optional

import torch
from transformers.models.mamba.modeling_mamba import MambaConfig as MambaConfigHF, MambaModel as MambaModelHF

from models.backbone import BackboneModel, BackboneConfig


class MambaConfig(BackboneConfig):
    pass


class MambaModel(BackboneModel):
    config: MambaConfig
    config_class = MambaConfig

    def __init__(self, config: MambaConfig):
        super().__init__(config=config)

        mamba_config = MambaConfigHF(
            vocab_size=2,
            hidden_size=self.config.embedding_dim,
            num_hidden_layers=self.config.num_layers,
        )

        self.model = MambaModelHF(config=mamba_config)

    def forward(
        self,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        hidden_states = inputs_embeds

        for mixer_block in self.model.layers:
            hidden_states = mixer_block(
                hidden_states,
                attention_mask=attention_mask,
            )

        return self.model.norm_f(hidden_states)

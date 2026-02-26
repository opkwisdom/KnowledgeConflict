from transformers import RobertaConfig

class SCAFormerConfig(RobertaConfig):
    model_type = "scaformer"
    
    def __init__(
        self,
        query_length=8,
        llm_width=4096,
        add_cross_attention=True,
        is_decoder=True,
        **kwargs
    ):
        super().__init__(
            add_cross_attention=add_cross_attention,
            is_decoder=is_decoder,
            **kwargs
        )
        self.query_length = query_length
        self.llm_width = llm_width

config = SCAFormerConfig.from_pretrained("roberta-base")
print(config)
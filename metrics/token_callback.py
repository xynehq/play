from transformers import TrainerCallback

class TokenCountCallback(TrainerCallback):
    def __init__(self):
        self.total_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        # kwargs may have 'inputs' depending on HF version; safer to hook in data_collator
        return control

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch._inductor.config


torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.set_float32_matmul_precision("high")


TADA_MODEL_ID = "HumeAI/tada-3b-ml"
TADA_ENCODER_ID = "HumeAI/tada-codec"
NUM_TRANSITION_STEPS = 5


def make_locked_inference_options():
    from tada.modules.tada import InferenceOptions
    return InferenceOptions(
        num_flow_matching_steps=20,
        num_acoustic_candidates=4,
        scorer="spkr_verification",
        spkr_verification_weight=1.0,
        acoustic_cfg_scale=1.8,
        duration_cfg_scale=1.0,
        noise_temperature=0.75,
        time_schedule="logsnr",
        speed_up_factor=1.2,
    )


@dataclass
class LoadedTada:
    encoder: object
    model: object
    tokenizer: object
    device: str
    model_id: str

    @classmethod
    def load(
        cls,
        device: str | None = None,
        model_id: str = TADA_MODEL_ID,
        encoder_id: str = TADA_ENCODER_ID,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LoadedTada":
        from tada.modules.encoder import Encoder
        from tada.modules.tada import TadaForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = Encoder.from_pretrained(encoder_id).to(device)
        model = TadaForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        return cls(
            encoder=encoder,
            model=model,
            tokenizer=model.tokenizer,
            device=device,
            model_id=model_id,
        )

    def encode_voice(self, audio: torch.Tensor, sample_rate: int):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device).float()
        audio = audio / audio.abs().max().clamp(min=1e-8)
        with torch.inference_mode():
            return self.encoder(audio, sample_rate=sample_rate)

    def compile(self) -> None:
        self.model.compile()

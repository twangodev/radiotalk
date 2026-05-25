from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


HIGGS_MODEL_ID = "eustlb/higgs-audio-v2-generation-3B-base"


@dataclass(frozen=True)
class HiggsInferenceOptions:
    max_new_tokens: int = 600
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class LoadedHiggs:
    """Wraps the upstream HiggsAudioV2 model + processor for voice-cloned TTS.

    The cloning prompt shape, per the HF model card:
        [system] Generate audio following instruction.
        [scene]  Audio is a VHF AM aviation radio transmission with mild radio noise.
        [user]   <voice reference transcript>
        [asst]   <voice reference audio>
        [user]   <target text>
    """
    model: object
    processor: object
    device: str
    model_id: str

    SYSTEM_TEXT: str = "Generate audio following instruction."
    SCENE_TEXT: str = (
        "Audio is a VHF AM aviation radio transmission with mild radio noise."
    )

    @classmethod
    def load(
        cls,
        device: str | None = None,
        model_id: str = HIGGS_MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LoadedHiggs":
        from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        model.eval()
        return cls(model=model, processor=processor, device=device, model_id=model_id)

    @property
    def sampling_rate(self) -> int:
        # 24 kHz matches voices-2k and the v1 dataset format.
        return 24000

    def build_conversation(
        self,
        voice_text: str,
        voice_ref_path: str | Path,
        text: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": [
                {"type": "text", "text": self.SYSTEM_TEXT}]},
            {"role": "scene", "content": [
                {"type": "text", "text": self.SCENE_TEXT}]},
            {"role": "user", "content": [
                {"type": "text", "text": voice_text.strip()}]},
            {"role": "assistant", "content": [
                {"type": "audio", "url": str(voice_ref_path)}]},
            {"role": "user", "content": [
                {"type": "text", "text": text}]},
        ]

from cog import BasePredictor, Input, Path

import json
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, Word, TranscriptionInfo
import time
from typing import Optional, List
from faster_whisper.tokenizer import _LANGUAGE_CODES

class Predictor(BasePredictor):
    use_flash_attention_2 = True

    def setup(self):
        model = "large-v2"

        # Run on GPU with FP16
        # self.model = WhisperModel(model, device="cuda", compute_type="float16")

        # or run on GPU with INT8
        self.model = WhisperModel(model, device="cuda", compute_type="int8_float16")
        
        # or run on CPU with INT8
        # model = WhisperModel(model, device="cpu", compute_type="int8")

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        language: str = Input(
            description="Language to use for ASR",
            default="autodetect",
            choices=["autodetect"] + list(_LANGUAGE_CODES),
        ),
        initial_prompt: str = Input(description="Initial prompt for decoder", default=""),
        beam_size: int = Input(description="Beam size",default=5,),
        vad_filter: bool = Input(description="Apply VAD (Silero) toggle", default=True),
    ) -> str:
        t = time.time()
        if language == 'autodetect':
            language = None
        segments, info = self.model.transcribe(
            str(audio),
            initial_prompt=initial_prompt if initial_prompt != "" else None,
            language=language, 
            beam_size=beam_size,
            vad_filter=vad_filter,
            )
        payload = {"segments": [convert_segment(s) for s in segments], "info": convert_transcription_info(info)}
        response = json.dumps(payload, ensure_ascii=False)
        print(f'processing took {time.time() - t:0.2f}s for {audio}')
        return response


def convert_segment(segment:Segment)->dict:
    seg = segment._asdict()
    words: Optional[List[Word]] = seg['words']
    if words is not None:
        seg['words'] = [word._asdict() for word in words]
    return seg

def convert_transcription_info(ti: TranscriptionInfo) -> dict :
    ti = ti._asdict()
    ti['transcription_options'] = ti['transcription_options']._asdict()
    ti['vad_options'] = ti['vad_options']._asdict()
    return ti

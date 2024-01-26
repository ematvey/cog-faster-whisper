import json
import math
import time

from cog import BasePredictor, Input, Path
from faster_whisper import WhisperModel
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
        initial_prompt: str = Input(
            description="Initial prompt for decoder", default=""
        ),
        beam_size: int = Input(description="Beam size", default=5),
        word_timestamps: bool = Input(description="Word timestamps", default=False),
        vad_filter: bool = Input(description="Apply VAD (Silero) toggle", default=True),
        condition_on_previous_text: bool = Input(
            description="If True, the previous output of the model is provided as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.",
            default=True,
        ),
    ) -> str:
        t = time.time()
        if language == "autodetect":
            language = None
        segments, info = self.model.transcribe(
            str(audio),
            initial_prompt=initial_prompt if initial_prompt != "" else None,
            language=language,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )
        payload = {
            "segments": [unpack(s) for s in segments],
            "info": unpack(info),
        }
        response = json.dumps(payload, ensure_ascii=False)
        print(f"processing took {time.time() - t:0.2f}s for {audio}")
        return response


def isnamedtupleinstance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i) == str for i in fields)


def unpack(obj):
    if isinstance(obj, dict):
        return {key: unpack(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: unpack(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(unpack(value) for value in obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        else:
            return obj
    else:
        return obj

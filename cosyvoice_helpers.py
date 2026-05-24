"""CosyVoice3 helpers: zero-shot and instruct (style) synthesis.

Usage:
    from cosyvoice_helpers import load_model, synthesize_v3, synthesize_v3_styled
    load_model(cpu_only=True)  # set False to use GPU
    synthesize_v3(...)
    synthesize_v3_styled(...)
"""
import os
import sys
import time
from pathlib import Path

# This file is expected to live at the root of the CosyVoice repo.
COSYVOICE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = COSYVOICE_ROOT / 'pretrained_models' / 'Fun-CosyVoice3-0.5B'

_cosyvoice = None
_torch = None
_torchaudio = None


def load_model(cpu_only=True, model_dir=None):
    """Load CosyVoice. Call once before synthesize_v3 / synthesize_v3_styled.

    Args:
        cpu_only: hide GPU from torch/ONNX. Needed for cards with <8GB VRAM.
        model_dir: override model path (defaults to CosyVoice2-0.5B in this repo).
    """
    global _cosyvoice, _torch, _torchaudio

    if cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    sys.path.insert(0, str(COSYVOICE_ROOT))
    sys.path.insert(0, str(COSYVOICE_ROOT / 'third_party' / 'Matcha-TTS'))

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    _torch = torch
    _torchaudio = torchaudio
    _cosyvoice = AutoModel(model_dir=str(model_dir or DEFAULT_MODEL_DIR))
    return _cosyvoice


def _check_loaded():
    if _cosyvoice is None:
        raise RuntimeError('Call load_model() first.')


def synthesize_v3(
    text,
    reference_wav,
    reference_text,
    output_wav='output.wav',
    system_prompt='You are a helpful assistant.',
):
    """CosyVoice3 zero-shot synthesis (voice cloning, neutral delivery).

    Args:
        text:           text to speak
        reference_wav:  path to a short clean clip (~3-10s) of the target voice
        reference_text: transcript of what is said in reference_wav
        output_wav:     where to write the synthesized audio
        system_prompt:  v3-style preamble; rarely needs changing
    """
    _check_loaded()
    start = time.perf_counter()
    prompt_text = f'{system_prompt}<|endofprompt|>{reference_text}'
    chunks = [j['tts_speech'] for j in _cosyvoice.inference_zero_shot(
        text, prompt_text, str(reference_wav), stream=False)]
    audio = _torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    _torchaudio.save(str(output_wav), audio, _cosyvoice.sample_rate)
    print(f'synthesized {output_wav} in {time.perf_counter() - start:.2f}s')
    return output_wav


def synthesize_v3_styled(
    text,
    instruction,
    reference_wav,
    output_wav='output.wav',
    system_prompt='You are a helpful assistant.',
):
    """CosyVoice3 instruct synthesis: `instruction` controls delivery style.

    No reference_text needed — instruct2 doesn't use it.

    Args:
        text:           text to speak
        instruction:    natural-language delivery instruction, e.g.:
                          'Speak slowly with a sad tone.'
                          'Speak in a British accent.'
                          '请用广东话表达。'
        reference_wav:  path to a short clean clip (~3-10s) of the target voice
        output_wav:     where to write the synthesized audio
        system_prompt:  v3-style preamble; rarely needs changing
    """
    _check_loaded()
    start = time.perf_counter()
    full_instruction = f'{system_prompt} {instruction}<|endofprompt|>'
    chunks = [j['tts_speech'] for j in _cosyvoice.inference_instruct2(
        text, full_instruction, str(reference_wav), stream=False)]
    audio = _torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    _torchaudio.save(str(output_wav), audio, _cosyvoice.sample_rate)
    print(f'synthesized {output_wav} in {time.perf_counter() - start:.2f}s')
    return output_wav


if __name__ == '__main__':
    load_model(cpu_only=True)

    ref = COSYVOICE_ROOT / 'asset' / 'zero_shot_prompt.wav'

    # Zero-shot: voice cloning, neutral delivery.
    synthesize_v3(
        text='Hello world, this is a test of CosyVoice version 3.',
        reference_wav=ref,
        reference_text='希望你以后能够做的比我还好呦。',
        output_wav='v3_zero_shot.wav',
    )

    # Styled: same voice, delivery shaped by the instruction.
    synthesize_v3_styled(
        text='Hello world, this is a test of CosyVoice version 3.',
        instruction='Speak slowly with a sad tone.',
        reference_wav=ref,
        output_wav='v3_styled.wav',
    )

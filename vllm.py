#!/usr/bin/env python3

import argparse
import sys
import os
import torch
import torchaudio
from tqdm import tqdm
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import logging

# Add third_party to path
sys.path.append('third_party/Matcha-TTS')

def get_args():
    parser = argparse.ArgumentParser(description="CosyVoice Inference CLI with vLLM and Apple Silicon support")

    # Model parameters
    parser.add_argument('--model_dir', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='Path to the model directory')
    parser.add_argument('--ttsfrd_path', type=str, default='pretrained_models/CosyVoice-ttsfrd',
                        help='Path to ttsfrd resources/package')

    # Inference inputs
    parser.add_argument('--text', type=str,
                        default='Hello, this is a test of the CosyVoice text to speech system.',
                        help='Text to synthesize')
    parser.add_argument('--prompt_text', type=str,
                        default='You are a helpful assistant.<|endofprompt|>',
                        help='Prompt text for zero-shot inference (include <|endofprompt|> for CosyVoice3)')
    parser.add_argument('--prompt_wav', type=str, default='./asset/zero_shot_prompt.wav',
                        help='Path to prompt audio file')
    parser.add_argument('--source_wav', type=str, default=None,
                        help='Path to source audio file for VC mode')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path')

    # Modes
    parser.add_argument('--mode', type=str, default='zero_shot',
                        choices=['zero_shot', 'sft', 'instruct', 'cross_lingual', 'vc'],
                        help='Inference mode')
    parser.add_argument('--spk_id', type=str, default='default',
                        help='Speaker ID for SFT/Instruct modes')
    parser.add_argument('--instruct_text', type=str, default='',
                        help='Instruction text for instruct mode')

    # Backend / Performance options
    parser.add_argument('--use_vllm', action='store_true', default=False,
                        help='Use vLLM for inference (Linux/CUDA only typically)')
    parser.add_argument('--use_trt', action='store_true', default=False,
                        help='Use TensorRT (Linux/CUDA only)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use FP16 precision')
    parser.add_argument('--no_stream', action='store_true',
                        help='Disable streaming output')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed factor (1.0 is normal)')
    parser.add_argument('--no_text_frontend', action='store_false', dest='text_frontend', default=True,
                        help='Disable text frontend normalization')

    args = parser.parse_args()
    return args

def register_vllm_if_needed(use_vllm):
    if not use_vllm:
        return

    try:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        logging.info("Registered CosyVoice2ForCausalLM with vLLM")
    except ImportError as e:
        logging.warning(f"Failed to import vLLM: {e}. Falling back to standard inference.")

def main():
    args = get_args()

    # Apple Silicon / Platform optimizations
    if sys.platform == 'darwin':
        logging.info("Running on macOS/Apple Silicon")
        # if args.use_vllm:
        #     logging.warning("vLLM typically not supported on macOS. Disabling vLLM.")
        #     args.use_vllm = False
        if args.use_trt:
            logging.warning("TensorRT not supported on macOS. Disabling TensorRT.")
            args.use_trt = False
        # Ensure we don't try to use CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Handle ttsfrd path
    if args.ttsfrd_path and os.path.exists(args.ttsfrd_path):
        logging.info(f"Adding {args.ttsfrd_path} to sys.path")
        sys.path.append(args.ttsfrd_path)

    # Register vLLM model
    if args.use_vllm:
        register_vllm_if_needed(True)

    from cosyvoice.cli.cosyvoice import AutoModel

    logging.info(f"Initializing model from {args.model_dir}")
    try:
        cosyvoice = AutoModel(
            model_dir=args.model_dir,
            load_trt=args.use_trt,
            load_vllm=args.use_vllm,
            fp16=args.fp16
        )
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    logging.info(f"Inference mode: {args.mode}")
    set_all_random_seed(1234) # Use fixed seed or allow param? Using fixed for reproducibility

    output_generator = None

    try:
        if args.mode == 'zero_shot':
            # Check prompt for CosyVoice3
            prompt = args.prompt_text
            if 'CosyVoice3' in args.model_dir or 'Fun-CosyVoice3' in args.model_dir:
                if '<|endofprompt|>' not in prompt:
                    logging.warning("CosyVoice3 detected but <|endofprompt|> missing in prompt_text. Appending it to the end.")
                    prompt = prompt + '<|endofprompt|>'

            output_generator = cosyvoice.inference_zero_shot(
                args.text, prompt, args.prompt_wav, stream=not args.no_stream, speed=args.speed, text_frontend=args.text_frontend
            )

        elif args.mode == 'sft':
            output_generator = cosyvoice.inference_sft(
                args.text, args.spk_id, stream=not args.no_stream, speed=args.speed, text_frontend=args.text_frontend
            )

        elif args.mode == 'instruct':
            if cosyvoice.__class__.__name__ == 'CosyVoice':
                output_generator = cosyvoice.inference_instruct(
                    args.text, args.spk_id, args.instruct_text, stream=not args.no_stream, speed=args.speed, text_frontend=args.text_frontend
                )
            else:
                # CosyVoice2 or CosyVoice3 use inference_instruct2 which is zero-shot based
                logging.info(f"Using inference_instruct2 for {cosyvoice.__class__.__name__}")
                output_generator = cosyvoice.inference_instruct2(
                    args.text, args.instruct_text, args.prompt_wav, stream=not args.no_stream, speed=args.speed, text_frontend=args.text_frontend
                )

        elif args.mode == 'cross_lingual':
            output_generator = cosyvoice.inference_cross_lingual(
                args.text, args.prompt_wav, stream=not args.no_stream, speed=args.speed, text_frontend=args.text_frontend
            )

        elif args.mode == 'vc':
            if not args.source_wav:
                 logging.error("VC mode requires --source_wav argument.")
                 sys.exit(1)
            output_generator = cosyvoice.inference_vc(
                args.source_wav, args.prompt_wav, stream=not args.no_stream, speed=args.speed
            )

        # Collect audio
        result_wavs = []
        for i, model_output in enumerate(tqdm(output_generator, desc="Synthesizing")):
            result_wavs.append(model_output['tts_speech'])

        if result_wavs:
            final_wav = torch.cat(result_wavs, dim=1)
            torchaudio.save(args.output, final_wav, cosyvoice.sample_rate)
            logging.info(f"Saved generated audio to {args.output}")
        else:
            logging.warning("No audio generated.")

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

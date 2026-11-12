# Local web UI for Video-Depth-Anything that works on CPU or GPU.
# Run:  python app_local.py   -> opens http://127.0.0.1:7860 in your browser.
import os
import glob
import time

import numpy as np

import gradio as gr
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ON_CPU = DEVICE == 'cpu'

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
LABELS = {'vits': 'Small (fast)', 'vitb': 'Base', 'vitl': 'Large (best, slow)'}


def ckpt_path(key):
    # key is 'vits' (relative) or 'metric_vits' (metric depth, in meters)
    if key.startswith('metric_'):
        return f'./checkpoints/metric_video_depth_anything_{key[7:]}.pth'
    return f'./checkpoints/video_depth_anything_{key}.pth'


def label(key):
    if key.startswith('metric_'):
        return '(Metric) ' + LABELS[key[7:]]
    return '(Relative) ' + LABELS[key]


def available_encoders():
    found = []
    for enc in ['vits', 'vitb', 'vitl', 'metric_vits', 'metric_vitb', 'metric_vitl']:
        if os.path.exists(ckpt_path(enc)):
            found.append(enc)
    return found


_models = {}


def get_model(encoder):
    if encoder not in _models:
        _models.clear()  # keep only one model in memory
        metric = encoder.startswith('metric_')
        model = VideoDepthAnything(**MODEL_CONFIGS[encoder.replace('metric_', '')], metric=metric)
        state = torch.load(ckpt_path(encoder), map_location='cpu')
        model.load_state_dict(state, strict=True)
        _models[encoder] = model.to(DEVICE).eval()
    return _models[encoder]


def run(input_video, encoder, max_len, target_fps, max_res, grayscale, progress=gr.Progress(track_tqdm=True)):
    if not input_video:
        raise gr.Error('Please upload a video first.')
    start = time.time()
    model = get_model(encoder)
    frames, fps = read_video_frames(input_video, int(max_len), int(target_fps), int(max_res))
    depths, fps = model.infer_video_depth(frames, fps, input_size=518, device=DEVICE, fp32=ON_CPU)

    out_dir = './outputs'
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_video))[0]
    src_path = os.path.join(out_dir, f'{name}_src.mp4')
    vis_path = os.path.join(out_dir, f'{name}_vis.mp4')
    save_video(frames, src_path, fps=fps)
    save_video(depths, vis_path, fps=fps, is_depths=True, grayscale=grayscale)
    msg = f'Done: {len(frames)} frames in {time.time() - start:.0f}s on {DEVICE.upper()}. Saved to {os.path.abspath(vis_path)}'
    if encoder.startswith('metric_'):
        d = np.asarray(depths, dtype=np.float32)
        npz_path = os.path.join(out_dir, f'{name}_depths_meters.npz')
        np.savez_compressed(npz_path, depths=d)
        msg += (f'\n\n**Metric depth (meters):** min {d.min():.2f} m, median {np.median(d):.2f} m, max {d.max():.2f} m'
                f'\n\nRaw per-pixel depth in meters saved to {os.path.abspath(npz_path)}')
    return src_path, vis_path, msg


def build():
    encoders = available_encoders()
    if not encoders:
        raise SystemExit('No checkpoints found in ./checkpoints - run download_weights.bat first.')
    with gr.Blocks(title='Video Depth Anything') as demo:
        gr.Markdown(
            '# Video Depth Anything\n'
            f'Running on **{DEVICE.upper()}**.'
            + (' CPU is slow: keep videos short and use the defaults below.' if ON_CPU else '')
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(label='Upload video')
                encoder = gr.Dropdown(choices=[(label(e), e) for e in encoders], value=encoders[0], label='Model')
                with gr.Accordion('Settings', open=True):
                    max_len = gr.Slider(-1, 1000, value=60 if ON_CPU else -1, step=1,
                                        label='Max frames (-1 = whole video)')
                    target_fps = gr.Slider(-1, 30, value=10 if ON_CPU else -1, step=1,
                                           label='Target FPS (-1 = original)')
                    max_res = gr.Slider(256, 1920, value=640 if ON_CPU else 1280, step=2,
                                        label='Max resolution (longest side)')
                    grayscale = gr.Checkbox(False, label='Grayscale depth (no color palette)')
                btn = gr.Button('Generate depth', variant='primary')
            with gr.Column(scale=2):
                with gr.Row():
                    src_out = gr.Video(label='Processed input', interactive=False, autoplay=True, loop=True)
                    depth_out = gr.Video(label='Depth video', interactive=False, autoplay=True, loop=True)
                status = gr.Markdown()
        btn.click(run, [input_video, encoder, max_len, target_fps, max_res, grayscale],
                  [src_out, depth_out, status])
        gr.Examples([['assets/example_videos/davis_rollercoaster.mp4']], inputs=[input_video])
    return demo


if __name__ == '__main__':
    build().queue().launch(inbrowser=True)

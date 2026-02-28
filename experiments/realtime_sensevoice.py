#!/usr/bin/env python3
"""
基于仓库内 SenseVoiceSmall 的实时录音转写脚本。
录音设备检测到说话开始后进入录音，检测到持续静默（--silence 秒）则结束会话并退出。
依赖: pip install funasr sounddevice soundfile numpy
使用方法示例:
  python realtime_sensevoice.py --model ./SenseVoiceSmall --silence 1.5
"""
import argparse
import queue
import threading
import sys
import time
import tempfile
import os

import numpy as np
import sounddevice as sd
import soundfile as sf

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def choose_samplerate(device, requested):
    import sounddevice as _sd
    # 如果用户指定采样率，先尝试检查
    if requested is not None:
        try:
            _sd.check_input_settings(device=device, samplerate=requested)
            return int(requested)
        except Exception:
            pass

    # 查询设备默认采样率
    try:
        info = _sd.query_devices(device)
        default = info.get('default_samplerate', None)
        if default and default > 0:
            try:
                _sd.check_input_settings(device=device, samplerate=int(default))
                return int(default)
            except Exception:
                pass
    except Exception:
        pass

    # 逐个尝试常用采样率
    for r in (16000, 22050, 32000, 44100, 48000):
        try:
            _sd.check_input_settings(device=device, samplerate=r)
            return int(r)
        except Exception:
            continue

    raise RuntimeError('无法找到设备支持的采样率，请使用 --device 指定其它设备或检查系统设置。')


def record_session(model_dir, device, samplerate, silence_timeout, threshold):
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # 选择可用采样率（先确定采样率再创建模型）
    import sounddevice as sd
    try:
        used_samplerate = choose_samplerate(device, samplerate)
    except RuntimeError as e:
        print(e)
        return 1

    print(f'Using samplerate: {used_samplerate}')
    print('Loading model (in background)...')

    # 创建模型对象并在后台预热以并行加载 heavy 资源
    model = AutoModel(model=model_dir, trust_remote_code=True, remote_code=os.path.join(model_dir, 'model.py'), device='cpu')

    def _warmup(m, sr):
        try:
            import tempfile as _tempfile, soundfile as _sf, numpy as _np
            with _tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as _f:
                tmp = _f.name
            _sf.write(tmp, _np.zeros(int(0.1 * sr), dtype=_np.int16), sr)
            try:
                # 轻量调用以触发模型加载
                m.generate(input=tmp, language='auto', use_itn=True, batch_size_s=1)
            except Exception:
                pass
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception:
            pass

    import threading as _thr
    t_warm = _thr.Thread(target=_warmup, args=(model, used_samplerate), daemon=True)
    t_warm.start()

    print('Listening... speak to start. Session ends after {:.1f}s silence.'.format(silence_timeout))
    stop_event = threading.Event()

    def wait_enter():
        try:
            input('\nPress Enter to stop early.\n')
            stop_event.set()
        except Exception:
            pass

    t_stop = threading.Thread(target=wait_enter, daemon=True)
    t_stop.start()

    recording = False
    buffer = np.array([], dtype=np.int16)
    last_spoken_time = None

    with sd.InputStream(samplerate=used_samplerate, channels=1, dtype='int16', callback=callback, device=device):
        try:
            while True:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    # check silence after speech started
                    if stop_event.is_set():
                        print('\nStop requested by user')
                        break

                    if recording and last_spoken_time is not None and time.time() - last_spoken_time > silence_timeout:
                        # end session
                        break
                    continue

                chunk = chunk.flatten()
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                # 如果开启 verbose，可见实时能量
                if getattr(record_session, 'verbose', False):
                    print(f'RMS: {rms:.1f}', end='\r')
                if not recording:
                    if rms > threshold:
                        recording = True
                        print('Detected speech, recording...')
                        buffer = np.concatenate([buffer, chunk])
                        last_spoken_time = time.time()
                else:
                    buffer = np.concatenate([buffer, chunk])
                    if rms > threshold:
                        last_spoken_time = time.time()

        except KeyboardInterrupt:
            print('\nInterrupted by user')

    if buffer.size == 0:
        print('No speech captured.')
        return 1

    # write to temp wav and run model.generate
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_wav = f.name
    sf.write(tmp_wav, buffer.astype(np.int16), samplerate)

    try:
        res = model.generate(input=tmp_wav, language='auto', use_itn=True, batch_size_s=60)
        text = res[0]['text'] if res and isinstance(res, list) else ''
        text = rich_transcription_postprocess(text)
        print('\nRecognition result:')
        print(text)
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

    return 0


def parse_args():
    p = argparse.ArgumentParser(description='Realtime STT using SenseVoiceSmall')
    p.add_argument('--model', '-m', default='./SenseVoiceSmall', help='SenseVoiceSmall 模型目录')
    p.add_argument('--device', '-d', type=int, default=None, help='输入设备索引（sounddevice 列表索引）')
    p.add_argument('--samplerate', '-r', type=int, default=16000, help='采样率')
    p.add_argument('--silence', '-s', type=float, default=1.5, help='静默多长时间（秒）结束会话')
    p.add_argument('--threshold', '-t', type=float, default=300.0, help='能量阈值，数值越小越敏感')
    p.add_argument('--max-duration', type=float, default=0.0, help='最大录音时长（秒），0 表示不限制')
    p.add_argument('--verbose', action='store_true', help='打印实时 RMS 值便于调试')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 将 verbose 传入函数属性以便回显
    record_session.verbose = args.verbose
    sys.exit(record_session(args.model, args.device, args.samplerate, args.silence, args.threshold))

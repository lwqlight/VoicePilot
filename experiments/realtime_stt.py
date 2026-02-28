#!/usr/bin/env python3
"""
实时麦克风语音转文字示例（基于 VOSK + sounddevice）
按说话，检测到一段静默后认为对话结束并退出。
依赖: pip install vosk sounddevice
把离线模型解压到某个目录并通过参数 --model 指定。
"""
import argparse
import json
import os
import queue
import sys
import time

import sounddevice as sd
from vosk import Model, KaldiRecognizer


def int16_bytes(data):
    return data.tobytes()


def record_and_recognize(model_path, device, samplerate, silence_timeout):
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return 1

    model = Model(model_path)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16', channels=1, callback=callback, device=device):
            rec = KaldiRecognizer(model, samplerate)
            print('Start speaking. Silence for {:.1f}s ends session.'.format(silence_timeout))
            last_spoken_time = time.time()
            partial_shown = ''
            while True:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    # 检查静默超时
                    if time.time() - last_spoken_time > silence_timeout:
                        print('\nDetected silence timeout — exiting.')
                        break
                    continue

                raw = int16_bytes(data)
                if rec.AcceptWaveform(raw):
                    res = json.loads(rec.Result())
                    text = res.get('text', '').strip()
                    if text:
                        print('\nFinal:', text)
                        last_spoken_time = time.time()
                    partial_shown = ''
                else:
                    partial = json.loads(rec.PartialResult()).get('partial', '')
                    if partial and partial != partial_shown:
                        # 覆盖同一行显示实时 partial
                        print('Partial: ' + partial, end='\r')
                        partial_shown = partial
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    return 0


def parse_args():
    p = argparse.ArgumentParser(description='实时麦克风语音转文字（VOSK）')
    p.add_argument('--model', '-m', required=True, help='VOSK 模型目录路径')
    p.add_argument('--device', '-d', type=int, default=None, help='输入设备索引（sounddevice 列表索引）')
    p.add_argument('--samplerate', '-r', type=int, default=16000, help='采样率，模型通常为 16000')
    p.add_argument('--silence', '-s', type=float, default=1.5, help='静默多长时间（秒）结束会话')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sys.exit(record_and_recognize(args.model, args.device, args.samplerate, args.silence))

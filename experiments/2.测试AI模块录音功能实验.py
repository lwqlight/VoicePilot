import pygame
import time
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="output.wav", sample_rate=48000):
    print("按下 Enter 开始录音...")
    input()  # 等待用户按下 Enter 键开始录音
    print("录音中... 按下 Enter 键结束录音")
    
    # 开始录音
    recording = []
    try:
        def callback(indata, frames, time, status):
            recording.append(indata.copy())
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            input()  # 等待用户再次按下 Enter 键结束录音
    except Exception as e:
        print(f"录音出现错误: {e}")
        return
    
    # 将录音数据合并并保存为 WAV 文件
    audio_data = np.concatenate(recording, axis=0)
    write(filename, sample_rate, (audio_data * 32767).astype(np.int16))
    print(f"录音已保存为 {filename}")


# --- 播放音频（新增音量调整功能）---
def play_audio(file_path, volume=0.05):  # volume默认0.05（5%），范围0.0-1.0
    try:
        pygame.mixer.init()
        # 设置全局音量（0.0=静音，1.0=最大音量，0.05=5%）
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)  # 缩短轮询间隔，更流畅
        print(f"播放完成！当前播放音量为 {volume*100}%")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

# 使用函数录音，作为输入
record_audio("./example/my_recording.wav")
# 播放音频，指定音量为5%（也可省略参数，使用函数默认值）
play_audio('./example/my_recording.wav', volume=0.05)
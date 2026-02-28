import pyaudio
import wave
import threading
import numpy as np
import time
import os
import torch
import pygame
import edge_tts
import asyncio
import langid
import webrtcvad
import traceback
import re
from queue import Queue
from funasr import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 配置类 ---
class Config:
    HF_ENDPOINT = 'https://hf-mirror.com'
    OUTPUT_DIR = "./output"
    MODEL_DIR_SENSEVOICE = "./SenseVoiceSmall"
    MODEL_NAME_QWEN = "./Qwen2.5-0.5B-Instruct"
    
    DEVICE = "cpu" 
    
    AUDIO_RATE = 48000      
    AUDIO_CHANNELS = 1
    CHUNK = 2048           
    
    VAD_MODE = 1              
    NO_SPEECH_THRESHOLD = 0.8 
    
    SYSTEM_PROMPT = "你叫千问，是一个18岁的女大学生，性格活泼开朗。请用简短的语言回答（50字以内）。"

os.environ['HF_ENDPOINT'] = Config.HF_ENDPOINT
os.environ["OMP_NUM_THREADS"] = "4" 

class VoiceAssistant:
    def __init__(self):
        self.running = False
        self.segments_to_save = []
        self.audio_file_count = 0
        self.last_active_time = time.time()
        
        # --- 新增：忙碌状态标志 ---
        # True = 正在推理或播放，不听录音
        # False = 空闲，可以听录音
        self.is_busy = False 
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(Config.VAD_MODE)
        
        pygame.mixer.init()
        
        print(f">>> [系统] 正在树莓派 CPU 上加载模型...")
        self._load_models()
        print(">>> [系统] 模型加载完成！")

    def _load_models(self):
        try:
            print(f" -> 加载 ASR: {Config.MODEL_DIR_SENSEVOICE}")
            self.asr_model = AutoModel(
                model=Config.MODEL_DIR_SENSEVOICE,
                trust_remote_code=True,
                remote_code=os.path.join(Config.MODEL_DIR_SENSEVOICE, "model.py"),
                device=Config.DEVICE,
                disable_update=True,
            )
            
            print(f" -> 加载 LLM: {Config.MODEL_NAME_QWEN}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME_QWEN)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME_QWEN,
                torch_dtype=torch.float32,
                device_map=Config.DEVICE
            )
            self.llm_model.eval()
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            traceback.print_exc()
            exit(1)

    def check_vad_activity(self, audio_data):
        try:
            window_duration = 0.02 
            bytes_per_window = int(Config.AUDIO_RATE * window_duration * 2) 
            num_speech = 0
            total_windows = 0
            for i in range(0, len(audio_data), bytes_per_window):
                chunk = audio_data[i:i + bytes_per_window]
                if len(chunk) == bytes_per_window:
                    total_windows += 1
                    if self.vad.is_speech(chunk, sample_rate=Config.AUDIO_RATE):
                        num_speech += 1
            if total_windows == 0: return False
            return (num_speech / total_windows) > 0.6
        except Exception:
            return False

    def save_audio_segment(self):
        if not self.segments_to_save:
            return None

        total_len = len(self.segments_to_save) * Config.CHUNK
        duration = total_len / Config.AUDIO_RATE
        if duration < 0.6:
            print(f"[忽略] 噪音太短 ({duration:.1f}s)")
            self.segments_to_save.clear()
            return None

        self.audio_file_count += 1
        filename = f"{Config.OUTPUT_DIR}/temp_audio_{self.audio_file_count}.wav"
        
        audio_frames = [seg for seg, _ in self.segments_to_save]
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(Config.AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(Config.AUDIO_RATE)
            wf.writeframes(b''.join(audio_frames))
        
        self.segments_to_save.clear()
        return filename

    def clean_asr_text(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()

    def process_inference(self, audio_path):
        """执行推理，并管理 busy 状态"""
        if not audio_path or not os.path.exists(audio_path): 
            # 如果文件无效，立即解除忙碌状态
            self.is_busy = False
            print(">>> [状态] 恢复监听")
            return

        try:
            print(f"\n--- 停止监听，开始处理 ---")
            
            # 1. ASR
            res = self.asr_model.generate(input=audio_path, cache={}, language="auto", use_itn=False)
            raw_text = res[0].get('text', "") if isinstance(res, list) else res.get('text', "")
            user_text = self.clean_asr_text(raw_text)
            print(f"┌── [识别结果]: {user_text}")

            # 2. 逻辑过滤
            if len(user_text) < 2 and user_text in ["嗯", "啊", "哦", "。", "？"]:
                print(f"└── [忽略] 无效输入")
                return # 这里的 return 会触发 finally 解除忙碌状态

            if not user_text:
                print(f"└── [忽略] 空内容")
                return

            # 3. LLM 推理
            print("│   正在思考...", end="", flush=True)
            messages = [
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            text_input = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text_input], return_tensors="pt").to(Config.DEVICE)

            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            ai_response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"\r└── [千问回复]: {ai_response}")

            # 4. TTS 与播放
            if ai_response:
                self.text_to_speech_and_play(ai_response)

        except Exception as e:
            print(f"\n[Error] {e}")
            traceback.print_exc()
        finally:
            # --- 关键：无论成功还是失败，最后都要删除文件并恢复监听 ---
            if os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass
            
            # 解除忙碌状态，允许再次录音
            self.is_busy = False
            print(f">>> [状态] 恢复监听 (请说话)...")

    def text_to_speech_and_play(self, text):
        tts_file = os.path.join(Config.OUTPUT_DIR, f"temp_tts_{self.audio_file_count}.mp3")
        try:
            voice = "zh-CN-XiaoyiNeural"
            asyncio.run(self._edge_tts_save(text, voice, tts_file))
            
            pygame.mixer.music.load(tts_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload() 
            
        except Exception as e:
            print(f"[TTS Error] {e}")
        finally:
            if os.path.exists(tts_file):
                try: os.remove(tts_file)
                except: pass

    async def _edge_tts_save(self, text, voice, output_file):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    def audio_listener_loop(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=Config.AUDIO_CHANNELS,
                            rate=Config.AUDIO_RATE,
                            input=True,
                            frames_per_buffer=Config.CHUNK)
        except Exception as e:
            print(f"\n!!! 麦克风错误: {e}")
            return

        print("\n>>> 监听中... (对着麦克风说话)")
        self.running = True
        audio_buffer = [] 
        
        while self.running:
            try:
                # 无论是否忙碌，都要读取数据，防止缓冲区溢出
                data = stream.read(Config.CHUNK, exception_on_overflow=False)
                
                # --- 关键修改：如果系统忙碌，直接丢弃数据，不进行VAD ---
                if self.is_busy:
                    audio_buffer = [] # 清空缓存，防止积压
                    time.sleep(0.01)  # 稍微让出CPU
                    continue
                
                # --- 只有不忙碌时，才进行语音活动检测 ---
                audio_buffer.append(data)
                
                if len(audio_buffer) >= 5: 
                    raw_audio = b''.join(audio_buffer)
                    is_speech = self.check_vad_activity(raw_audio)
                    
                    if is_speech:
                        self.last_active_time = time.time()
                        self.segments_to_save.append((raw_audio, time.time()))
                    
                    audio_buffer = [] 

                if time.time() - self.last_active_time > Config.NO_SPEECH_THRESHOLD:
                    if self.segments_to_save:
                        wav_path = self.save_audio_segment()
                        if wav_path:
                            # 1. 标记为忙碌
                            self.is_busy = True
                            # 2. 启动处理线程
                            threading.Thread(target=self.process_inference, args=(wav_path,)).start()
                        
                        self.last_active_time = time.time()

            except IOError:
                continue
            except Exception as e:
                print(e)
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def start(self):
        try:
            self.audio_listener_loop()
        except KeyboardInterrupt:
            self.running = False
            print("\n退出中...")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.start()
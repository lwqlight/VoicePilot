import pyaudio
import wave
import threading
import numpy as np
import time
import os
import pygame
import edge_tts
import asyncio
import traceback
import re
from funasr import AutoModel
from llama_cpp import Llama
 

# --- 配置类 ---
class Config:
    OUTPUT_DIR = "./output"
    MODEL_DIR_SENSEVOICE = "./SenseVoiceSmall"
    # 你的 GGUF 模型路径
    MODEL_PATH_LLM = "./qwen3-0.6B-gguf/qwen3_0.6B_q4_k_m.gguf"
    
    DEVICE = "cpu" 
    
    AUDIO_RATE = 48000  
    AUDIO_CHANNELS = 1
    CHUNK = 4096           # 加大缓冲区，防止爆音
    
    # --- 关键修改：音量阈值 ---
    # 麦克风收音很小时，调小这个值（如 300）
    # 麦克风很灵敏时，调大这个值（如 1000-2000）
    MIN_VOLUME = 500       
    
    # 静音等待时间：说完话后停顿多久算结束
    # 静音自动关闭时间：说完话间隔SILENCE_TIMEOUT秒后自动关闭录音
    SILENCE_TIMEOUT = 1.0  
    
    SYSTEM_PROMPT = "你叫千问，是一个18岁的女大学生，性格活泼开朗。请用简短的语言回答（50字以内）。注意：不需要思考，直接输出。"

os.environ["OMP_NUM_THREADS"] = "4"

class VoiceAssistant:
    def __init__(self):
        self.is_busy = False 
        self.audio_file_count = 0
        
        # 录音相关状态
        self.recording = False      # 正在录音标志
        self.frames = []            # 音频缓存
        self.last_speech_time = 0   # 上次听到声音的时间
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # 移除 VAD，只初始化播放器
        pygame.mixer.init()
        
        print(f">>> [系统] 正在初始化 (纯音量触发版)...")
        self._load_models()
        print(">>> [系统] 全部模型加载完成！")

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
            
            print(f" -> 加载 LLM: {Config.MODEL_PATH_LLM}")
            if not os.path.exists(Config.MODEL_PATH_LLM):
                raise FileNotFoundError(f"找不到模型文件: {Config.MODEL_PATH_LLM}")

            self.llm = Llama(
                model_path=Config.MODEL_PATH_LLM,
                n_ctx=1024,
                n_gpu_layers=0,
                n_threads=4,
                n_batch=512,       
                use_mmap=False,
                verbose=False
            )
            
            # 预热
            print(" -> 正在预热 LLM...")
            self.llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            traceback.print_exc()
            exit(1)

    # --- 新增：计算音量函数 ---
    def calculate_volume(self, audio_data):
        # 将字节流转为 numpy 数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # 计算绝对值的平均值作为音量（也可以用 RMS）
        if len(audio_array) == 0: return 0
        return np.mean(np.abs(audio_array))

    def save_audio(self):
        if not self.frames: return None
        
        # 时长过滤
        duration = (len(self.frames) * Config.CHUNK) / Config.AUDIO_RATE
        if duration < 0.5:
            print(f"[忽略] 声音太短 ({duration:.2f}s)")
            self.frames = []
            return None

        self.audio_file_count += 1
        filename = f"{Config.OUTPUT_DIR}/audio_{self.audio_file_count}.wav"
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(Config.AUDIO_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(Config.AUDIO_RATE)
            wf.writeframes(b''.join(self.frames))
        
        self.frames = []
        return filename

    def clean_asr_text(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()

    def process_inference(self, audio_path):
        if not audio_path or not os.path.exists(audio_path): 
            self.is_busy = False
            return

        try:
            print(f"\n--- 处理中 ---")
            t_start = time.time()
            
            # ASR
            res = self.asr_model.generate(input=audio_path, cache={}, language="auto", use_itn=False)
            raw_text = res[0].get('text', "") if isinstance(res, list) else res.get('text', "")
            user_text = self.clean_asr_text(raw_text)
            print(f"┌── [听到]: {user_text}")

            if len(user_text) < 1 or user_text in ["嗯", "。", "？"]:
                print(f"└── [忽略] 无效")
                return 

            # LLM
            print("│   思考中...", end="", flush=True)
            messages = [
                {"role": "system", "content": Config.SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
            
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7,
            )
            
            ai_response = output['choices'][0]['message']['content']
            t_cost = time.time() - t_start
            print(f"\r└── [回复] ({t_cost:.2f}s): {ai_response}")

            # TTS
            if ai_response:
                self.text_to_speech_and_play(ai_response)

        except Exception as e:
            print(f"\n[Error] {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass
            
            self.is_busy = False
            self.recording = False # 确保重置录音状态
            print(f">>> [状态] 恢复监听...")

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
        stream = p.open(format=pyaudio.paInt16,
                        channels=Config.AUDIO_CHANNELS,
                        rate=Config.AUDIO_RATE,
                        input=True,
                        frames_per_buffer=Config.CHUNK)

        print("\n>>> 监听中 (基于音量触发)...")
        self.running = True
        
        while self.running:
            try:
                data = stream.read(Config.CHUNK, exception_on_overflow=False)
                
                # 忙碌时不处理音频
                if self.is_busy:
                    time.sleep(0.02)
                    continue

                # --- 核心修改：音量检测逻辑 ---
                volume = self.calculate_volume(data)
                
                # 可视化音量条 (调试用，如果太刷屏可以注释掉)
                # if volume > 100: print(f"音量: {int(volume)}")
                # print(f"音量: {int(volume)}", end='\r') 短促的声音将不会录入，避免误触发

                if not self.recording:
                    # [状态：等待说话]
                    if volume > Config.MIN_VOLUME:
                        print(f"[触发] 检测到声音 (Vol:{int(volume)})")
                        self.recording = True
                        self.frames = [data]
                        self.last_speech_time = time.time()
                else:
                    # [状态：正在录音]
                    self.frames.append(data)
                    
                    if volume > Config.MIN_VOLUME:
                        # 只要还在说话，就刷新计时器
                        self.last_speech_time = time.time()
                    
                    # 如果持续安静超过设定时间，认为说话结束
                    if time.time() - self.last_speech_time > Config.SILENCE_TIMEOUT:
                        print("[结束] 说话结束")
                        wav_path = self.save_audio()
                        if wav_path:
                            self.is_busy = True
                            threading.Thread(target=self.process_inference, args=(wav_path,)).start()
                        else:
                            self.recording = False # 没保存成功（太短），重置状态

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

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.start()
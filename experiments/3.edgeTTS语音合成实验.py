#!/usr/bin/env python3
"""
增强版 edge_tts 文字转语音工具
支持自定义文本/语音/输出文件，自动播放生成的音频并调整音量
"""
import asyncio
import pygame
import edge_tts

# ===================== 可自定义配置 =====================
TEXT = "你好，我的名字是Edge TTS。"  # 支持多语言混合
VOICE = "zh-CN-XiaoyiNeural"  # 语音名称（可替换为下方列表中的值）
OUTPUT_FILE = "tts_output.mp3"  # 输出音频文件路径
PLAY_VOLUME = 0.2  # 播放音量（0.0-1.0，0.05=5%）
# ========================================================

# 常用语音列表（按需选择）
VOICE_LIST = {
    # 中文
    "zh-CN-YunjianNeural": "中文-云健（男）",
    "zh-CN-YunxiNeural": "中文-云希（男）",
    "zh-CN-XiaoyiNeural": "中文-小艺（女）",
    # 英文
    "en-US-AnaNeural": "英文-安娜（女）",
    "en-US-AndrewNeural": "英文-安德鲁（男）",
    # 日语
    "ja-JP-NanamiNeural": "日语-七海（女）",
    # 法语
    "fr-FR-DeniseNeural": "法语-丹尼斯（女）",
    # 西班牙语
    "es-ES-JoanaNeural": "西班牙语-乔安娜（女）"
}

async def text_to_speech(text: str, voice: str, output_file: str) -> None:
    """
    文字转语音核心函数
    :param text: 要转换的文本
    :param voice: 语音名称
    :param output_file: 输出音频文件路径
    """
    try:
        print(f"开始转换文字到语音，使用语音：{VOICE_LIST.get(voice, voice)}")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"语音文件已保存至：{output_file}")
    except Exception as e:
        print(f"文字转语音失败：{e}")
        raise

def play_audio(file_path: str, volume: float = 1.0) -> None:
    """
    播放音频文件（支持音量调整）
    :param file_path: 音频文件路径
    :param volume: 播放音量（0.0-1.0）
    """
    try:
        # 初始化pygame音频模块
        pygame.mixer.init()
        # 设置播放音量
        pygame.mixer.music.set_volume(volume)
        # 加载并播放音频
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print(f"音频播放完成（音量：{volume*100}%）")
    except Exception as e:
        print(f"音频播放失败：{e}")
    finally:
        # 释放资源
        pygame.mixer.quit()

async def amain() -> None:
    """主函数：执行转语音 + 播放音频"""
    # 1. 文字转语音
    await text_to_speech(TEXT, VOICE, OUTPUT_FILE)
    # 2. 播放生成的音频
    play_audio(OUTPUT_FILE, PLAY_VOLUME)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(amain())
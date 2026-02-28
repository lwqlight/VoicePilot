import os
import subprocess

# 关键修正：替换为真正的麦克风设备名（Source #78 的 Name）
mic_name = "alsa_input.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"

def get_mute_status(source_name):
    """获取设备静音状态（更优雅的方式，避免直接调用os.system）"""
    try:
        # 执行命令并捕获输出，而非直接打印
        result = subprocess.check_output(
            ['pactl', 'get-source-mute', source_name],
            text=True,
            stderr=subprocess.STDOUT
        )
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"获取状态失败：{e.output}"

def set_mute_status(source_name, mute):
    """设置设备静音/取消静音"""
    # mute: 1=静音，0=取消静音
    try:
        os.system(f'pactl set-source-mute {source_name} {mute}')
        return f"已{'静音' if mute else '取消静音'}设备：{source_name}"
    except Exception as e:
        return f"操作失败：{e}"

# 1. 获取当前麦克风静音状态
current_status = get_mute_status(mic_name)
print("当前麦克风静音状态：", current_status)  # 正确输出应为 Mute: no / Mute: yes

# 2. 测试静音（取消注释执行）
# print(set_mute_status(mic_name, 1))
# print("静音后状态：", get_mute_status(mic_name))  # 应输出 Mute: yes

# 3. 测试取消静音
print(set_mute_status(mic_name, 0))
print("取消静音后状态：", get_mute_status(mic_name))  # 应输出 Mute: no
# 贡献指南 (Contributing Guide)

感谢您对本地语音助手项目的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议：

1. 检查 [Issues](../../issues) 是否已有类似问题
2. 如果没有，创建新的 Issue，并提供：
   - 清晰的标题
   - 详细的问题描述
   - 复现步骤
   - 系统环境信息 (OS, Python版本等)
   - 相关日志或截图

### 提交代码

1. **Fork 本仓库**
   ```bash
   git clone https://github.com/your-username/local_LLM_test.git
   cd local_LLM_test
   ```

2. **创建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **进行修改**
   - 遵循现有代码风格
   - 添加必要的注释
   - 更新相关文档

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加xxx功能"
   ```

5. **推送到您的 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建 Pull Request**
   - 描述您的更改
   - 关联相关 Issue
   - 等待代码审查

## 📝 提交信息规范

使用语义化提交信息：

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 添加离线TTS支持` |
| `fix` | 修复bug | `fix: 修复麦克风静音问题` |
| `docs` | 文档更新 | `docs: 更新README安装说明` |
| `style` | 代码格式 | `style: 统一代码缩进` |
| `refactor` | 重构 | `refactor: 优化音频处理模块` |
| `test` | 测试 | `test: 添加ASR单元测试` |
| `chore` | 构建/工具 | `chore: 更新依赖版本` |

## 🎨 代码风格

- 遵循 PEP 8 Python 代码规范
- 使用有意义的变量和函数名
- 添加类型提示 (Type Hints)
- 编写清晰的注释和文档字符串

```python
def process_audio(input_file: str, output_dir: str) -> None:
    """
    处理音频文件并保存到输出目录

    Args:
        input_file: 输入音频文件路径
        output_dir: 输出目录路径
    """
    pass
```

## 🧪 测试

提交代码前请确保：
- 代码能够正常运行
- 没有明显的 bug
- 新功能有相应的测试

## 📄 许可证

提交代码即表示您同意您的贡献将根据 [MIT License](LICENSE) 进行许可。

## 💬 联系方式

如有任何问题，请通过以下方式联系：
- 提交 [Issue](../../issues)
- 发起 [Discussion](../../discussions)

---

再次感谢您的贡献！🎉

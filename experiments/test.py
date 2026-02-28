from funasr import AutoModel

model_dir = "./SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./SenseVoiceSmall/model.py",
    device="cpu",
)

res = model.generate(
    input="./example/zh.mp3",
    language="auto",
    use_itn=True,
    batch_size_s=60,
)

print(res[0]["text"])
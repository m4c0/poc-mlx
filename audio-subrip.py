from mlx_audio.stt.utils import load

def time_fmt(f):
    n = int(f * 1000.0)
    ms = n % 1000
    s = (n // 1000) % 60
    m = (n // 60000) % 60
    h = (n // 3600000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def fmt(s, idx, f):
    st = time_fmt(s.start)
    et = time_fmt(s.end)
    print(idx,              file=f)
    print(f'{st} --> {et}', file=f)
    print(s.text,           file=f)
    print("",               file=f)

model = load("mlx-community/parakeet-tdt-0.6b-v3")

# for c in model.stream_generate("track-2.wav"): fmt(c)

idx = 1
res = model.generate("track-2.wav", chunk_duration=60)
with open('track-2.srt', mode="w") as f:
    for s in res.sentences:
        fmt(s, idx, f)
        idx = idx + 1

from mlx_audio.stt.utils import load

model = load("mlx-community/parakeet-tdt-0.6b-v3")

# for chunk in model.stream_generate("audio.wav"):
#     print(f'{chunk.start_time:.2f} {chunk.end_time:.2f} {chunk.text}')

res = model.generate("audio.wav")
for s in res.sentences: print(f'{s.start:.2f} {s.end:.2f} {s.text}')

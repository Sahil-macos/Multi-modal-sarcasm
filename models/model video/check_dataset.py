import av

video_path = "data/raw/video/final_utterance_videos/1_10004_u.mp4"

container = av.open(video_path)
stream = container.streams.video[0]

print("Duration:", float(container.duration / 1000000), "seconds")
print("Width:", stream.width)
print("Height:", stream.height)
print("FPS:", float(stream.average_rate))
print("Total frames:", stream.frames)

container.close()
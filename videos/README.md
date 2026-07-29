Dashboard-ready annotated traffic videos.

`flowsense_hybrid_video_1.mp4` through `flowsense_hybrid_video_4.mp4` are the
official H.264 outputs. They were rendered from clean source footage with
tracking boxes, canonical IDs, observed paths, and learned-hybrid prediction
extensions. Other annotated videos in this directory are retained as legacy
project artifacts.

`flowsense_web_hybrid_video_1.mp4` through
`flowsense_web_hybrid_video_4.mp4` are compressed 720p deployment copies of
those same official outputs. They preserve the full frame count and frame rate,
use H.264/yuv420p with fast-start metadata, and are committed as ordinary Git
files so the public Streamlit dashboard can play them without Git LFS access or
authentication.

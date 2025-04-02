# video-summarization

Build container
`docker build -t video-summarization .`

Run container
`docker run --gpus all --name video-summarization -p 7860:7860 video-summarization`

Exec container
`docker run -it video-summarization /bin/sh`

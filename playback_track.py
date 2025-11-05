import asyncio
from typing import Optional

from aiortc import MediaStreamTrack, RTCDataChannel
from aiortc.contrib.media import MediaPlayer

class PlaybackTrack(MediaStreamTrack):
    kind = "audio"
    track: MediaStreamTrack = None
    time: float = 0.0

    def __init__(self):
        super().__init__()

    def select(self, filename):
        self.track = MediaPlayer(filename, format="wav", loop=False).audio

    async def recv(self):
        if self.track is None:
            self.track = MediaPlayer("silence.wav", format="wav", loop=True).audio

        # We want to keep playback running forever so once we get an exception we fallback to silence
        try:
            async with asyncio.timeout(1):
                frame = await self.track.recv()
        except Exception as e:
            self.track = MediaPlayer("silence.wav", format="wav", loop=True).audio
            frame = await self.track.recv()

        if frame.pts < frame.sample_rate * self.time:
            frame.pts = frame.sample_rate * self.time
        self.time += 0.02

        return frame

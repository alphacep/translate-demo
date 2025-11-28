#!/usr/bin/env python3

import json
import ssl
import sys
import os
import concurrent.futures
import asyncio

from timeit import default_timer as timer
from pathlib import Path
from aiohttp import web
from aiohttp.web_exceptions import HTTPServiceUnavailable
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration
from av.audio.resampler import AudioResampler

from playback_track import PlaybackTrack

import sherpa_onnx
import numpy as np
from llama_cpp import Llama
import soundfile as sf

ROOT = Path(__file__).parent

vosk_interface = os.environ.get('VOSK_SERVER_INTERFACE', '0.0.0.0')
vosk_port = int(os.environ.get('VOSK_SERVER_PORT', 2700))
vosk_model_path = os.environ.get('VOSK_MODEL_PATH', 'model')
vosk_cert_file = os.environ.get('VOSK_CERT_FILE', None)
vosk_key_file = os.environ.get('VOSK_KEY_FILE', None)
vosk_dump_file = os.environ.get('VOSK_DUMP_FILE', None)
vosk_udp_port_range = os.environ.get('VOSK_UDP_PORT_RANGE', '35000:35100')

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder="asr-model/am-onnx/encoder.onnx",
            decoder="asr-model/am-onnx/decoder.onnx",
            joiner="asr-model/am-onnx/joiner.onnx",
            tokens="asr-model/lang/tokens.txt",
            num_threads=4,
            sample_rate=16000,
            dither=3e-5,
            decoding_method="modified_beam_search",
            max_active_paths=10,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.0,
            rule2_min_trailing_silence=0.5,
            rule3_min_utterance_length=300,  # it essentially disables this rule
            provider="cuda",
            model_type="zipformer2")


llama_model = Llama(
        model_path = "llama-model/gemma-3-4b-it-Q8_0.gguf",
        n_ctx = 4096,
        n_gpu_layers = 100,
        verbose = False,
    )
SYSTEM_PROMPT = "Ты переводчик с русского на английский. Переводи дословно всё, что я говорю."

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model = "kokoro-model/model.onnx",
            voices = "kokoro-model/voices.bin",
            tokens = "kokoro-model/tokens.txt",
            data_dir = "kokoro-model/espeak-ng-data"), num_threads=4, provider="cuda"))
tts_model = sherpa_onnx.OfflineTts(tts_config)

pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
dump_fd = None if vosk_dump_file is None else open(vosk_dump_file, "wb")

students = []


def process_chunk(stream, messages, message):
    samples_int16 = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(16000, samples_int16)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    is_endpoint = recognizer.is_endpoint(stream)
    result = recognizer.get_result(stream)
    print (result)
    if is_endpoint:
        recognizer.reset(stream)

        if result != "":

            start_llm = timer()

            messages.append({"role": "user", "content": result})
            response = llama_model.create_chat_completion(
                messages,
                temperature=0.95,
                top_k=64,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            response = response["choices"][0]["message"]["content"].replace("\"", "")
            print (f"Translatiton {response}", flush=True)
            end_llm = timer()
            messages.append({"role": "assistant", "content": response})

            audio = tts_model.generate(response, sid=0, speed=1.0)
            sf.write("test-en.wav", audio.samples, audio.sample_rate, subtype="PCM_16")
            end_tts = timer()

            print (f"LLM {end_llm - start_llm:.3f} TTS {end_tts - end_llm:.3f}", flush=True)

            return f"{{ \"text\": \"{result}\", \"translation\": \"{response}\" }}", True
        else:
            return f"{{ \"text\": \"{result}\"}}", False
    else:
        return f"{{ \"partial\": \"{result}\" }}", False


class VoskTask:
    def __init__(self, user_connection):
        self.__resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        self.__pc = user_connection
        self.__audio_task = None
        self.__track = None
        self.__channel = None
        self.__stream = recognizer.create_stream()
        self.__messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def set_audio_track(self, track):
        self.__track = track

    async def set_text_channel(self, channel):
        self.__channel = channel

    async def start(self):
        self.__audio_task = asyncio.create_task(self.__run_audio_xfer())

    async def stop(self):
        if self.__audio_task is not None:
            self.__audio_task.cancel()
            self.__audio_task = None

    async def __run_audio_xfer(self):
        loop = asyncio.get_running_loop()

        max_frames = 10
        frames = []
        while True:
            fr = await self.__track.recv()
            frames.append(fr)

            # We need to collect frames so we don't send partial results too often
            if len(frames) < max_frames:
               continue

            dataframes = bytearray(b'')
            for fr in frames:
                for rfr in self.__resampler.resample(fr):
                    dataframes += bytes(rfr.planes[0])[:rfr.samples * 2]
            frames.clear()

            if dump_fd != None:
                dump_fd.write(bytes(dataframes))

            result, final = await loop.run_in_executor(pool, process_chunk, self.__stream, self.__messages, bytes(dataframes))
            print(result, flush=True)

            if final:
                for student in students:
                    await student.send_result(result)

            self.__channel.send(result)


class StudentTask:
    def __init__(self, user_connection):
        self.__pc = user_connection
        self.__playback_track = None
        self.__track = None
        self.__channel = None

    async def set_text_channel(self, channel):
        self.__channel = channel

    async def set_playback_track(self, track):
        self.__playback_track = track

    async def send_result(self, result):
        self.__playback_track.select("test-en.wav")
        self.__channel.send(result)


async def index_lecture(request):
    content = open(str(ROOT / 'static' / 'lecture' / 'index.html')).read()
    return web.Response(content_type='text/html', text=content)

async def index_student(request):
    content = open(str(ROOT / 'static' / 'student' / 'index.html')).read()
    return web.Response(content_type='text/html', text=content)


async def offer_lecture(request):

    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection(RTCConfiguration(portRange=vosk_udp_port_range))

    vosk = VoskTask(pc)

    @pc.on('datachannel')
    async def on_datachannel(channel):
        channel.send('{}') # Dummy message to make the UI change to "Listening"
        await vosk.set_text_channel(channel)
        await vosk.start()

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == 'failed':
            await pc.close()

    @pc.on('track')
    async def on_track(track):
        if track.kind == 'audio':
            await vosk.set_audio_track(track)

        @track.on('ended')
        async def on_ended():
            await vosk.stop()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))




async def offer_student(request):

    params = await request.json()
    offer = RTCSessionDescription(
        sdp=params['sdp'],
        type=params['type'])

    pc = RTCPeerConnection(RTCConfiguration(portRange=vosk_udp_port_range))

    student = StudentTask(pc)
    playback_track = PlaybackTrack()
    pc.addTrack(playback_track)
    await student.set_playback_track(playback_track)

    @pc.on('datachannel')
    async def on_datachannel(channel):
        channel.send('{}') # Dummy message to make the UI change to "Listening"
        await student.set_text_channel(channel)

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == 'failed':
            await pc.close()

    @pc.on('track')
    async def on_track(track):
        if track.kind == 'audio':
            await student.set_audio_track(track)

        @track.on('ended')
        async def on_ended():
            students.remove(student)
            await student.stop()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    students.append(student)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }))


if __name__ == '__main__':

    if vosk_cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(vosk_cert_file, vosk_key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.router.add_post('/lecture/offer_lecture', offer_lecture)
    app.router.add_post('/student/offer_student', offer_student)

    app.router.add_get('/lecture/', index_lecture)
    app.router.add_get('/student/', index_student)
    app.router.add_static('/static/', path=ROOT / 'static', name='static')

    web.run_app(app, port=vosk_port, ssl_context=ssl_context)

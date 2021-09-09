import speech_recognition as sr
import pyaudio
import wave


def voice_recognition(user_sec):
	CHUNK = 2048
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 48000
	RECORD_SECONDS = user_sec
	WAVE_OUTPUT_FILENAME = "output.wav"

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
		        channels=CHANNELS,
		        rate=RATE,
		        input=True,
		        input_device_index = 12,
		        frames_per_buffer=CHUNK)

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data)

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	device = sr.Recognizer()
	audio_target = sr.AudioFile('./output.wav')
	with audio_target as source:
	    audio = device.listen(source)

	result_voice = device.recognize_google(audio, language='ko-KR')
	print(result_voice)

	return result_voice

if __name__ == "__main__":
	voice_recognition(2)
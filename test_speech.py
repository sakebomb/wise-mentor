import json
import tempfile
import unittest
from pathlib import Path

import httpx

from memory import MemoryStore, extract_facts
from speech import (
    AudioTooLarge,
    PendingClip,
    SpeechToText,
    VoiceNoteResult,
    format_heard_prefix,
    from_config,
    heard_field,
    TextToSpeech,
    audio_filename,
    is_audio_attachment,
    is_voice_note,
    merge_message_text,
    message_has_voice_note,
    resolve_voice_note,
    should_answer_with_model,
    should_respond,
    should_retry_voice,
    speakable_text,
    spoken_reply_audio,
    strip_heard_prefix,
    tts_from_config,
)


class AudioDetectionTests(unittest.TestCase):
    def test_ogg_content_type_is_a_voice_note(self):
        self.assertTrue(is_audio_attachment("audio/ogg", "voice-message.ogg"))

    def test_codec_parameter_still_counts_as_audio(self):
        self.assertTrue(is_audio_attachment("audio/ogg; codecs=opus", None))

    def test_voice_filename_counts_when_discord_omits_the_type(self):
        self.assertTrue(is_audio_attachment(None, "voice-message.ogg"))

    def test_image_and_text_are_not_voice_notes(self):
        self.assertFalse(is_audio_attachment("image/png", "pic.png"))
        self.assertFalse(is_audio_attachment("text/plain", "notes.txt"))
        self.assertFalse(is_audio_attachment(None, None))
        self.assertFalse(is_audio_attachment("", "no-extension"))


class TranscriptTextTests(unittest.TestCase):
    def test_merge_appends_transcript_after_typed_text(self):
        self.assertEqual(
            merge_message_text("what about this", "I like rockets"),
            "what about this\nI like rockets",
        )

    def test_merge_skips_blank_parts(self):
        self.assertEqual(merge_message_text("", "  ", "I play lacrosse"), "I play lacrosse")

    def test_transcript_feeds_the_same_fact_extractor_as_typed_text(self):
        merged = merge_message_text("", "my name is Sam and I like rockets")
        facts = extract_facts(f"<@1550704484030750741>: {merged}")
        joined = " ".join(facts).lower()
        self.assertIn("sam", joined)
        self.assertIn("rockets", joined)

    def test_heard_field_names_the_transcript_and_truncates(self):
        field = heard_field("I like rockets")
        self.assertEqual(field["name"], "Heard")
        self.assertEqual(field["value"], "I like rockets")
        self.assertFalse(field["inline"])
        long_field = heard_field("x" * 1100)
        self.assertEqual(len(long_field["value"]), 1024)
        self.assertTrue(long_field["value"].endswith("..."))

    def test_plain_reply_prefix_round_trips_off_the_answer(self):
        prefix = format_heard_prefix("I like rockets\nand chess")
        visible = prefix + "Rockets are worth the trouble."
        self.assertEqual(strip_heard_prefix(visible), "Rockets are worth the trouble.")
        self.assertEqual(strip_heard_prefix("No prefix here"), "No prefix here")


class TranscribeTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_model_and_file_and_returns_text(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = request.content
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"text": "  I like rockets  "})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(
            base_url="http://stt.example/v1",
            model="Systran/faster-distil-whisper-large-v3",
            api_key="secret-token",
            client=client,
        )
        text = await stt.transcribe(
            filename="voice-message.ogg",
            content_type="audio/ogg; codecs=opus",
            audio=b"hello-audio",
        )
        self.assertEqual(text, "I like rockets")
        self.assertEqual(seen["path"], "/v1/audio/transcriptions")
        self.assertIn(b'name="model"', seen["body"])
        self.assertIn(b"Systran/faster-distil-whisper-large-v3", seen["body"])
        self.assertIn(b"voice-message.ogg", seen["body"])
        self.assertIn(b"hello-audio", seen["body"])
        self.assertEqual(seen["auth"], "Bearer secret-token")
        await client.aclose()

    async def test_omits_auth_header_when_no_key_is_configured(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"text": "hi"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client)
        await stt.transcribe(filename="a.ogg", content_type="audio/ogg", audio=b"x")
        self.assertIsNone(seen["auth"])
        await client.aclose()

    async def test_rejects_oversize_audio_before_any_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("oversize audio was uploaded")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client, max_bytes=4)
        with self.assertRaises(AudioTooLarge):
            await stt.transcribe(filename="a.ogg", content_type="audio/ogg", audio=b"12345")
        await client.aclose()

    async def test_http_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "down"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client)
        with self.assertRaises(httpx.HTTPStatusError):
            await stt.transcribe(filename="a.ogg", content_type="audio/ogg", audio=b"x")
        await client.aclose()


class ConfigTests(unittest.TestCase):
    def test_disabled_when_base_url_is_missing(self):
        client = httpx.AsyncClient()
        self.assertIsNone(from_config({}, client))
        self.assertIsNone(from_config({"stt": {"model": "whisper-1"}}, client))
        self.assertIsNone(from_config({"stt": {"base_url": "http://stt.example"}}, client))

    def test_builds_when_url_and_model_are_set(self):
        client = httpx.AsyncClient()
        stt = from_config(
            {"stt": {"base_url": "http://stt.example/", "model": "whisper-1", "max_bytes": 12}},
            client,
        )
        self.assertIsInstance(stt, SpeechToText)
        self.assertEqual(stt.max_bytes, 12)


class ResolveVoiceNoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_transcript_skips_download_and_upload(self):
        async def fetch() -> bytes:
            raise AssertionError("saved transcript should not download audio")

        result = await resolve_voice_note(
            message_id=10,
            user_id=1,
            clips=[PendingClip("voice.ogg", "audio/ogg", 10, fetch)],
            stt=None,
            load_saved=lambda message_id: "I like rockets",
            store_text=lambda *args: (_ for _ in ()).throw(AssertionError("already saved")),
            max_bytes=100,
            max_clips=3,
        )
        self.assertEqual(result.transcript, "I like rockets")
        self.assertTrue(result.had_audio)

    async def test_unconfigured_stt_does_not_download(self):
        async def fetch() -> bytes:
            raise AssertionError("unconfigured stt should not download audio")

        result = await resolve_voice_note(
            message_id=10,
            user_id=1,
            clips=[PendingClip("voice.ogg", "audio/ogg", 10, fetch)],
            stt=None,
            load_saved=lambda message_id: None,
            store_text=lambda *args: None,
            max_bytes=100,
            max_clips=3,
        )
        self.assertTrue(result.unconfigured)
        self.assertEqual(result.transcript, "")
        self.assertIn("aren't set up", " ".join(result.warnings()))
        self.assertIn("Type it", result.fallback_reply())

    async def test_stores_text_only_and_drops_the_audio_bytes(self):
        audio = b"OggS-unique-audio-bytes-not-for-storage"
        stored = {}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"text": "I like rockets"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client)

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            result = await resolve_voice_note(
                message_id=42,
                user_id=7,
                clips=[PendingClip("voice.ogg", "audio/ogg", len(audio), lambda: _ready(audio))],
                stt=stt,
                load_saved=store.get_transcript,
                store_text=store.save_transcript,
                max_bytes=10_000,
                max_clips=3,
            )
            raw = b"".join(path.read_bytes() for path in Path(tmp).iterdir() if path.is_file())
            stored["text"] = store.get_transcript(42)

        self.assertEqual(result.transcript, "I like rockets")
        self.assertEqual(stored["text"], "I like rockets")
        self.assertNotIn(audio, raw)
        await client.aclose()

    async def test_too_long_clip_is_not_uploaded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("oversize clip was uploaded")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client, max_bytes=100)

        async def fetch() -> bytes:
            return b"y" * 500

        result = await resolve_voice_note(
            message_id=1,
            user_id=1,
            clips=[PendingClip("voice.ogg", "audio/ogg", 500, fetch)],
            stt=stt,
            load_saved=lambda message_id: None,
            store_text=lambda *args: (_ for _ in ()).throw(AssertionError("nothing to store")),
            max_bytes=100,
            max_clips=3,
        )
        self.assertEqual(result.too_large, 1)
        self.assertEqual(result.transcript, "")
        self.assertFalse(should_answer_with_model(text="", image_count=0, had_audio=result.had_audio))
        await client.aclose()

    async def test_failed_transcription_does_not_call_the_model_for_audio_only(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="nope")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client)
        result = await resolve_voice_note(
            message_id=1,
            user_id=1,
            clips=[PendingClip("voice.ogg", "audio/ogg", 4, lambda: _ready(b"data"))],
            stt=stt,
            load_saved=lambda message_id: None,
            store_text=lambda *args: None,
            max_bytes=100,
            max_clips=3,
        )
        self.assertEqual(result.failed, 1)
        self.assertFalse(should_answer_with_model(text="", image_count=0, had_audio=True))
        self.assertTrue(should_answer_with_model(text="what about this", image_count=0, had_audio=True))
        await client.aclose()

    async def test_extra_clips_past_the_cap_are_not_fetched(self):
        fetched = 0

        async def fetch() -> bytes:
            nonlocal fetched
            fetched += 1
            return b"clip"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"text": "word"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        stt = SpeechToText(base_url="http://stt.example", model="whisper-1", client=client)
        clips = [PendingClip(f"{i}.ogg", "audio/ogg", 4, fetch) for i in range(4)]
        result = await resolve_voice_note(
            message_id=1,
            user_id=1,
            clips=clips,
            stt=stt,
            load_saved=lambda message_id: None,
            store_text=lambda *args: None,
            max_bytes=100,
            max_clips=3,
        )
        self.assertEqual(fetched, 3)
        self.assertEqual(result.dropped_extra, 1)
        self.assertEqual(result.transcript, "word\nword\nword")
        self.assertTrue(any("first 3" in note for note in result.warnings()))
        await client.aclose()


class WakeTests(unittest.TestCase):
    def test_discord_voice_message_wakes_without_a_mention(self):
        self.assertTrue(is_voice_note("audio/ogg; codecs=opus", "voice-message.ogg"))
        self.assertTrue(is_voice_note(None, "voice-message.ogg"))
        self.assertFalse(is_voice_note("audio/mpeg", "song.mp3"))
        note = _Att("audio/ogg", "voice-message.ogg")
        song = _Att("audio/mpeg", "song.mp3")
        self.assertTrue(message_has_voice_note([note]))
        self.assertFalse(message_has_voice_note([song]))
        self.assertTrue(message_has_voice_note([], voice_flag=True))

    def test_server_voice_note_is_addressed_to_the_bot(self):
        self.assertTrue(should_respond(is_dm=False, mentioned=False, is_bot=False, has_voice_note=True))
        self.assertTrue(should_respond(is_dm=False, mentioned=True, is_bot=False, has_voice_note=False))
        self.assertFalse(should_respond(is_dm=False, mentioned=False, is_bot=False, has_voice_note=False))

    def test_dms_still_answer_and_bots_never_do(self):
        self.assertTrue(should_respond(is_dm=True, mentioned=False, is_bot=False, has_voice_note=False))
        self.assertFalse(should_respond(is_dm=True, mentioned=True, is_bot=True, has_voice_note=True))


class SpeakTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_voice_and_strips_markup(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = request.read()
            seen["accept"] = request.headers.get("accept")
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, content=b"RIFF-fake-wav")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        speaker = TextToSpeech(
            base_url="http://tts.example/v1",
            model="tts-1",
            voice="af_heart",
            api_key="secret-token",
            client=client,
            max_chars=80,
        )
        audio = await speaker.speak("**Rockets** are worth the trouble. " + ("word " * 40))
        self.assertEqual(audio, b"RIFF-fake-wav")
        self.assertEqual(seen["path"], "/v1/audio/speech")
        self.assertEqual(seen["accept"], "audio/mpeg")
        self.assertEqual(seen["auth"], "Bearer secret-token")
        self.assertIn(b'"voice":"af_heart"', seen["body"])
        self.assertIn(b'"response_format":"mp3"', seen["body"])
        self.assertEqual(audio_filename(b"ID3\x04rest"), "wise-mentor.mp3")
        self.assertEqual(audio_filename(b"RIFF...."), "wise-mentor.wav")
        self.assertEqual(audio_filename(b"OggSrest"), "wise-mentor.ogg")
        self.assertIn(b"Rockets are worth the trouble.", seen["body"])
        self.assertNotIn(b"**", seen["body"])
        self.assertLessEqual(len(json.loads(seen["body"])["input"]), 80)
        await client.aclose()

    async def test_typed_chat_and_failures_stay_silent(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("typed chat was spoken")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        speaker = TextToSpeech(base_url="http://tts.example", model="tts-1", voice="af_heart", client=client)
        silent = await spoken_reply_audio(answer="Hello", transcript="", speaker=speaker)
        self.assertEqual(silent, b"")

        def fail(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="nope")

        failing = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        broken = TextToSpeech(base_url="http://tts.example", model="tts-1", voice="af_heart", client=failing)
        missed = await spoken_reply_audio(answer="Hello", transcript="I like rockets", speaker=broken)
        self.assertEqual(missed, b"")
        broken.max_audio_bytes = 3

        def huge(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"RIFF-too-big")

        big = TextToSpeech(
            base_url="http://tts.example",
            model="tts-1",
            voice="af_heart",
            client=httpx.AsyncClient(transport=httpx.MockTransport(huge)),
            max_audio_bytes=4,
        )
        self.assertEqual(await spoken_reply_audio(answer="Hello", transcript="hey", speaker=big), b"")
        await client.aclose()
        await failing.aclose()

    def test_tts_config_requires_url_model_and_voice(self):
        client = httpx.AsyncClient()
        self.assertIsNone(tts_from_config({}, client))
        self.assertIsNone(tts_from_config({"tts": {"base_url": "http://tts.example", "model": "tts-1"}}, client))
        speaker = tts_from_config(
            {"tts": {"base_url": "http://tts.example/", "model": "tts-1", "voice": "af_heart", "max_chars": 40}},
            client,
        )
        self.assertIsInstance(speaker, TextToSpeech)
        self.assertEqual(speaker.max_chars, 40)
        self.assertEqual(speakable_text("**Hi** there", 80), "Hi there")


class AnswerGateTests(unittest.TestCase):
    def test_typed_text_and_images_still_answer_without_audio(self):
        self.assertTrue(should_answer_with_model(text="hello", image_count=0, had_audio=False))
        self.assertTrue(should_answer_with_model(text="", image_count=1, had_audio=False))
        self.assertTrue(should_answer_with_model(text="", image_count=0, had_audio=False))

    def test_audio_only_silence_does_not_answer(self):
        voice = VoiceNoteResult(clip_count=1, empty=1)
        self.assertTrue(voice.had_audio)
        self.assertFalse(should_answer_with_model(text="", image_count=0, had_audio=voice.had_audio))


class RetryTests(unittest.TestCase):
    def test_retries_audio_only_when_speech_service_failed(self):
        self.assertTrue(
            should_retry_voice(
                had_audio=True,
                transcript="",
                retryable=True,
                has_other_text=False,
                has_images=False,
            )
        )

    def test_keeps_typed_text_and_successful_transcripts(self):
        self.assertFalse(
            should_retry_voice(
                had_audio=True,
                transcript="",
                retryable=True,
                has_other_text=True,
                has_images=False,
            )
        )
        self.assertFalse(
            should_retry_voice(
                had_audio=True,
                transcript="I like rockets",
                retryable=True,
                has_other_text=False,
                has_images=False,
            )
        )
        self.assertFalse(
            should_retry_voice(
                had_audio=True,
                transcript="",
                retryable=False,
                has_other_text=False,
                has_images=False,
            )
        )


class _Att:
    def __init__(self, content_type, filename):
        self.content_type = content_type
        self.filename = filename


def _ready(payload: bytes):
    async def fetch() -> bytes:
        return payload

    return fetch()


if __name__ == "__main__":
    unittest.main()

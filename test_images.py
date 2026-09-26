import base64
import unittest

import httpx

from images import (
    ImageGenerator,
    image_filename,
    image_prompt_from_text,
    images_from_config,
    refuses_image,
)


class ImageAskTests(unittest.TestCase):
    def test_draw_requests_become_a_prompt(self):
        cases = {
            "draw me a rocket": "a rocket",
            "<@1550704484030750741>: draw me a rocket": "a rocket",
            "can you draw a dragon?": "a dragon",
            "hey, sketch the moon": "the moon",
            "please draw me a picture of a castle": "a castle",
            "picture of a saturn v": "a saturn v",
            "show me a picture of a lacrosse goal": "a lacrosse goal",
            "make me an image of a red bike": "a red bike",
            "show me what a neutron star looks like": "a neutron star",
            "draw me as an astronaut": "me as an astronaut",
        }
        for text, prompt in cases.items():
            with self.subTest(text=text):
                self.assertEqual(image_prompt_from_text(text), prompt)

    def test_ordinary_chat_does_not_ask_for_a_picture(self):
        for text in ("what is 2+2", "I like to draw dragons", "how to draw a face", "show me a dog", "", "draw me"):
            with self.subTest(text=text):
                self.assertIsNone(image_prompt_from_text(text))

    def test_sexual_and_nude_asks_are_refused(self):
        self.assertTrue(refuses_image("a nude person"))
        self.assertTrue(refuses_image("naked teen"))
        self.assertTrue(refuses_image("a porn scene"))
        self.assertTrue(refuses_image("a sexy portrait"))
        self.assertTrue(refuses_image("a hot girl at the beach"))
        self.assertFalse(refuses_image("a rocket"))
        self.assertFalse(refuses_image("a peacock"))
        self.assertFalse(refuses_image("a cocktail"))
        self.assertFalse(refuses_image("a naked mole rat in a tunnel"))

    def test_filename_follows_the_image_bytes(self):
        self.assertEqual(image_filename(b"\x89PNG\r\n\x1a\nfake"), "wise-mentor.png")
        self.assertEqual(image_filename(b"\xff\xd8\xfffake"), "wise-mentor.jpg")


class ImageGenerateTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_the_prompt_and_returns_png_bytes(self):
        png = b"\x89PNG\r\n\x1a\nfake"
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = request.content.decode()
            encoded = base64.b64encode(png).decode()
            return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        painter = ImageGenerator(
            base_url="https://api.x.ai/v1",
            model="grok-imagine-image-2.0",
            api_key="secret-token",
            client=client,
        )
        image = await painter.generate("a rocket")
        self.assertEqual(image, png)
        self.assertEqual(seen["path"], "/v1/images/generations")
        self.assertEqual(seen["auth"], "Bearer secret-token")
        self.assertIn("grok-imagine-image-2.0", seen["body"])
        self.assertIn("a rocket", seen["body"])
        self.assertIn("b64_json", seen["body"])

    async def test_fetches_a_url_when_the_service_returns_one(self):
        png = b"\x89PNG\r\n\x1a\nfrom-url"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/generations"):
                return httpx.Response(200, json={"data": [{"url": "https://cdn.example/rocket.png"}]})
            return httpx.Response(200, content=png)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        painter = ImageGenerator(base_url="http://images.example/v1", model="grok-imagine-image-2.0", api_key="k", client=client)
        self.assertEqual(await painter.generate("a rocket"), png)

    async def test_refusal_and_errors_return_no_bytes(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("refused prompts are not posted")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        painter = ImageGenerator(base_url="http://images.example/v1", model="m", api_key="k", client=client)
        self.assertEqual(await painter.generate("a nude portrait"), b"")

        def broken(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        broken_client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
        broken_painter = ImageGenerator(base_url="http://images.example/v1", model="m", api_key="k", client=broken_client)
        self.assertEqual(await broken_painter.generate("a rocket"), b"")

    def test_config_stays_off_without_a_key(self):
        client = httpx.AsyncClient()
        self.assertIsNone(images_from_config({"images": {"base_url": "https://api.x.ai/v1", "model": "grok-imagine-image-2.0"}}, client))
        painter = images_from_config(
            {"images": {"base_url": "https://api.x.ai/v1", "model": "grok-imagine-image-2.0", "api_key": "k"}},
            client,
        )
        self.assertIsNotNone(painter)

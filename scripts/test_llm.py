import os
import sys
from io import BytesIO
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense.llm import generate, generate_vision


def main():
    print("Testing generate()...")
    reply1 = generate("Reply with the single word: pong")
    print(f"Reply 1: {reply1}")

    # Build 200x200 Pillow image: solid red square on white
    img = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill="red")

    buf = BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    print("Testing generate_vision()...")
    reply2 = generate_vision("What color is the square? Answer in one word.", jpeg_bytes)
    print(f"Reply 2: {reply2}")

    pass_pong = "pong" in reply1.lower()
    pass_red = "red" in reply2.lower()

    status1 = "PASS" if pass_pong else "FAIL"
    status2 = "PASS" if pass_red else "FAIL"

    print(f"Text test: {status1} ('pong' present)")
    print(f"Vision test: {status2} ('red' present)")

    if pass_pong and pass_red:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

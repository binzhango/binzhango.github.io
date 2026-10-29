"""Derive registered contour ink and typeset the fixed card furniture."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

root = Path(__file__).resolve().parent
assets = root / 'assets'
copy = json.loads((root / 'card-config.json').read_text())
subject = Image.open(assets / 'subject.png').convert('RGBA')
W, H = subject.size
# Derive contours from the actual cutout; never regenerate a different pose.
white = Image.new('RGBA', subject.size, 'white')
white.alpha_composite(subject)
gray = white.convert('L').filter(ImageFilter.GaussianBlur(2.2))
edges = gray.filter(ImageFilter.FIND_EDGES)
lineart = ImageOps.invert(edges.point(lambda v: 255 if v > 28 else 0))
# FIND_EDGES also marks the outer canvas edge; keep that edge white.
ImageDraw.Draw(lineart).rectangle((0, 0, W - 1, H - 1), outline='white', width=5)
lineart.convert('RGB').save(assets / 'lineart.png')

background = Image.open(assets / 'background.png').convert('RGB')
assert background.size == subject.size == (1024, 1536)
background.save(assets / 'background.png')
text = Image.new('RGBA', subject.size)
d = ImageDraw.Draw(text)
ink = '#202127'
muted = '#655543'
font_dir = Path('/System/Library/Fonts/Supplemental')
def label(value, pos, size, bold=False, fill=ink, anchor='la', serif=False):
    font = ImageFont.truetype(str(font_dir / ('Georgia.ttf' if serif else ('Arial Bold.ttf' if bold else 'Arial.ttf'))), size)
    d.text(pos, value, font=font, fill=fill, anchor=anchor)

d.rounded_rectangle((38, 38, 986, 1498), radius=26, outline=ink, width=3)
label(copy['collection'], (74, 72), 22, True)
label('BZ', (950, 72), 22, anchor='ra')
label(copy['title'], (72, 111), 92, serif=True)
label(copy['subtitle'], (76, 220), 29, fill=muted)
d.line((74, 270, 950, 270), fill=ink, width=2)
d.line((74, 1285, 950, 1285), fill=ink, width=2)
label(copy['technique'], (512, 1315), 60, anchor='ma', serif=True)
label(copy['tagline'], (512, 1399), 25, fill=muted, anchor='ma')
label(copy['edition'], (74, 1461), 22)
label('TECHNICAL WRITING', (950, 1461), 18, anchor='ra')
text.save(assets / 'text.png')
# Contact sheet for visual validation only; not used as the card model.
preview = background.convert('RGBA')
preview.alpha_composite(subject)
preview.alpha_composite(text)
preview.save(root / 'layer-preview.png')
print('Prepared lineart and transparent typography:', subject.size)

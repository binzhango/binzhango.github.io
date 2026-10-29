"""Open the generated Blender scene, make it portable, and render three angles."""
import bpy
import json
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
bpy.ops.wm.open_mainfile(filepath=str(root / 'card.blend'))
# Refresh the packed typography from its editable source before saving/rendering.
for image in bpy.data.images:
    if image.source == 'FILE' and Path(image.filepath).name == 'text.png':
        if image.packed_file:
            image.unpack(method='REMOVE')
        image.filepath = str(root / 'assets' / 'text.png')
        image.reload()
        image.pack()
scene = bpy.context.scene
scene['Artwork provenance'] = 'AI-assisted interpretation of Bin Zhang’s original black-cat favicon; source in assets/favicon.png.'
scene['素材来源'] = 'Blog favicon identity reference; generated transparent cat and background; derived contours and typeset text.'
group = bpy.data.node_groups.get('视差 · 通用 UV / Parallax')
if group:
    group.name = '视差效果'
for image in bpy.data.images:
    if image.source == 'FILE':
        image.filepath = '//assets/' + Path(image.filepath).name
        if not image.packed_file:
            image.pack()
scene.render.resolution_x = 900
scene.render.resolution_y = 1250
scene.cycles.samples = 24
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    metal = False
    for device in prefs.devices:
        device.use = device.type == 'METAL'
        metal = metal or device.use
    scene.cycles.device = 'GPU' if metal else 'CPU'
except Exception:
    scene.cycles.device = 'CPU'
bpy.context.preferences.view.language = 'en_US'
bpy.context.preferences.view.use_translate_interface = False
scene.frame_set(25)
bpy.ops.wm.save_as_mainfile(filepath=str(root / 'card.blend'))
for name, frame in [('hero', 25), ('tilt-left', 1), ('tilt-right', 49)]:
    scene.frame_set(frame)
    scene.render.filepath = str(root / 'renders' / (name + '.png'))
    bpy.ops.render.render(write_still=True)
    print('RENDER_COMPLETE', name, flush=True)
scene.frame_set(25)
scene.render.filepath = str(root / 'renders' / 'hero.png')
bpy.ops.wm.save_as_mainfile(filepath=str(root / 'card.blend'))
(root / 'renders' / 'report.json').write_text(json.dumps({
    'engine': scene.render.engine, 'device': scene.cycles.device,
    'views': ['hero.png', 'tilt-left.png', 'tilt-right.png'],
    'resolution': [scene.render.resolution_x, scene.render.resolution_y],
    'packedAssets': all(i.packed_file is not None for i in bpy.data.images if i.source == 'FILE')
}, indent=2))

"""Apply the favicon card's English UI to the skill's copied viewer."""
from pathlib import Path
root = Path(__file__).resolve().parent
replacements = {
    'zh-CN': 'en', '白相首页': 'Card studio home', '白相 · White Atelier': 'Bin Zhang · Engineering Notes',
    'WHITE ATELIER': 'ENGINEERING NOTES', '白相': 'BZ', '个人典藏': 'Engineering notes',
    '作品信息': 'About this card', '保存卡片图片': 'Save card image', '卡片展台': 'Card gallery',
    '可旋转的全息卡片': 'Interactive holographic card; drag to rotate, use arrow keys to tilt',
    '正在装裱作品': 'Preparing your card', '景深与构图': 'Depth & composition',
    '画面比例': 'Artwork scale', '画面景深': 'Cat depth', '特效景深': 'Effects depth',
    '底纹景深': 'Background depth', '作品与材质': 'Artwork & finish', '卡片朝向': 'Card side',
    '正面': 'Front', '背面': 'Back', '显示或隐藏景深调整': 'Show or hide depth controls',
    '景深调整': 'Adjust depth', '卡面质感': 'Finish', '卡面材质': 'Card finish',
    '珠光': 'Pearl', '银箔': 'Silver', '烫金': 'Gold', '原画': 'Original', '光泽': 'Foil',
    '光影之间，自有回响。': 'Drag to tilt · Scroll to zoom · Arrow keys to explore',
    'PERSONAL ART COLLECTION': 'BIN ZHANG / ENGINEERING NOTES',
    '关闭作品信息': 'Close card information', '用户提供的参考作品': 'Original blog favicon',
    '个人艺术卡片习作': 'Layered holographic card', '来源': 'Source', '呈现': 'Format', '编号': 'Edition',
    '共生 / BOND': 'AI systems & data engineering / Bin Zhang', '光与影，彼此相依。': 'Notes on language models, retrieval systems, and data infrastructure. From underlying concepts to working code.',
    '共生': 'AI systems & data engineering', 'BOND': 'Bin Zhang', 'ART STUDY': 'ENGINEERING / NOTES',
    '暂停旋转': 'Pause rotation', '自动旋转': 'Auto rotate',
    '图形显示已暂停，请刷新页面恢复': 'Graphics paused. Reload the page to resume.',
    '卡片图片已保存': 'Card image saved', '图片未能保存，请重试': 'Could not save the image. Try again.',
    '重新加载': 'Reload', '作品暂时无法加载。': 'The card could not load.',
    '作品配置未找到': 'Card configuration not found', '当前设备无法显示卡面材质': 'Card shader unavailable on this device',
    '卡片模型缺少正面材质': 'The card model is missing its front material',
    '卡片加载超时，请检查网络或刷新重试': 'Card loading timed out. Check the connection and reload.',
    '浏览器未开启 WebGL：已用轻量 3D 模式显示（层次保留）': 'WebGL unavailable. Showing a simplified 3D preview.',
}
for filename in ['index.html', 'app.js']:
    path = root / 'web' / filename
    contents = path.read_text()
    for old, new in sorted(replacements.items(), key=lambda entry: len(entry[0]), reverse=True):
        contents = contents.replace(old, new)
    if filename == 'app.js':
        contents = contents.replace('max(abs(uView.z), .4) * depth * .10', 'max(abs(uView.z), .35) * depth * .14')
        contents = contents.replace('$("front").onclick = () => flip(false);', '$("auto").onclick = () => setAuto(!auto);\n  $("reset").onclick = () => reset();\n  $("front").onclick = () => flip(false);')
    if filename == 'index.html':
        contents = contents.replace('href="data:,"', 'href="./assets/favicon.png"')
        contents = contents.replace('<div class="header-actions">', '<div class="header-actions">\n<button class="icon-button" id="auto" aria-label="Auto rotate" title="Auto rotate" aria-pressed="false"><i data-lucide="play"></i></button>\n<button class="icon-button" id="reset" aria-label="Reset view" title="Reset view"><i data-lucide="rotate-ccw"></i></button>')
    path.write_text(contents)
style = root / 'web' / 'style.css'
css = style.read_text().replace('.title-line {\n    gap: 13px;', '.title-line {\n    gap: 13px;\n    flex-wrap: wrap;')
css = css.replace('.title-line h1 {\n    font-size: 34px;', '.title-line h1 {\n    font-size: 34px;\n    white-space: nowrap;')
style.write_text(css)
print('Customized the viewer; rebuild app.bundle.js next.')

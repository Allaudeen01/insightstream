import matplotlib
import os
font_path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
print(f'Font path: {font_path}')
print(f'Font exists: {os.path.isfile(font_path)}')

try:
    from fontTools.ttLib import TTFont as FTFont
    ft = FTFont(font_path)
    cmap = ft.getBestCmap()
    rupee_codepoint = 0x20B9  # ₹
    has_rupee = rupee_codepoint in cmap
    print(f'DejaVuSans has rupee glyph (U+20B9): {has_rupee}')
    if has_rupee:
        print(f'Glyph name: {cmap[rupee_codepoint]}')
    else:
        print('GLYPH MISSING — this font version does not contain U+20B9')
        # Show nearby codepoints to understand font coverage
        nearby = {k: v for k, v in cmap.items() if 0x20A0 <= k <= 0x20CF}
        print(f'Currency block coverage (U+20A0-20CF): {nearby}')
except ImportError:
    print('fontTools not installed')
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'fonttools', '-q'])
    print('fonttools installed — re-run this script')

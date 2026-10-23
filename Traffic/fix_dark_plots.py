"""
Fix all remaining dark backgrounds in plot scripts.
Targets: train_sequential_200ep.py, plot_performance_updated.py,
         train_tabular_concentrated.py, train_gnn_peaks_long.py,
         train_gnn_8mi_long.py, plot_satellite_map.py
"""
import os
import re

TRAFFIC = '/Users/andrewsmac/CCIR/Traffic'

# Dark hex colours to replace with white
DARK_COLORS = [
    '#0d1117', '#0D1117', '#1e1e1e', '#1E1E1E',
    '#161B22', '#161b22', '#0D1117', '#111111',
    '#2d2d2d', '#2D2D2D',
]

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    original = content

    # 1. Replace dark hex facecolors in set_facecolor / patch.set_facecolor / savefig facecolor=
    for dark in DARK_COLORS:
        content = content.replace(f"'{dark}'", "'white'")
        content = content.replace(f'"{dark}"', '"white"')

    # 2. Fix tick_params(colors='white') → 'black'  (text on axes)
    content = re.sub(r"tick_params\(colors=['\"]white['\"]", "tick_params(colors='black'", content)
    # Also handle labelcolor='white' inside tick_params
    content = re.sub(r"tick_params\(axis=['\"]([^'\"]+)['\"],\s*labelcolor=['\"]white['\"]",
                     r"tick_params(axis='\1', labelcolor='black'", content)

    # 3. Fix spine edgecolor dark values
    content = re.sub(r"spine\.set_edgecolor\(['\"]#333['\"]",
                     "spine.set_edgecolor('lightgray'", content)
    content = re.sub(r"spine\.set_edgecolor\(['\"]#[0-9a-fA-F]{3,6}['\"]",
                     "spine.set_edgecolor('lightgray'", content)

    # 4. Val Loss plotted with color='white' (shows on dark bg) → 'black'
    content = re.sub(r"(label=['\"]Val Loss['\"].*?)color=['\"]white['\"]",
                     r"\1color='black'", content)
    content = re.sub(r"color=['\"]white['\"](\s*,\s*linewidth.*?label=['\"]Val Loss['\"])",
                     r"color='black'\1", content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed: {os.path.basename(filepath)}")
    else:
        print(f"  — No changes: {os.path.basename(filepath)}")

FILES = [
    'train_sequential_200ep.py',
    'plot_performance_updated.py',
    'train_tabular_concentrated.py',
    'train_gnn_peaks_long.py',
    'train_gnn_8mi_long.py',
    'plot_satellite_map.py',
]

print("Fixing dark backgrounds...")
for fname in FILES:
    fix_file(os.path.join(TRAFFIC, fname))
print("Done.")

import os
import re

directory = '/Users/andrewsmac/CCIR/Traffic'

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = content
    # Find ", facecolor='white')" and remove it if it was erroneously added inside os.path.join
    # wait, it was added right before the first closing parenthesis of os.path.join.
    # So it looks like: os.path.join(OUT, 'st_gnn_performance_curve.png', facecolor='white'), dpi=200...
    
    # Let's fix this specific mistake:
    new_content = new_content.replace(", facecolor='white')", ")")
    # And then we correctly add facecolor='white' at the end of the plt.savefig line
    # Since each plt.savefig is usually on one line, we can just do line by line.
    
    lines = new_content.split('\n')
    for i, line in enumerate(lines):
        if 'plt.savefig(' in line:
            # If it doesn't already have facecolor='white' at the very end
            if "facecolor='white'" not in line and "facecolor=" not in line:
                # find the LAST closing parenthesis
                last_paren = line.rfind(')')
                if last_paren != -1:
                    line = line[:last_paren] + ", facecolor='white')" + line[last_paren+1:]
                    lines[i] = line
                    
    new_content = '\n'.join(lines)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.py') and file != 'fix_savefig_mess.py':
            process_file(os.path.join(root, file))

print("Done fixing savefig mess.")

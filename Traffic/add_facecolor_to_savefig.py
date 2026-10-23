import os
import re

directory = '/Users/andrewsmac/CCIR/Traffic'

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = content
    
    # Add facecolor='white' to savefig if not present
    def replace_savefig(match):
        call_content = match.group(0)
        if 'facecolor' not in call_content:
            # Check if it ends with )
            if call_content.endswith(')'):
                return call_content[:-1] + ")"
        return call_content

    new_content = re.sub(r'plt\.savefig\([^)]+\)', replace_savefig, new_content)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Added facecolor='white' in {filepath}")

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.py'):
            process_file(os.path.join(root, file))

print("Done adding facecolor='white' to savefig calls.")

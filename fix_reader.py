"""
Script to fix the corrupted reader.html file
"""
import re

# Read the corrupted file
with open('templates/reader.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where the corruption starts (where CSS ends and HTML begins incorrectly)
# The corruption appears to be around line 182 where HTML starts inside CSS

# Let's extract the good parts and rebuild
# The file should have: DOCTYPE, html, head with style, body with content, script

print("File has", len(content), "characters")
print("Checking for corruption...")

# Check if there's a proper closing style tag
if '</style>' in content:
    style_end = content.find('</style>')
    print(f"Found </style> at position {style_end}")
    
    # Check what comes after
    after_style = content[style_end:style_end+200]
    print("After </style>:")
    print(after_style[:100])
else:
    print("No </style> tag found - file is corrupted")

# Let's find the structure
doctype_pos = content.find('<!DOCTYPE')
html_start = content.find('<html')
head_start = content.find('<head>')
style_start = content.find('<style>')
style_end = content.find('</style>')

print(f"\nStructure positions:")
print(f"DOCTYPE: {doctype_pos}")
print(f"<html>: {html_start}")
print(f"<head>: {head_start}")
print(f"<style>: {style_start}")
print(f"</style>: {style_end}")

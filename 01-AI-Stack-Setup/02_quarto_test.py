# TEST QUARTO
# ***

# quarto install tinytex

text = """
---
title: "Quarto Basics"
format:
  pdf:
    toc: true
jupyter: python3
---

For a demonstration of a line plot on a polar axis, see @fig-polar.

```{python}
#| label: fig-polar
#| fig-cap: "A line plot on a polar axis"

import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r
fig, ax = plt.subplots(
  subplot_kw = {'projection': 'polar'} 
)
ax.plot(theta, r)
ax.set_rticks([0.5, 1, 1.5, 2])
ax.grid(True)
plt.show()
```
"""

import tempfile
import os
import quarto

with tempfile.NamedTemporaryFile(delete=False, suffix=".qmd", mode='w') as md_file:
    md_file.write(text)
    md_file_path = md_file.name

quarto.render(  
    input = md_file_path,
    output_format = "pdf",
    output_file = "my_pdf.pdf",
)

os.remove(md_file_path)

# Move file to downloads folder:
import shutil
import os

def move_file_to_downloads(filename):
    # Try to get the Downloads directory from environment variables
    try:
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    except KeyError:
        print("Could not find the Downloads directory.")
        return

    # Define the source path of your file
    source_path = filename

    # Define the destination path to the Downloads directory
    destination_path = os.path.join(downloads_path, os.path.basename(filename))

    # Move the file
    try:
        shutil.move(source_path, destination_path)
        print(f"File moved to: {destination_path}")
    except Exception as e:
        print(f"Error moving file: {e}")

# Path to the file you want to move
file_to_move = "my_pdf.pdf"

# Call the function to move the file
move_file_to_downloads(file_to_move)

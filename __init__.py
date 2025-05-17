# ComfyUI_ASCIINodes/__init__.py

# Import the node classes from their respective .py files
# These files (ascii_art_node.py, ascii_animation_node.py, and color_ascii_animation_node.py)
# should be in the same directory as this __init__.py file.

from .ascii_art_node import ASCIIArtGeneratorNode 
from .ascii_animation_node import ASCIIAnimationGeneratorNode
from .color_ascii_animation_node import ColorASCIIAnimationGeneratorNode # New import

# A dictionary that ComfyUI uses to map node_class names to node_display_names
NODE_CLASS_MAPPINGS = {
    "ASCIIArtGenerator": ASCIIArtGeneratorNode,
    "ASCIIAnimationGenerator": ASCIIAnimationGeneratorNode,
    "ColorASCIIAnimationGenerator": ColorASCIIAnimationGeneratorNode # New mapping
}

# A dictionary that ComfyUI uses to map node_class names to their display names
# This is what will appear in the ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCIIArtGenerator": "ASCII Art Generator (Static)",
    "ASCIIAnimationGenerator": "ASCII Typing Animation Generator",
    "ColorASCIIAnimationGenerator": "Color ASCII Typing Animation" # New display name
}

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("ComfyUI_ASCIINodes: Loaded ASCII Art (Static), ASCII Typing Animation, and Color ASCII Typing Animation Nodes")

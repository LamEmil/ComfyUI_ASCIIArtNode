# ComfyUI_ASCIINodes/__init__.py

# Import the node classes from their respective .py files
# These files should be in the same directory as this __init__.py file.

from .ascii_art_node import ASCIIArtGeneratorNode 
from .ascii_animation_node import ASCIIAnimationGeneratorNode
from .color_ascii_animation_node import ColorASCIIAnimationGeneratorNode
from .realistic_color_ascii_animation_node import RealisticColorASCIIAnimationNode
from .video_to_color_static_ascii_node import VideoToColorStaticASCIIArtNode
from .video_to_dynamic_color_ascii_node import VideoToDynamicColorASCIIArtNode
from .two_pass_typing_node import TwoPassTypingColorASCIIAnimationNode
from .sequential_two_pass_node import SequentialTwoPassTypingNode # Newest import

# A dictionary that ComfyUI uses to map node_class names to node_display_names
NODE_CLASS_MAPPINGS = {
    "ASCIIArtGenerator": ASCIIArtGeneratorNode,
    "ASCIIAnimationGenerator": ASCIIAnimationGeneratorNode,
    "ColorASCIIAnimationGenerator": ColorASCIIAnimationGeneratorNode,
    "RealisticColorASCIIAnimationGenerator": RealisticColorASCIIAnimationNode,
    "VideoToColorStaticASCIIArt": VideoToColorStaticASCIIArtNode,
    "VideoToDynamicColorASCIIArt": VideoToDynamicColorASCIIArtNode,
    "TwoPassTypingColorASCIIAnimation": TwoPassTypingColorASCIIAnimationNode, # Concurrent version
    "SequentialTwoPassTypingColorASCIIAnimation": SequentialTwoPassTypingNode # Newest mapping (Sequential version)
}

# A dictionary that ComfyUI uses to map node_class names to their display names
# This is what will appear in the ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "ASCIIArtGenerator": "ASCII Art Generator (Static)",
    "ASCIIAnimationGenerator": "ASCII Typing Animation Generator",
    "ColorASCIIAnimationGenerator": "Color ASCII Typing Animation",
    "RealisticColorASCIIAnimationGenerator": "Realistic Color ASCII Typing Animation",
    "VideoToColorStaticASCIIArt": "Video to Color Static ASCII Art",
    "VideoToDynamicColorASCIIArt": "Video to Dynamic Color ASCII Art",
    "TwoPassTypingColorASCIIAnimation": "Two-Pass Color Typing (Concurrent)", # Clarified name
    "SequentialTwoPassTypingColorASCIIAnimation": "Sequential Two-Pass Color Typing" # Newest display name
}

# Export the mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("ComfyUI_ASCIINodes: Loaded Static ASCII, Basic Animation, Color Animation, Realistic Animation, Static Video ASCII, Dynamic Video ASCII, Concurrent Two-Pass Typing, and Sequential Two-Pass Typing ASCII Nodes")

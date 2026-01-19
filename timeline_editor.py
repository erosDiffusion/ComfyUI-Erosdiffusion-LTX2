class LTXVTimelineEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": ("STRING", {
                    "multiline": True, 
                    "default": '(2.0s: "Describe your scene here" { 0.0s: $0:1.0 })',
                    "tooltip": "The generated script string. Use the Custom UI (below or hidden) to visually edit chunks, prompts, and guides."
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script_string",)
    FUNCTION = "execute"
    CATEGORY = "ErosDiffusion/LTXV"
    OUTPUT_NODE = False

    DESCRIPTION = "A visual editor for constructing LTXV Scene Extender scripts. \nUse the Custom UI to add Timeline Chunks, set durations, transitions, and place Image Guides ($0, $1) at specific timestamps."

    def execute(self, script):
        return (script,)

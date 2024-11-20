import gradio as gr
import asyncio
import nest_asyncio
import os
from PIL import Image as PILImage
import numpy as np
from typing import Tuple, List, Dict
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nest_asyncio.apply()

class GradioInterface:
    def __init__(self, generator):
        self.generator = generator
        self.latest_result = None
        self.processing_steps = [
            "Analyzing facial features...",
            "Detecting body pose...",
            "Matching spirit animals...",
            "Generating visualizations..."
        ]

    def process_image(self, input_image: np.ndarray, progress=gr.Progress()) -> Tuple[str, List[np.ndarray], str, str]:
        try:
            temp_path = "temp_input.jpg"
            PILImage.fromarray(input_image).save(temp_path)

            # Show processing progress
            for i, step in enumerate(self.processing_steps):
                progress(i/len(self.processing_steps), desc=step)
                time.sleep(0.5)  # Give user time to see each step

            # Process image
            result = asyncio.get_event_loop().run_until_complete(
                self.generator.process_image(temp_path)
            )
            self.latest_result = result

            # Prepare all generated images
            generated_images = []
            for path in result['generated_images'].values():
                img = PILImage.open(path)
                generated_images.append(np.array(img))

            # Create analysis markdown with enhanced formatting
            analysis_md = self._create_detailed_analysis(result)

            # Create summary
            summary = self._create_summary(result)

            # Create overview panel
            overview = self._create_overview(result)

            return analysis_md, generated_images, summary, overview

        except Exception as e:
            error_message = f"""
            ### ❌ Error Occurred

            Something went wrong while processing your image:
            - {str(e)}

            Please try again with a different photo or ensure your image:
            - Shows a clear view of your face
            - Has good lighting
            - Shows your full body pose
            """
            return error_message, [], "An error occurred", ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _create_detailed_analysis(self, result: Dict) -> str:
        """Creates a detailed, well-formatted analysis markdown"""
        analysis_md = f"""
        ## 🎭 Analysis Results

        ### 👤 Facial Features
        - **Shape:** {result['face_analysis']['features']['face_shape'].title()}
        - **Expression:** {result['face_analysis']['emotion'].title()}
        - **Confidence:** {'⭐' * int(result['face_analysis']['confidence'] * 5)} ({result['face_analysis']['confidence']:.2f})

        ### 🦁 Spirit Animal Analysis

        #### Primary Spirit Animal: {result['spirit_animal'].primary_animal.title()}
        *{result['spirit_animal'].reasoning}*

        #### Feature-Specific Spirit Animals:
        """

        # Add each animal with emoji indicators
        emoji_map = {
            "eyes": "👁️",
            "expression": "😊",
            "face_shape": "💫",
            "posture": "🧘",
            "energy": "⚡"
        }

        for animal in result['spirit_animal'].animals:
            feature = animal['associated_feature'].lower()
            emoji = emoji_map.get(feature, "✨")

            analysis_md += f"""
            {emoji} **{animal['associated_feature']}**: {animal['animal']} ({animal['category']})
            - *Connection*: {animal['reasoning']}
            """

        return analysis_md

    def _create_summary(self, result: Dict) -> str:
        """Creates a concise summary of the results"""
        primary = result['spirit_animal'].primary_animal
        others = [animal['animal'] for animal in result['spirit_animal'].animals
                 if animal['animal'] != primary]

        summary = f"🎉 Your Spirit Animals | Primary: {primary.title()}"
        if others:
            summary += f" | Secondary: {', '.join(a.title() for a in others)}"

        return summary

    def _create_overview(self, result: Dict) -> str:
      """Creates an overview panel with key insights"""
      characteristics = result['spirit_animal'].characteristics

      # Create traits list separately
      traits_list = '\n'.join(f'- {trait}' for trait in characteristics)

      return f"""
      ### 🔮 Key Insights

      Your spirit animals reflect these core traits:
      {traits_list}

      **Pose Energy:** {result['pose_data']['pose_type'].title()}
      **Overall Confidence:** {result['spirit_animal'].confidence:.2%}
      """
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Spirit Animal Generator",
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="blue"
            ),
            css=".gradio-container {background-color: #f8f9fa}"
        ) as interface:
            gr.Markdown("""
            # 🐾 Spirit Animal Generator
            Discover the animals that reflect your essence through AI analysis
            """)

            with gr.Tabs():
                with gr.TabItem("✨ Generate"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            input_image = gr.Image(
                                label="Upload Your Photo",
                                type="numpy",
                                elem_id="input-image",
                                height=300
                            )
                            with gr.Row():
                                submit_btn = gr.Button(
                                    "🔮 Discover Your Spirit Animals",
                                    variant="primary"
                                )
                                clear_btn = gr.Button(
                                    "🔄 Clear",
                                    variant="secondary"
                                )

                    # Quick overview panel
                    overview_panel = gr.Markdown(
                        label="Overview",
                        value="Upload a photo to begin your spirit animal journey...",
                        elem_id="overview-panel"
                    )

                    with gr.Row():
                        # Results summary
                        result_summary = gr.Markdown(
                            label="Summary",
                            elem_id="result-summary"
                        )

                    # Gallery with generated images
                    gallery = gr.Gallery(
                        label="Your Spirit Animals",
                        show_label=True,
                        elem_id="gallery",
                        columns=[2, 2],
                        height="auto",
                        object_fit="contain"
                    )

                    with gr.Accordion("🔍 Detailed Analysis", open=False):
                        output_analysis = gr.Markdown(
                            label="Analysis"
                        )

                with gr.TabItem("ℹ️ How It Works"):
                    gr.Markdown("""
                    ### 🎯 Getting Started
                    1. **Upload a Photo**
                       - Clear face visibility
                       - Good lighting
                       - Full body visible if possible

                    ### 🔍 The Analysis Process
                    1. **Facial Analysis**
                       - Shape and symmetry
                       - Expression and emotions
                       - Key features

                    2. **Pose Detection**
                       - Body position
                       - Movement energy
                       - Overall presence

                    3. **Spirit Animal Matching**
                       - Primary animal connection
                       - Feature-specific matches
                       - Personality traits

                    ### 🎨 Visualization
                    - Unique AI-generated artwork
                    - Multiple perspectives
                    - Natural environments

                    ### 🔒 Privacy
                    Your photos are processed securely and not stored permanently.
                    """)

            # Event handlers
            submit_btn.click(
                fn=self.process_image,
                inputs=[input_image],
                outputs=[output_analysis, gallery, result_summary, overview_panel]
            )

            clear_btn.click(
                fn=lambda: (None, None, "Ready for a new discovery...", "Upload a photo to begin..."),
                inputs=[],
                outputs=[input_image, gallery, result_summary, overview_panel]
            )

        return interface
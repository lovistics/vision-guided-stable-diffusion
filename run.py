from main import SpiritAnimalGenerator, setup_environment
from interface import GradioInterface

def run_demo():
    """Main demo function"""
    try:
        print("🎉 Welcome to Spirit Animal Generator!")

        # Setup
        config = setup_environment()
        generator = SpiritAnimalGenerator(config['openai_api_key'])

        # Create
        interface = GradioInterface(generator)
        demo = interface.create_interface()

        # Launch
        print("\n🚀 Launching web interface...")
        demo.launch(
            share=True,
            show_error=True,
            favicon_path="🐾"
        )

    except Exception as e:
        print(f"\n❌ Error in demo: {str(e)}")
        raise

if __name__ == "__main__":
    run_demo()
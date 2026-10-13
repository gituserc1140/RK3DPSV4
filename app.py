import logging
from streamlit_app import run_app

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main entry point for the Streamlit application.

    This function initializes and runs the Streamlit app defined in `streamlit_app.py`.
    It includes error handling to log exceptions that occur during execution.
    """
    try:
        run_app()
    except Exception:
        logging.exception("An error occurred while starting the Streamlit app.")
        raise

if __name__ == "__main__":
    main()
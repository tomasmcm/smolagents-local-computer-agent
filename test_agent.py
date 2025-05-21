import unittest
import os
import tempfile
import shutil
import time
from PIL import Image

# Assuming e2bqwen.py is in the same directory or accessible via PYTHONPATH
from e2bqwen import E2BVisionAgent
from smolagents import InferenceClientModel # For dummy model instantiation

# Ensure TMP_DIR exists, as the agent might write logs/screenshots there via data_dir
# This is handled by TemporaryDirectory in setUp for data_dir,
# but if the agent itself uses a global TMP_DIR, it should be checked.
# The agent's data_dir is what matters for screenshots.
TMP_DIR = "./tmp/" # Matches app.py, agent uses data_dir inside this
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
    print(f"Created {TMP_DIR} for test logs/artifacts.")


class TestE2BVisionAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment; create a temporary directory for agent data."""
        self.temp_dir_obj = tempfile.TemporaryDirectory(dir=TMP_DIR, prefix="test_agent_")
        self.data_dir = self.temp_dir_obj.name
        # Dummy model for agent initialization, as tests don't rely on model inference
        # Using a non-existent endpoint for the dummy model to avoid actual network calls.
        self.model = InferenceClientModel(
            model_id="http://localhost/dummy_model_for_testing", 
            token="dummy_token" 
        )
        self.agent = None # Initialize agent to None for cleanup

    def tearDown(self):
        """Clean up test environment; remove the temporary directory."""
        if self.agent:
            try:
                self.agent.close()
            except Exception as e:
                print(f"Error closing agent in tearDown: {e}")
        self.temp_dir_obj.cleanup()

    def test_agent_initialization(self):
        """Test that the E2BVisionAgent initializes correctly."""
        print("Running test_agent_initialization...")
        try:
            self.agent = E2BVisionAgent(model=self.model, data_dir=self.data_dir)
            self.assertIsNotNone(self.agent.client, "Docker client should be initialized.")
            self.assertIsNotNone(self.agent.container, "Docker container should be started.")
            self.assertIsNotNone(self.agent.vnc_client, "VNC client should be connected.")
            print("Agent initialized successfully.")
        except Exception as e:
            self.fail(f"Agent initialization failed: {e}")
        # Agent will be closed in tearDown

    def test_take_screenshot(self):
        """Test that screen capture returns a PIL Image."""
        print("Running test_take_screenshot...")
        try:
            self.agent = E2BVisionAgent(model=self.model, data_dir=self.data_dir)
            self.assertIsNotNone(self.agent.vnc_client, "VNC client not initialized before screenshot.")
            
            # Allow a moment for VNC server to be fully ready if needed
            time.sleep(2) # Small delay before first capture
            
            img = self.agent.vnc_client.capture_screen()
            self.assertIsInstance(img, Image.Image, "capture_screen should return a PIL Image.")
            print("Screenshot captured successfully.")
        except Exception as e:
            self.fail(f"Taking screenshot failed: {e}")
        # Agent will be closed in tearDown

    def test_open_url_tool(self):
        """Test the open_url tool execution."""
        print("Running test_open_url_tool...")
        try:
            self.agent = E2BVisionAgent(model=self.model, data_dir=self.data_dir)
            self.assertIsNotNone(self.agent.tools, "Agent tools not initialized.")
            
            open_url_tool = self.agent.tools.get("open_url")
            self.assertIsNotNone(open_url_tool, "open_url tool not found in agent tools.")
            
            print("Executing open_url tool with https://www.example.com...")
            # The main assertion is that this call does not raise an exception.
            open_url_tool("https://www.example.com") 
            
            print("Waiting for URL to open (5 seconds)...")
            time.sleep(5) # Give time for the action to attempt execution
            
            # Capture a screenshot to ensure VNC interaction is still possible
            img = self.agent.vnc_client.capture_screen()
            self.assertIsInstance(img, Image.Image, "capture_screen after open_url should return a PIL Image.")
            print("open_url tool executed and subsequent screenshot captured successfully.")
            
        except Exception as e:
            # If open_url_tool itself fails, this will catch it.
            self.fail(f"open_url tool execution or subsequent screenshot failed: {e}")
        # Agent will be closed in tearDown

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    unittest.main()

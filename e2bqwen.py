import os
import time
import unicodedata
from datetime import datetime
from io import BytesIO
from typing import List

import docker
import vncdotool.api # Added vncdotool
from PIL import Image, ImageDraw

# SmolaAgents imports
from smolagents import CodeAgent, HfApiModel, tool
from smolagents.agent_types import AgentImage
from smolagents.memory import ActionStep, TaskStep
from smolagents.monitoring import LogLevel

E2B_SYSTEM_PROMPT_TEMPLATE = """You are a desktop automation assistant that can control a remote desktop environment. The current date is <<current_date>>.

<action process>
You will be given a task to solve in several steps. At each step you will perform an action.
After each action, you'll receive an updated screenshot. 
Then you will proceed as follows, with these sections: don't skip any!

Short term goal: ...
What I see: ...
Reflection: ...
Action:
```python
click(254, 308)
```<end_code>

Akways format your action ('Action:' part) as Python code blocks as shown above.
</action_process>

<tools>
On top of performing computations in the Python code snippets that you create, you only have access to these tools to interact with the desktop, no additional ones:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
{%- endfor %}
</tools>

<click_guidelines>
Look at elements on the screen to determine what to click or interact with.
The desktop has a resolution of <<resolution_x>>x<<resolution_y>> pixels, take it into account to decide clicking coordinates. NEVER USE HYPOTHETIC OR ASSUMED COORDINATES, USE TRUE COORDINATES that you can see from the screenshot.
Use precise coordinates based on the current screenshot for mouse movements and clicks. 
Whenever you click, MAKE SURE to click in the middle of the button, text, link or any other clickable element. Not under, not on the side. IN THE MIDDLE, else you risk to miss it.
In menus it is always better to click in the middle of the text rather than in the tiny icon. Calculate extremelly well the coordinates. A mistake here can make the full task fail.
Sometimes you may have missed a click, so never assume that you're on the right page, always make sure that your previous action worked.
In the screenshot you will see a green crosshair displayed over the position of your last click: this way can inspect if the mouse pointer is off of the targeted element, pay special attention to it.
</click_guidelines>

<task_resolution_example>
For a task like "Open a text editor and type 'Hello World'":
Step 1:
Short term goal: I want to open a text editor.
What I see: I am on the homepage of my desktop. I see the applications
Reflection: I think that a notes application would fit in the Applications menu, let's open it. I'll carefully click in the middle of the text 'Applications'/
Action:
```python
click(51, 8) 
```<end_code>

Step 2:
Short term goal: I want to open a text editor.
What I see: I am on the homepage of my desktop, with the applications menu open. I see an Accessories section, I see it is a section in the menu thanks to the tiny white triangle after the text accessories.
Reflection: I think that a notes application would fit the Accessories section. I SHOULD NOT try to move through the menus with scroll, it won't work:
I'll look for Accessories and click on it being very precise, clicking in the middle of the text 'Accessories'.
Action:
```python
click(76, 195) 
```<end_code>

Step 3:
Short term goal: I want to open a text editor.
What I see: I am under the Accessories menu. Under the open submenu Accessories, I've found 'Text Editor'.
Reflection: This must be my notes app. I remember that menus are navigated through clicking. I will now click on it being very precise, clicking in the middle of the text 'Text Editor'.
Action:
```python
click(251, 441) 
```<end_code>

Step 4:
Short term goal: I want to open a text editor.
What I see: I am still under the Accessories menu. Nothing has changed compared to previous screenshot. Under the open submenu Accessories, I still see 'Text Editor'. The green cross is off from the element.
Reflection: My last click must have been off. Let's correct this. I will click the correct place, right in the middle of the element.
Action:
```python
click(241, 441) 
```<end_code>

Step 5:
Short term goal: I want to type 'Hello World'.
What I see: I have opened a Notepad. The Notepad app is open on an empty page
Reflection: Now Notepad is open as intended, time to type text.
Action:
```python
type_text("Hello World")
```<end_code>

Step 6:
Short term goal: I want to type 'Hello World'.
What I see: The Notepad app displays 'Hello World'
Reflection: Now that I've 1. Opened the notepad and 2. typed 'Hello World', and 3. the result seems correct, I think the Task is completed. I will return a confirmation that the task is completed.
Action:
```python
final_answer("Done")
```<end_code>
</task_resolution_example>

<general_guidelines>
Always analyze the latest screenshot carefully before performing actions.
You can wait for appropriate loading times using the wait() tool. But don't wait forever, sometimes you've just misclicked and the process didn't launch.
Execute one action at a time: don't try to pack a click and typing in one action.
On each step, look at the last screenshot and action to validate if previous steps worked and decide the next action. If you repeated an action already without effect, it means that this action is useless: don't repeat it and try something else.
Use click to move through menus on the desktop and scroll for web and specific applications.
Always analyze the latest screenshot carefully before performing actions.
Desktop menus usually expand with more options, the tiny triangle next to some text in a menu means that menu expands. For example in Office in the Applications menu expands showing presentation or writing applications. 
NEVER CLICK THE WEB BROWSER ICON TO OPEN THE WEB BROWSER: use open_url directly.
In browser, ignore any sign-in popups while they don't interfere with the elements you want to interact with.
</general_guidelines>
""".replace("<<current_date>>", datetime.now().strftime("%A, %d-%B-%Y"))


def draw_marker_on_image(image_copy, click_coordinates):
    x, y = click_coordinates
    draw = ImageDraw.Draw(image_copy)
    cross_size, linewidth = 10, 3
    # Draw cross
    draw.line((x - cross_size, y, x + cross_size, y), fill="green", width=linewidth)
    draw.line((x, y - cross_size, x, y + cross_size), fill="green", width=linewidth)
    # Add a circle around it for better visibility
    draw.ellipse(
        (
            x - cross_size * 2,
            y - cross_size * 2,
            x + cross_size * 2,
            y + cross_size * 2,
        ),
        outline="green",
        width=linewidth,
    )
    return image_copy


def get_agent_summary_erase_images(agent):
    for memory_step in agent.memory.steps:
        if hasattr(memory_step, "observations_images"):
            memory_step.observations_images = None
        if hasattr(memory_step, "task_images"):
            memory_step.task_images = None
    return agent.write_memory_to_messages()


class E2BVisionAgent(CodeAgent):
    """Agent for e2b desktop automation with Qwen2.5VL vision capabilities"""

    def __init__(
        self,
        model: HfApiModel,
        data_dir: str,
        # desktop: Sandbox, # Sandbox replaced with Docker
        tools: List[tool] = None,
        max_steps: int = 200,
        verbosity_level: LogLevel = 2,
        planning_interval: int = None,
        use_v1_prompt: bool = False,
        **kwargs,
    ):
        # self.desktop = desktop # Sandbox replaced with Docker
        self.client = docker.from_env()  # Initialize Docker client
        
        # Build and run Docker container with VNC server from local Dockerfile
        self.image_name = "local/desktop-automation-env:latest"
        self.vnc_port = 5901 # Matches Dockerfile EXPOSE and CMD
        self.vnc_password = "password" # Matches Dockerfile ENV VNC_PW
        self.width, self.height = 1280, 960 # Matches Dockerfile GEOMETRY and CMD

        try:
            self.client.images.get(self.image_name)
            print(f"Image '{self.image_name}' found locally.")
        except docker.errors.ImageNotFound:
            print(f"Image '{self.image_name}' not found locally. Building...")
            try:
                # Assuming Dockerfile is in the root of the repository context (path=".")
                logs = self.client.images.build(path=".", dockerfile="Dockerfile", tag=self.image_name, rm=True)
                for log_chunk in logs:
                    if 'stream' in log_chunk:
                        for line in log_chunk['stream'].splitlines():
                            print(f"[Build Log] {line}")
                    elif 'error' in log_chunk:
                        print(f"[Build Error] {log_chunk['errorDetail']}")
                    else:
                        print(f"[Build Log] {log_chunk}")
                print(f"Image '{self.image_name}' built successfully.")
            except docker.errors.BuildError as e:
                print(f"Failed to build image '{self.image_name}': {e}")
                # Optionally, print detailed build logs from the exception
                for log_entry in e.build_log:
                    if 'stream' in log_entry:
                        print(f"[Build Log from Exception] {log_entry['stream']}")
                    elif 'error' in log_entry:
                         print(f"[Build Error from Exception] {log_entry['error']}")
                raise # Re-raise the exception to stop execution if build fails
        
        self.container = self.client.containers.run(
            self.image_name,
            detach=True,
            ports={f'{self.vnc_port}/tcp': self.vnc_port},
            # Environment variables for VNC might be needed if not baked into image CMD fully,
            # but our CMD handles geometry and display number. Password is set by vncpasswd.
        )
        print(f"Started container '{self.container.name}' ({self.container.id}) from image '{self.image_name}'. VNC exposed on host port {self.vnc_port}")
        
        # Allow time for VNC server to start within the container
        print("Waiting for VNC server to start...")
        time.sleep(10) # Adjust as needed, might depend on system performance

        # Connect VNC client
        try:
            self.vnc_client = vncdotool.api.connect(f'localhost:{self.vnc_port}', password=self.vnc_password)
            print(f"Connected VNC client to localhost:{self.vnc_port}")
        except Exception as e:
            print(f"Failed to connect to VNC server: {e}")
            # Attempt to get container logs for debugging
            try:
                container_logs = self.container.logs(stdout=True, stderr=True, tail=50)
                print(f"Last 50 lines of container logs for {self.container.name}:\n{container_logs.decode('utf-8')}")
            except Exception as log_e:
                print(f"Could not retrieve container logs: {log_e}")
            raise # Re-raise the connection error

        self.data_dir = data_dir
        self.planning_interval = planning_interval
        
        # Screen size is now set based on Dockerfile's geometry
        print(f"Screen size set to: {self.width}x{self.height} (from Dockerfile)")

        # Set up temp directory
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Screenshots and steps will be saved to: {self.data_dir}")

        self.use_v1_prompt = use_v1_prompt
        # Initialize base agent
        super().__init__(
            tools=tools or [],
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=self.planning_interval,
            stream_outputs=True,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = E2B_SYSTEM_PROMPT_TEMPLATE.replace( 
            "<<resolution_x>>", str(self.width)
        ).replace("<<resolution_y>>", str(self.height))

        # Add screen info to state
        self.state["screen_width"] = self.width 
        self.state["screen_height"] = self.height 

        # Add default tools
        self.logger.log("Setting up agent tools...") 
        self._setup_desktop_tools() 
        self.step_callbacks.append(self.take_screenshot_callback) # Uncommented for VNC

    def _setup_desktop_tools(self): 
        """Register all desktop tools"""

        @tool
        def click(x: int, y: int) -> str:
            """
            Performs a left-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.vnc_client.mouse_move(x, y)
            self.vnc_client.mouse_press(1) # 1 for left button
            self.click_coordinates = [x, y] # For drawing marker on screenshot
            self.logger.log(f"Clicked at coordinates ({x}, {y})")
            return f"Clicked at coordinates ({x}, {y})"

        @tool
        def right_click(x: int, y: int) -> str:
            """
            Performs a right-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.vnc_client.mouse_move(x, y)
            self.vnc_client.mouse_press(3) # 3 for right button
            self.click_coordinates = [x, y]
            self.logger.log(f"Right-clicked at coordinates ({x}, {y})")
            return f"Right-clicked at coordinates ({x}, {y})"

        @tool
        def double_click(x: int, y: int) -> str:
            """
            Performs a double-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.vnc_client.mouse_move(x, y)
            # vncdotool's mouse_click does a press and release.
            # Some VNC servers might need a slight delay.
            # Using two mouse_click calls for double_click.
            self.vnc_client.mouse_click(1) 
            time.sleep(0.05) # 50ms delay, adjust if needed
            self.vnc_client.mouse_click(1)
            self.click_coordinates = [x, y]
            self.logger.log(f"Double-clicked at coordinates ({x}, {y})")
            return f"Double-clicked at coordinates ({x}, {y})"

        @tool
        def move_mouse(x: int, y: int) -> str:
            """
            Moves the mouse cursor to the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.vnc_client.mouse_move(x, y)
            self.logger.log(f"Moved mouse to coordinates ({x}, {y})")
            return f"Moved mouse to coordinates ({x}, {y})"

        def normalize_text(text): # Keep this helper
            return "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if not unicodedata.combining(c)
            )

        @tool
        def type_text(text: str) -> str:
            """
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
            """
            clean_text = normalize_text(text)
            # For vncdotool, key_press types a single character.
            # For strings, it's often better to use type_string if available,
            # but iterating key_press is a common fallback.
            # Special characters might need mapping or different handling.
            for char_to_type in clean_text:
                self.vnc_client.key_press(char_to_type)
            self.logger.log(f"Typed text: '{clean_text}'")
            return f"Typed text: '{clean_text}'"

        @tool
        def press_key(key: str) -> str:
            """
            Presses a keyboard key. Common keys: 'enter', 'space', 'backspace', 'alt_l', 'ctrl_l', 'left', 'right', 'up', 'down', 'esc', 'tab'.
            Args:
                key: The key to press (e.g. "enter", "space", "backspace", "ctrl_l", "alt_l", "left").
            """
            # Map common names to vncdotool names if necessary, though many are direct.
            # Examples: 'enter', 'space', 'backspace', 'alt_l', 'ctrl_l', 'left', 'right', 'up', 'down', 'esc', 'tab', 'f5'
            self.vnc_client.key_press(key)
            self.logger.log(f"Pressed key: {key}")
            return f"Pressed key: {key}"

        @tool
        def go_back() -> str:
            """
            Goes back to the previous page in the browser (Alt+Left Arrow).
            """
            self.vnc_client.key_down('alt_l') # Use 'alt_l' for Left Alt
            self.vnc_client.key_press('left')
            self.vnc_client.key_up('alt_l')
            self.logger.log("Went back one page (Alt+Left Arrow)")
            return "Went back one page"

        @tool
        def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Clicks at [x1, y1], drags mouse to [x2, y2], then releases click.
            Args:
                x1: origin x coordinate
                y1: origin y coordinate
                x2: end x coordinate
                y2: end y coordinate
            """
            self.vnc_client.mouse_move(x1, y1)
            self.vnc_client.mouse_down(1) # Press left mouse button
            time.sleep(0.1) # Short delay for drag to register
            self.vnc_client.mouse_move(x2, y2)
            time.sleep(0.1) # Short delay
            self.vnc_client.mouse_up(1)   # Release left mouse button
            message = f"Dragged and dropped from [{x1}, {y1}] to [{x2}, {y2}]"
            self.logger.log(message)
            return message

        @tool
        def scroll(x: int, y: int, direction: str = "down", amount: int = 2) -> str:
            """
            Moves the mouse to selected coordinates, then uses the scroll wheel.
            Args:
                x: The x coordinate (horizontal position) of the element to scroll/zoom
                y: The y coordinate (vertical position) of the element to scroll/zoom
                direction: The direction to scroll ("up" or "down"), defaults to "down".
                amount: The amount to scroll (number of wheel "clicks"). A good amount is 1 or 2.
            """
            self.vnc_client.mouse_move(x, y)
            scroll_value = amount if direction == "down" else -amount
            # vncdotool's wheel method usually takes an integer.
            # Positive for down, negative for up. Let's assume 'wheel' is the method.
            # If not, it might be scroll_up/scroll_down called 'amount' times.
            # Based on common usage, wheel(delta) is typical.
            for _ in range(abs(scroll_value)):
                self.vnc_client.wheel(1 if scroll_value > 0 else -1) # wheel(1) for down, wheel(-1) for up
                time.sleep(0.05) # Small delay between scroll events
            
            message = f"Scrolled {direction} by {amount} at ({x},{y})"
            self.logger.log(message)
            return message
            
        @tool
        def wait(seconds: float) -> str: # This tool remains the same
            """
            Waits for the specified number of seconds. Very useful in case the prior order is still executing.
            Args:
                seconds: Number of seconds to wait, generally 3 is enough.
            """
            time.sleep(seconds)
            self.logger.log(f"Waited for {seconds} seconds")
            return f"Waited for {seconds} seconds"

        @tool
        def open_url(url: str) -> str:
            """
            Opens a URL using xdg-open in a terminal.
            Args:
                url: The URL to open
            """
            # Make sure URL has http/https prefix
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            # Open terminal (Ctrl+Alt+T is common for LXDE)
            self.vnc_client.key_down('ctrl_l')
            self.vnc_client.key_down('alt_l')
            self.vnc_client.key_press('t')
            self.vnc_client.key_up('alt_l')
            self.vnc_client.key_up('ctrl_l')
            time.sleep(1.5) # Wait for terminal to open

            # Type command
            command = f"xdg-open \"{url}\"\n" # Ensure URL is quoted, add newline to execute
            for char_to_type in command:
                self.vnc_client.key_press(char_to_type)
                time.sleep(0.02) # Small delay between key presses for reliability in terminal
            
            # Give it time to load (optional, can be handled by subsequent wait() if needed)
            time.sleep(3) 
            self.logger.log(f"Attempted to open URL: {url} using xdg-open")
            return f"Attempted to open URL: {url} using xdg-open"

        @tool
        def find_on_page_ctrl_f(search_string: str) -> str:
            """
            Simulates Ctrl+F, types search string, presses Enter, then Esc.
            Args:
                search_string: The string to search for on the page.
            """
            self.vnc_client.key_down('ctrl_l')
            self.vnc_client.key_press('f')
            self.vnc_client.key_up('ctrl_l')
            time.sleep(0.5) # Wait for find bar

            clean_text = normalize_text(search_string)
            for char_to_type in clean_text:
                self.vnc_client.key_press(char_to_type)
                time.sleep(0.02)
            time.sleep(0.3)

            self.vnc_client.key_press('enter')
            time.sleep(0.3)
            self.vnc_client.key_press('esc') # Close the find bar
            
            output_message = f"Performed Ctrl+F and searched for '{clean_text}'"
            self.logger.log(output_message)
            return output_message

        # Register the tools
        self.tools["click"] = click
        self.tools["right_click"] = right_click
        self.tools["double_click"] = double_click
        self.tools["move_mouse"] = move_mouse
        self.tools["type_text"] = type_text
        self.tools["press_key"] = press_key
        self.tools["scroll"] = scroll
        self.tools["wait"] = wait
        self.tools["open_url"] = open_url
        self.tools["go_back"] = go_back
        self.tools["drag_and_drop"] = drag_and_drop
        self.tools["find_on_page_ctrl_f"] = find_on_page_ctrl_f

    def take_screenshot_callback(self, memory_step: ActionStep, agent=None) -> None:
        """Callback that takes a screenshot + memory snapshot after a step completes"""
        self.logger.log("Analyzing screen content...")

        current_step = memory_step.step_number

        # time.sleep(2.5) # Original delay, may need adjustment with VNC
        # With VNC, actions are more direct. A shorter delay might be fine,
        # or it could be tool-dependent (e.g. after open_url).
        # For now, reducing it slightly.
        time.sleep(1.0) 

        # Replace desktop.screenshot with vnc_client.capture_screen()
        # capture_screen() is expected to return a PIL Image object.
        image = self.vnc_client.capture_screen()
        
        if image is None:
            self.logger.log("Failed to capture screenshot.", level=LogLevel.ERROR)
            # Handle error: maybe raise an exception or set observations_images to None
            memory_step.observations_images = None
            return

        # Create a filename with step number
        screenshot_path = os.path.join(self.data_dir, f"step_{current_step:03d}.png")
        try:
            image.save(screenshot_path)
        except Exception as e:
            self.logger.log(f"Error saving screenshot: {e}", level=LogLevel.ERROR)
            memory_step.observations_images = None # Or handle differently
            return


        image_copy = image.copy()

        if getattr(self, "click_coordinates", None):
            print("DRAWING MARKER") # Keep for debugging
            image_copy = draw_marker_on_image(image_copy, self.click_coordinates)

        self.last_marked_screenshot = AgentImage(screenshot_path)
        print(f"Saved screenshot for step {current_step} to {screenshot_path}") # Keep for debugging

        # The rest of the memory management logic remains the same
        for previous_memory_step in (
            agent.memory.steps
        ):  # Remove previous screenshots from logs for lean processing
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 1
            ):
                previous_memory_step.observations_images = None
            elif isinstance(previous_memory_step, TaskStep):
                previous_memory_step.task_images = None

            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number == current_step - 1
            ):
                if (
                    previous_memory_step.tool_calls
                    and getattr(previous_memory_step.tool_calls[0], "arguments", None)
                    and memory_step.tool_calls
                    and getattr(memory_step.tool_calls[0], "arguments", None)
                ):
                    if (
                        previous_memory_step.tool_calls[0].arguments
                        == memory_step.tool_calls[0].arguments
                    ):
                        memory_step.observations += "\nWARNING: You've executed the same action several times in a row. MAKE SURE TO NOT UNNECESSARILY REPEAT ACTIONS."

        # Add the marker-edited image to the current memory step
        memory_step.observations_images = [image_copy]
        
        self.click_coordinates = None  # Reset click marker

    def close(self):
        """Clean up resources"""
        # if self.desktop: # Desktop specific
        #     print("Stopping e2b stream and killing sandbox...")
        #     self.desktop.stream.stop()
        #     self.desktop.kill()
        #     print("E2B sandbox terminated")
        if hasattr(self, 'vnc_client') and self.vnc_client:
            self.vnc_client.disconnect()
            print("VNC client disconnected.")
        if hasattr(self, 'container') and self.container:
            print(f"Stopping and removing container '{self.container.name}' ({self.container.id})...")
            try:
                self.container.stop()
                self.container.remove()
                print("Docker container stopped and removed.")
            except docker.errors.NotFound:
                print(f"Container {self.container.id} not found, likely already removed.")
            except Exception as e:
                print(f"Error stopping/removing container: {e}")

        if hasattr(self, 'client') and self.client:
            self.client.close()
            print("Docker client closed.")

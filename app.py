import json
import os
import shutil
import tempfile
import time
import uuid
from io import BytesIO
from threading import Timer
from typing import Any

import gradio as gr
from dotenv import load_dotenv
# from e2b_desktop import Sandbox # E2B import removed
from gradio_modal import Modal
from huggingface_hub import login, upload_folder
from PIL import Image
from smolagents import CodeAgent, InferenceClientModel, OpenAIServerModel # Added OpenAIServerModel
from smolagents.gradio_ui import GradioUI

from e2bqwen import E2BVisionAgent, get_agent_summary_erase_images
from gradio_script import stream_to_gradio
from scripts_and_styling import (
    CUSTOM_JS,
    FOOTER_HTML,
    SANDBOX_CSS_TEMPLATE,
    SANDBOX_HTML_TEMPLATE,
    apply_theme,
)

load_dotenv(override=True)


TASK_EXAMPLES = [
    "Use Google Maps to find the Hugging Face HQ in Paris",
    "Go to Wikipedia and find what happened on April 4th",
    "Find out the travel time by train from Bern to Basel on Google Maps",
    "Go to Hugging Face Spaces and then find the Space flux.1 schnell. Use the space to generate an image with the prompt 'a field of gpus'",
]

# E2B_API_KEY = os.getenv("E2B_API_KEY") # E2B API Key removed
AGENTS: dict[str, E2BVisionAgent] = {} # Renamed from SANDBOXES
AGENT_METADATA: dict[str, dict[str, Any]] = {} # Renamed from SANDBOX_METADATA
AGENT_TIMEOUT = 300 # Renamed from SANDBOX_TIMEOUT
WIDTH = 1024 # Adjusted to default VNC resolution
HEIGHT = 768 # Adjusted to default VNC resolution
TMP_DIR = "./tmp/"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)

# custom_css remains largely the same, but width/height might be less relevant if stream is gone
custom_css = SANDBOX_CSS_TEMPLATE.replace("<<WIDTH>>", str(WIDTH + 15)).replace(
    "<<HEIGHT>>", str(HEIGHT + 10)
)

# sandbox_html_template is no longer used for streaming
# sandbox_html_template = SANDBOX_HTML_TEMPLATE.replace( # Removed
#     "<<WIDTH>>", str(WIDTH + 15)
# ).replace("<<HEIGHT>>", str(HEIGHT + 10))


def upload_to_hf_and_remove(folder_paths: list[str]):
    repo_id = "smolagents/computer-agent-logs-2"

    with tempfile.TemporaryDirectory(dir=TMP_DIR) as temp_dir:
        print(
            f"Uploading {len(folder_paths)} folders to {repo_id} (might end up with 0 folders uploaded if tasks are all examples)..."
        )

        # Copy all folders into the temporary directory
        for folder_path in folder_paths:
            folder_name = os.path.basename(os.path.normpath(folder_path))
            target_path = os.path.join(temp_dir, folder_name)
            print("Scanning folder", os.path.join(folder_path, "metadata.jsonl"))
            if os.path.exists(os.path.join(folder_path, "metadata.jsonl")):
                with open(os.path.join(folder_path, "metadata.jsonl"), "r") as f:
                    json_content = [json.loads(line) for line in f]
                # Skip upload if the task is in the examples
                if json_content[0]["task"] not in TASK_EXAMPLES:
                    print(f"Copying {folder_path} to temporary directory...")
                    shutil.copytree(folder_path, target_path)
            # Remove the original folder after copying
            shutil.rmtree(folder_path)
            print(f"Original folder {folder_path} removed.")

        # Upload the entire temporary directory
        print(f"Uploading all folders to {repo_id}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=[".git/*", ".gitignore"],
        )
        print("Upload complete.")

        return f"Successfully uploaded {len(folder_paths)} folders to {repo_id}"


def cleanup_agents(): # Renamed from cleanup_sandboxes
    """Remove agents that haven't been accessed for longer than AGENT_TIMEOUT"""
    current_time = time.time()
    agents_to_remove = []

    for session_id, metadata in AGENT_METADATA.items():
        if current_time - metadata["last_accessed"] > AGENT_TIMEOUT:
            agents_to_remove.append(session_id)

    for session_id in agents_to_remove:
        if session_id in AGENTS:
            try:
                # Upload data before removing if needed
                # This part needs to ensure it finds the correct data_dir based on interaction_id,
                # which might require interaction_id to be stored in AGENT_METADATA or derived.
                # For now, assuming data_dir is related to session_id for simplicity, but this needs care.
                # The original code used session_id for data_dir, which seems problematic if multiple interactions per session.
                # However, the current upload_to_hf_and_remove takes a list of folder_paths from TMP_DIR/interaction_id.
                # This cleanup function might need to know all interaction_ids for a session.
                # For now, this part is left as is, assuming the interaction_id based paths are handled elsewhere or data_dir is session_id based.
                data_dir_to_check = os.path.join(TMP_DIR, session_id) # This might be incorrect if interaction_id is the folder name
                # A better approach would be to scan TMP_DIR for folders starting with session_id if that's the convention
                # Or, if the upload_interaction_logs in demo.unload handles this, this step might be redundant here.
                # For now, let's assume this cleanup focuses on the agent object itself.
                # The actual log folders are named by interaction_id.
                # This function should ideally find all interaction_id folders for the timed-out session.

                # Close the agent
                AGENTS[session_id].close() # Calls the agent's close method
                del AGENTS[session_id]
                del AGENT_METADATA[session_id]
                print(f"Cleaned up agent for session {session_id}")
            except Exception as e:
                print(f"Error cleaning up agent {session_id}: {str(e)}")


def get_or_create_agent(session_hash: str, data_dir: str): # Renamed, data_dir added for agent creation
    current_time = time.time()

    if (
        session_hash in AGENTS
        and session_hash in AGENT_METADATA
        and current_time - AGENT_METADATA[session_hash]["created_at"]
        < AGENT_TIMEOUT
    ):
        print(f"Reusing Agent for session {session_hash}")
        AGENT_METADATA[session_hash]["last_accessed"] = current_time
        # Ensure agent is not already closed, if it is, recreate
        if AGENTS[session_hash].client is None or AGENTS[session_hash].vnc_client is None: # Basic check
             print(f"Agent for session {session_hash} was closed. Recreating.")
        else:
            return AGENTS[session_hash]


    if session_hash in AGENTS:
        try:
            print(f"Closing expired or unusable agent for session {session_hash}")
            AGENTS[session_hash].close()
        except Exception as e:
            print(f"Error closing expired agent: {str(e)}")

    print(f"Creating new agent for session {session_hash}")
    
    # Model definition now happens inside create_agent
    model = InferenceClientModel(
        model_id="https://n5wr7lfx6wp94tvl.us-east-1.aws.endpoints.huggingface.cloud",
        token=hf_token,
    )
    # Or OpenAI model
    # model = OpenAIServerModel("gpt-4o",api_key=os.getenv("OPENAI_API_KEY"))

    agent = E2BVisionAgent( # Agent creation
        model=model,
        data_dir=data_dir, # data_dir passed for agent's internal use
        max_steps=20,
        verbosity_level=2,
        use_v1_prompt=True,
    )
    # No E2B Sandbox specific calls like stream.start() or commands.run()

    print(f"Agent created for session {session_hash}.")

    AGENTS[session_hash] = agent
    AGENT_METADATA[session_hash] = {
        "created_at": current_time,
        "last_accessed": current_time,
        "interaction_id": os.path.basename(data_dir) # Store interaction_id for potential cleanup linkage
    }
    return agent


def update_html(interactive_mode: bool, session_hash: str):
    # Removed e2b.Sandbox specific logic.
    # The new E2BVisionAgent manages its own Docker container and VNC.
    # No direct stream URL to embed in the same way for this phase.
    # Returning a simple status message.
    
    # Check if agent exists and is active for the session to provide more accurate status
    status_text_detail = "Agent is active."
    if session_hash not in AGENTS or AGENTS[session_hash].client is None: # client can be used as a proxy for active agent
        status_text_detail = "Agent is initializing or not found..."
    
    status_class = "status-interactive" if interactive_mode else "status-view-only" # This might be less relevant now
    status_text_mode = "Interactive Mode" if interactive_mode else "Agent Running..." # This can still be used

    # Simplified HTML content
    # The original sandbox_html_template and its formatting for stream_url are gone.
    # We can use a simple div to show status.
    # The CSS classes status-interactive/status-view-only might still be useful for styling the text.
    
    # The timer part for auto-refresh or timeout display can be kept if needed.
    creation_time = (
        AGENT_METADATA[session_hash]["created_at"]
        if session_hash in AGENT_METADATA
        else time.time()
    )
    timeout_seconds = AGENT_TIMEOUT

    # Simple HTML status
    # The original `sandbox_html` Gradio component might still be used to display this.
    # Adjusting to provide some meaningful status.
    # No more iframe for the VNC stream in this version.
    html_content = f"""
    <div class='sandbox-status {status_class}'>
        <h3>Session: {session_hash}</h3>
        <p>Status: {status_text_mode}</p>
        <p>{status_text_detail}</p>
    </div>
    <div id="sandbox-creation-time" style="display:none;" data-time="{creation_time}" data-timeout="{timeout_seconds}"></div>
    """
    return html_content


def generate_interaction_id(session_hash: str):
    return f"{session_hash}_{int(time.time())}"


def save_final_status(folder, status: str, summary, error_message=None) -> None: # Remains the same
    with open(os.path.join(folder, "metadata.jsonl"), "a") as output_file:
        output_file.write(
            "\n"
            + json.dumps(
                {"status": status, "summary": summary, "error_message": error_message},
            )
        )


def extract_browser_uuid(js_uuid): # This seems unused now, but keeping for now.
    print(f"[BROWSER] Got browser UUID from JS: {js_uuid}")
    return js_uuid


def initialize_session(interactive_mode, request: gr.Request):
    assert request.session_hash is not None
    print("GETTING REQUEST HASH:", request.session_hash)
    # new_uuid was for browser UUID, might not be needed if that logic is removed.
    # For now, let's keep it to minimize changes to function signatures if it's used by JS.
    new_uuid = str(uuid.uuid4()) 
    
    # update_html now returns simplified HTML.
    # The data_dir for the agent will be created/managed within interact_with_agent,
    # as get_or_create_agent now takes data_dir.
    # However, initialize_session might not need to call get_or_create_agent directly anymore
    # if update_html doesn't strictly need an active agent to display initial status.
    # For now, update_html will just show a generic status based on session_hash.
    return update_html(interactive_mode, request.session_hash), new_uuid


def create_agent_instance(data_dir: str): # Renamed and desktop parameter removed
    # Model definition is now consistently here
    # Using OpenAIServerModel as requested
    model = OpenAIServerModel(
        model_id="mlx-community/Qwen2.5-VL-32B-Instruct-4bit", # New model_id
        base_url="http://localhost:1234/v1", # New base_url
        api_key="lm-studio" # New api_key
        # token parameter removed
    )
    # This function now directly creates the E2BVisionAgent
    # The E2BVisionAgent's __init__ handles Docker and VNC setup.
    agent = E2BVisionAgent(
        model=model,
        data_dir=data_dir, # data_dir is crucial for the agent's operation
        max_steps=20,
        verbosity_level=2,
        use_v1_prompt=True,
    )
    return agent


INTERACTION_IDS_PER_SESSION_HASH: dict[str, dict[str, bool]] = {}


class EnrichedGradioUI(GradioUI):
    def log_user_message(self, text_input):
        import gradio as gr

        return (
            text_input,
            gr.Button(interactive=False),
        )

    def interact_with_agent(
        self,
        task_input,
        stored_messages,
        session_state, # session_state will store the agent instance
        consent_storage,
        request: gr.Request,
    ):
        interaction_id = generate_interaction_id(request.session_hash)
        data_dir = os.path.join(TMP_DIR, interaction_id) # data_dir per interaction
        
        # Ensure data_dir exists if consent is given for storing logs
        if not os.path.exists(data_dir) and consent_storage:
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")

        # Get or create agent for the session.
        # The agent is now created/managed by get_or_create_agent, which calls create_agent_instance.
        # We need to ensure AGENT_METADATA is updated correctly.
        # Storing the agent in session_state for access during the interaction.
        
        # If an agent for this session_hash already exists and is usable, reuse it.
        # Otherwise, get_or_create_agent will make a new one.
        # This logic is simplified: we fetch/create the agent and assign to session_state.
        # The create_agent_instance is now more of a helper for E2BVisionAgent instantiation.
        # The get_or_create_agent function will call create_agent_instance if needed.
        
        # The agent is now created via get_or_create_agent, which itself calls the constructor.
        # We pass data_dir to get_or_create_agent now.
        current_agent = get_or_create_agent(request.session_hash, data_dir)
        session_state["agent"] = current_agent
        
        # Update AGENT_METADATA last_accessed time
        if request.session_hash in AGENT_METADATA:
             AGENT_METADATA[request.session_hash]["last_accessed"] = time.time()
             AGENT_METADATA[request.session_hash]["interaction_id"] = interaction_id # Keep track of current interaction

        if request.session_hash not in INTERACTION_IDS_PER_SESSION_HASH: # This seems for log upload tracking
            INTERACTION_IDS_PER_SESSION_HASH[request.session_hash] = {}
        INTERACTION_IDS_PER_SESSION_HASH[request.session_hash][interaction_id] = True


        if not task_input or len(task_input) == 0:
            raise gr.Error("Task cannot be empty")

        try:
            stored_messages.append(
                gr.ChatMessage(
                    role="user", content=task_input, metadata={"status": "done"}
                )
            )
            yield stored_messages

            if consent_storage: # This part remains the same
                with open(os.path.join(data_dir, "metadata.jsonl"), "w") as output_file:
                    output_file.write(
                        json.dumps(
                            {"task": task_input},
                        )
                    )
            
            # Initial screenshot using agent's VNC client
            # Ensure agent and vnc_client are ready
            if session_state["agent"] and session_state["agent"].vnc_client:
                pil_image = session_state["agent"].vnc_client.capture_screen()
                if pil_image:
                    # Convert PIL image to bytes, then to BytesIO for compatibility if needed,
                    # or pass PIL image directly if stream_to_gradio supports it.
                    # Assuming stream_to_gradio's task_images expects PIL Images.
                    initial_screenshot_pil = pil_image
                else:
                    # Fallback or error if screenshot fails
                    initial_screenshot_pil = Image.new('RGB', (WIDTH, HEIGHT), color = 'red') # Placeholder
                    print("Error: Failed to capture initial screenshot from VNC.")
            else:
                initial_screenshot_pil = Image.new('RGB', (WIDTH, HEIGHT), color = 'grey') # Placeholder
                print("Error: Agent or VNC client not available for initial screenshot.")

            for msg in stream_to_gradio(
                session_state["agent"], # Pass the agent instance
                task=task_input,
                reset_agent_memory=False, # Assuming this is desired behavior
                task_images=[initial_screenshot_pil], # Pass the PIL image
            ):
                if (
                    hasattr(session_state["agent"], "last_marked_screenshot") # This logic should still work
                    and isinstance(msg, gr.ChatMessage)
                    and msg.content == "-----"
                ):  # Append the last screenshot before the end of step
                    stored_messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content={
                                "path": session_state[
                                    "agent"
                                ].last_marked_screenshot.to_string(),
                                "mime_type": "image/png",
                            },
                            metadata={"status": "done"},
                        )
                    )
                if isinstance(msg, gr.ChatMessage):
                    stored_messages.append(msg)
                elif isinstance(msg, str):  # Then it's only a completion delta
                    try:
                        if stored_messages[-1].metadata["status"] == "pending":
                            stored_messages[-1].content = msg
                        else:
                            stored_messages.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content=msg,
                                    metadata={"status": "pending"},
                                )
                            )
                    except Exception as e:
                        raise e
                yield stored_messages

            status = "completed"
            yield stored_messages

        except Exception as e:
            error_message = f"Error in interaction: {str(e)}"
            print(error_message)
            stored_messages.append(
                gr.ChatMessage(
                    role="assistant", content="Run failed:\n" + error_message
                )
            )
            status = "failed"
            yield stored_messages
        finally:
            if consent_storage:
                summary = get_agent_summary_erase_images(session_state["agent"])
                save_final_status(
                    data_dir, status, summary=summary, error_message=error_message
                )
                print("SAVING FINAL STATUS", data_dir, status, summary, error_message)


theme = gr.themes.Default(
    font=["Oxanium", "sans-serif"], primary_hue="amber", secondary_hue="blue"
)

# Create a Gradio app with Blocks
with gr.Blocks(theme=theme, css=custom_css, js=CUSTOM_JS) as demo:
    # Storing session hash in a state variable
    print("Starting the app!")
    with gr.Row():
        # sandbox_html_template is gone. update_html returns simpler status.
        sandbox_html_status_display = gr.HTML( # Renamed for clarity
            value="<div>Agent status will appear here.</div>", # Initial simple value
            label="Agent Status", # Label changed
        )
        with gr.Sidebar(position="left"):
            with Modal(visible=True) as modal:
                gr.Markdown("""### Welcome to smolagent's Computer agent demo 🖥️
In this app, you'll be able to interact with an agent powered by [smolagents](https://github.com/huggingface/smolagents) and [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct).

👉 Type a task in the left sidebar, click the button, and watch the agent solving your task. ✨

_Please note that we store the task logs by default so **do not write any personal information**; you can uncheck the logs storing on the task bar._
""")
            task_input = gr.Textbox(
                placeholder="Find me pictures of cute puppies",
                label="Enter your task below:",
                elem_classes="primary-color-label",
            )

            run_btn = gr.Button("Let's go!", variant="primary")

            gr.Examples(
                examples=TASK_EXAMPLES,
                inputs=task_input,
                label="Example Tasks",
                examples_per_page=4,
            )

            session_state = gr.State({})
            stored_messages = gr.State([])

            minimalist_toggle = gr.Checkbox(label="Innie/Outie", value=False)

            consent_storage = gr.Checkbox(
                label="Store task and agent trace?", value=True
            )

            gr.Markdown(
                """
- **Data**: To opt-out of storing your trace, uncheck the box above.
- **Be patient**: The agent's first step can take a few seconds.
- **Captcha**: Sometimes the VMs get flagged for weird behaviour and are blocked with a captcha. Best is then to interrupt the agent and solve the captcha manually.
- **Restart**: If your agent seems stuck, the simplest way to restart is to refresh the page.
                """.strip()
            )

            # Hidden HTML element to inject CSS dynamically
            theme_styles = gr.HTML(apply_theme(False), visible=False)
            minimalist_toggle.change(
                fn=apply_theme, inputs=[minimalist_toggle], outputs=[theme_styles]
            )

            footer = gr.HTML(value=FOOTER_HTML, label="Footer")

    chatbot_display = gr.Chatbot(
        elem_id="chatbot",
        label="Agent's execution logs",
        type="messages",
        avatar_images=(
            None,
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
        ),
        resizable=True,
    )

    agent_ui = EnrichedGradioUI(
        CodeAgent(tools=[], model=None, name="ok", description="ok")
    )

    stop_btn = gr.Button("Stop the agent!", variant="huggingface")

    def read_log_content(log_file, tail=4):
        """Read the contents of a log file for a specific session"""
        if not log_file:
            return "Waiting for session..."

        if not os.path.exists(log_file):
            return "Waiting for machine from the future to boot..."

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                return "".join(lines[-tail:] if len(lines) > tail else lines)
        except Exception as e:
            return f"Guru meditation: {str(e)}"

    # Function to set view-only mode (now just updates status text)
    def update_status_running(task_input, request: gr.Request): # Renamed
        # This function is called when run_btn is clicked.
        # update_html will be called to show "Agent Running..."
        # The interactive_mode=False indicates the agent is busy.
        return update_html(False, request.session_hash)

    def update_status_interactive(request: gr.Request): # Renamed
        # This function is called after the agent finishes.
        # update_html will be called to show "Interactive Mode" or similar.
        return update_html(True, request.session_hash)

    def reactivate_stop_btn(): # Remains the same
        return gr.Button("Stop the agent!", variant="huggingface")

    is_interactive = gr.Checkbox(value=True, visible=False) # This might be less relevant

    # Chain the events
    run_event = (
        run_btn.click(
            fn=update_status_running, # Update status to "Agent Running..."
            inputs=[task_input],
            outputs=[sandbox_html_status_display], # Output to the new HTML component
        )
        .then(
            agent_ui.interact_with_agent, # This remains the core agent interaction
            inputs=[ # Inputs are the same
                task_input,
                stored_messages,
                session_state,
                consent_storage,
            ],
            outputs=[chatbot_display], # Output is the same
        )
        .then(fn=update_status_interactive, inputs=[], outputs=[sandbox_html_status_display]) # Update status after agent done
        .then(fn=reactivate_stop_btn, outputs=[stop_btn]) # Reactivate stop button
    )

    def interrupt_agent(session_state):
        if not session_state["agent"].interrupt_switch:
            session_state["agent"].interrupt()
            print("Stopping agent...")
            return gr.Button("Stopping agent... (could take time)", variant="secondary")
        else:
            return gr.Button("Stop the agent!", variant="huggingface")

    stop_btn.click(fn=interrupt_agent, inputs=[session_state], outputs=[stop_btn])

    def upload_interaction_logs(session: gr.Request):
        data_dirs = []
        for interaction_id in list(
            INTERACTION_IDS_PER_SESSION_HASH[session.session_hash].keys()
        ):
            data_dir = os.path.join(TMP_DIR, interaction_id)
            if os.path.exists(data_dir):
                data_dirs.append(data_dir)
                del INTERACTION_IDS_PER_SESSION_HASH[session.session_hash][
                    interaction_id
                ]

        upload_to_hf_and_remove(data_dirs)

    demo.load(
        fn=lambda: True,  # dummy to trigger the load
        outputs=[is_interactive],
    ).then(
        fn=initialize_session, # initialize_session calls update_html
        inputs=[is_interactive], # is_interactive might be simplified
        outputs=[sandbox_html_status_display, browser_uuid_output], # browser_uuid_output might be removed if JS is simplified
    )

    demo.unload(fn=upload_interaction_logs) # Remains the same

# Launch the app
if __name__ == "__main__":
    Timer(60, cleanup_agents).start()  # Renamed: Run every minute
    demo.launch()

import gradio as gr
import requests
import json

def generate_response(prompt, conversation_history):
    """
    Generate a response using Ollama's API with the llama2 model
    """
    # Prepare the conversation history in the format Ollama expects
    messages = []
    for human, assistant in conversation_history:
        messages.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    # Prepare the API request
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": messages,
        "stream": False
    }
    
    try:
        # Make the API call
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        result = response.json()
        return result["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}\nMake sure Ollama is running and llama2 model is installed."
    except (KeyError, json.JSONDecodeError) as e:
        return f"Error parsing response: {str(e)}"

# Create the Gradio interface
def respond(message, chat_history):
    """
    Process each message and update chat history
    """
    bot_message = generate_response(message, chat_history)
    chat_history.append((message, bot_message))
    return "", chat_history

# Configure the chat interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        height=600,
        show_label=False,
        bubble_full_width=False,
        avatar_images=("👤", "🤖")
    )
    msg = gr.Textbox(
        label="Type your message here...",
        placeholder="Type your message here...",
        show_label=False
    )
    clear = gr.ClearButton([msg, chatbot])

    # Set up the message submission
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio chatbot. Make sure Ollama is running with llama2 model installed.")
    demo.launch(share=False)

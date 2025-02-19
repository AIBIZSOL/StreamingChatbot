import gradio as gr
import requests
import json

def generate_streaming_response(prompt, conversation_history):
    messages = []
    for human, assistant in conversation_history:
        messages.extend([
            {"role": "user", "content": human},
            {"role": "assistant", "content": assistant}
        ])
    
    messages.append({"role": "user", "content": prompt})
    
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": messages,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'message' in json_response and 'content' in json_response['message']:
                    yield json_response['message']['content']
                
    except requests.exceptions.RequestException as e:
        yield f"Error: {str(e)}\nMake sure Ollama is running and llama3.1 model is installed."

def respond(message, chat_history):
    chat_history = chat_history + [(message, "")]
    yield "", chat_history
    
    current_response = ""
    for chunk in generate_streaming_response(message, chat_history[:-1]):
        current_response += chunk
        chat_history[-1] = (message, current_response)
        yield "", chat_history

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        height=600,
        show_label=False,
        bubble_full_width=False,
        avatar_images=(
            "user_image.png",  # Replace with your user image path
            "bot_image.png"    # Replace with your bot image path
        )
    )
    msg = gr.Textbox(
        label="Type your message here...",
        placeholder="Type your message here...",
        show_label=False
    )
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        lambda: gr.update(interactive=True), 
        None, 
        [msg]
    )

if __name__ == "__main__":
    print("Starting Gradio chatbot with streaming...")
    demo.launch(share=False)

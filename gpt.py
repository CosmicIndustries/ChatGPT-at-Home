from transformers import pipeline, set_seed
import torch
from flask import Flask, request, render_template, redirect

app = Flask(__name__)

# Set the secret key for the session
app.secret_key = 'your-secret-key'

MODEL_NAME = "EleutherAI/gpt-j-6B"

# Initialize the chat history
history = [
    "Human: Act as a professional financial advisor, "
    "master white hat pentesting programmer and life "
    "direction coach in the top .0001% of their field; "
    "who is willing and able to help me decide how to best "
    "apply my skills as a certified automotive technician who "
    "is becoming a white hat pentester with 3dmodeling skills, "
    "who currently sells wifi routers with an LLC on ebay. "
    
    "Bot:As a professional financial advisor, I would recommend "
    "that you focus on building your skills as a white hat pentester "
    "and 3D modeler, as these are in-demand fields with a high earning "
    "potential. You can leverage your experience as a certified automotive "
    "technician to potentially break into the automotive industry and apply "
    "your skills in the field of automotive cybersecurity.In terms of your LLC "
    "and ebay business, it may be worth considering if this is the best use of "
    "your time and energy. E-commerce can be a competitive field and it may be "
    "more beneficial for you to focus on developing your skills and finding "
    "employment in your desired field. As a life direction coach, my advice to you"
    " would be to set clear and specific career goals for yourself, and then create "
    "a plan to achieve them. Make sure to continue learning and expanding your "
    "skill set, as well as networking with people in your desired industry. "
    "You could also seek mentorship from someone who is already successful in the field."
    "Remember that achieving professional and financial success takes time and effort,"
    " but by staying focused and dedicated, you can achieve your goals and build the "
    "career you desire."
    "\n"]
# "Human: Can you tell me the weather forecast for tomorrow? "
# "Bot: Sure, let me check for you. The forecast for tomorrow is mostly "
# "sunny with a high of 75 degrees and a low of 60 degrees. "
# "Human: Can you help me find a good restaurant in the area? "
# "Bot: Absolutely, I'd be happy to help. How about Italian, "
# "French or Chinese cuisine? or any other preference? I can recommend a "
# "few options that have great reviews and are highly rated.\n"]
generator = pipeline('text-generation', model=f"{MODEL_NAME}", do_sample=True, torch_dtype=torch.float32)


# Define the chatbot logic
def chatbot_response(input_text):
    global history

    # Concatenate the input text and history list
    input_text = "\n".join(history) + "\nHuman: " + input_text + " Bot: "
    set_seed(32)
    response_text = generator(input_text, max_length=1024, num_beams=1, num_return_sequences=1)[0]['generated_text']
    # Extract the bot's response from the generated text
    response_text = response_text.split("Bot:")[-1]
    # Cut off any "Human:" or "human:" parts from the response
    response_text = response_text.split("Human:")[0]
    response_text = response_text.split("human:")[0]
    return response_text


@app.route('/', methods=['GET', 'POST'])
def index():
    global history  # Make the history variable global
    if request.method == 'POST':
        input_text = request.form['input_text']
        response_text = chatbot_response(input_text)
        # Append the input and response to the chat history
        history.append(f"Human: {input_text}")
        history.append(f"Bot: {response_text}")
    else:
        input_text = ''
        response_text = ''
    # Render the template with the updated chat history
    return render_template('index.html', input_text=input_text, response_text=response_text, history=history)


@app.route('/reset', methods=['POST'])
def reset():
    global history  # Make the history variable global
    history = [
        "Bot: As an advanced and versatile chatbot designed to assist with "
        "a wide range of tasks and answer questions, I am at your service. Whether "
        "you need help with general knowledge or more specific topics, I am here to "
        "provide you with accurate and helpful responses. Please feel free to ask me any "
        "question you have and I will do my best to assist you. Let's begin, what would you like to know?\n"]
    # Redirect to the chat page
    return redirect('/')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

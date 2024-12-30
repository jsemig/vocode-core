import random

from fastapi import FastAPI

from vocode.streaming.agent.onsai_agent import OnsaiGPTOutput

app = FastAPI()

# eine Beispieldatei zur Simulation von Onsia-Antworten mit SSML

@app.get("/")
def read_root():
    return {"Hello": "World"}


rand_texts = [
    "I am a bot.",
    "How can I assist you today?",
    "Please provide more details.",
    "Thank you for your patience.",
    "I am here to help.",
    "Can you please clarify?",
    "Let's solve this together.",
    "What else can I do for you?",
    "I appreciate your input.",
    "Feel free to ask me anything.",
    "I am listening.",
    "You can park in the Garage for free. How else may I help you?",
]


@app.get("/completion")
@app.post("/completion")
def completion():
    rand_text = random.choice(rand_texts)
    ssml = f"""<speak xmlns="https://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" version="1.0" xml:lang="en-US"><voice name="en-US-SteffanNeural"><mstts:silence value="500ms" type="Tailing-exact" /><prosody pitch="0%" rate="15%">{rand_text}</prosody></voice></speak>"""
    response = OnsaiGPTOutput(
        bot_response=ssml,
        end_conversation=False,
        conversation_id="f38e26dd-16d4-4ada-a77e-e09ad9b1a2e6"
    )
    print("response", response)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
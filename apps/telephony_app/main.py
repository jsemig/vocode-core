# Standard library imports
import os

from dotenv import load_dotenv

# Third-party imports
from fastapi import FastAPI

# Local application/library specific imports
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.agent import OnsaiGPTAgentConfig
from vocode.streaming.models.message import SSMLMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import AzureTranscriberConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.telephony.server.base import (
    TelephonyServer,
    TwilioInboundCallConfig,
)

# if running from python, this will load the local .env
# docker-compose will load the .env file by itself
load_dotenv()

configure_pretty_logging()


app = FastAPI(docs_url=None)

config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment if not using pyngrok")

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/twilio/inbound_call",
            agent_config=OnsaiGPTAgentConfig(
                base_url=os.getenv("ONSAI_BASE_URL"),
                send_raw_ssml=True,
                allowed_idle_time_seconds=30,
                goodbye_phrases=[
                    "Verabschiedung",
                    "Tschüss",
                    "Auf Wiedersehen",
                    "bye" "goodbye",
                ],
                initial_message=SSMLMessage(
                    ssml="<speak xmlns='https://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' version='1.0' xml:lang='de-DE'><voice name='de-DE-ConradNeural'><mstts:silence value='500ms' type='Tailing-exact' /><prosody pitch='0%' rate='15%'>Danke für Ihren Anruf. Wie kann ich Ihnen helfen?</prosody></voice></speak>",
                    text="Hello, how can I help you?",
                ),
            ),
            synthesizer_config=AzureSynthesizerConfig.from_telephone_output_device(
                azure_speech_key=os.environ["AZURE_SPEECH_KEY"],
                azure_speech_region=os.environ["AZURE_SPEECH_REGION"],
                language="en-US",  # de-DE
            ),
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
            transcriber_config=AzureTranscriberConfig.from_telephone_input_device(
                azure_speech_key=os.environ["AZURE_SPEECH_KEY"],
                azure_speech_region=os.environ["AZURE_SPEECH_REGION"],
                language="en-US",  # de-DE
            ),
        )
    ],
)

app.include_router(telephony_server.get_router())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1337)

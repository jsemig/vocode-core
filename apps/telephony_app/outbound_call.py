import os

from dotenv import load_dotenv

from vocode.streaming.models.agent import OnsaiGPTAgentConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import AzureTranscriberConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall

load_dotenv()


BASE_URL = os.environ["BASE_URL"]


async def main():
    config_manager = RedisConfigManager()

    outbound_call = OutboundCall(
        base_url=BASE_URL,
        to_phone=os.environ["TWILIO_TO_NUMBER"],
        from_phone=os.environ["TWILIO_FROM_NUMBER"],
        config_manager=config_manager,
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
        ),
        synthesizer_config=AzureSynthesizerConfig.from_telephone_output_device(
            language="en-US",  # de-DE
        ),
        transcriber_config=AzureTranscriberConfig.from_telephone_input_device(
            language="en-US",  # de-DE
        ),
        telephony_config=TwilioConfig(
            account_sid=os.environ["TWILIO_ACCOUNT_SID"],
            auth_token=os.environ["TWILIO_AUTH_TOKEN"],
        ),
    )

    print(outbound_call.telephony_config, "sid", os.getenv("TWILIO_ACCOUNT_SID"))
    input("Press enter to start call...")
    await outbound_call.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

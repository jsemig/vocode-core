import asyncio
import signal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.onsai_agent import OnsaiGPTAgent
from vocode.streaming.models.agent import OnsaiGPTAgentConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

configure_pretty_logging()

load_dotenv()

class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """

    openai_api_key: str = "ENTER_YOUR_OPENAI_API_KEY_HERE"
    azure_speech_key: str = "ENTER_YOUR_AZURE_KEY_HERE"
    deepgram_api_key: str = "ENTER_YOUR_DEEPGRAM_API_KEY_HERE"

    azure_speech_region: str = "eastus"

    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True,
    )

    conversation = StreamingConversation(
        output_device=speaker_output, # type: ignore
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input, # type: ignore
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ), # type: ignore
        agent=OnsaiGPTAgent(
            OnsaiGPTAgentConfig(
                base_url="http://localhost:8000",
                send_raw_ssml=True,
                allowed_idle_time_seconds=30,
                goodbye_phrases=["Verabschiedung", "Tschüss", "Auf Wiedersehen", "bye" "goodbye"],
            )
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output),
            azure_speech_key=settings.azure_speech_key,
            azure_speech_region=settings.azure_speech_region,
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())

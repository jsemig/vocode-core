from typing import AsyncGenerator, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import (
    AgentInput,
    GeneratedResponse,
    RespondAgent,
    StreamedResponse,
)
from vocode.streaming.models.agent import OnsaiGPTAgentConfig
from vocode.streaming.models.message import BaseMessage, SSMLMessage
from vocode.streaming.models.transcriber import Transcription
from vocode.streaming.utils.state_manager import TwilioPhoneConversationStateManager
from vocode.streaming.vector_db.factory import VectorDBFactory


class OnsaiGPTOutput(BaseModel):
    bot_response: str = Field(
        ...,
        description="<speak>You can park in the Garage for free. How else may I help you?</speak>",
    )
    end_conversation: bool = Field(
        False, description="True if the conversation should end"
    )
    conversation_id: str = Field(
        ..., description="the conversations id, f38e26dd-16d4-4ada-a77e-e09ad9b1a2e6"
    )


class OnsaiGPTAgentInput(BaseModel):
    user_input: str = Field(..., description="Where can I park?")
    phone_number: str | None = Field(None, description="+4917643806827")
    conversation_id: str = Field(
        ..., description="the conversations id, f38e26dd-16d4-4ada-a77e-e09ad9b1a2e6"
    )


OnsaiGPTAgentConfigType = TypeVar("OnsaiGPTAgentConfigType", bound=OnsaiGPTAgentConfig)


class OnsaiGPTAgent(RespondAgent[OnsaiGPTAgentConfigType]):
    def __init__(
        self,
        agent_config: OnsaiGPTAgentConfigType,
        action_factory: AbstractActionFactory = DefaultActionFactory(),
        vector_db_factory=VectorDBFactory(),
        **kwargs,
    ):
        super().__init__(
            agent_config=agent_config,
            action_factory=action_factory,
            **kwargs,
        )
        self.onsai_client = httpx.AsyncClient(
            base_url=str(agent_config.base_url),
        )
        logger.info(f"Initialized OnsaiGPTAgent with config: {agent_config}")

    async def token_generator(
        self,
        chunk: OnsaiGPTOutput,
    ) -> AsyncGenerator[str, None]:
        yield chunk.bot_response

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
        bot_was_in_medias_res: bool = False,
    ) -> AsyncGenerator[GeneratedResponse, None]:
        assert self.transcript is not None

        onsai_payload = OnsaiGPTAgentInput(
            user_input=human_input,
            phone_number=None,
            conversation_id=conversation_id,
        )

        if not isinstance(
            self.conversation_state_manager, TwilioPhoneConversationStateManager
        ):
            logger.error(
                "OnsaiGPTAgent requires TwilioPhoneConversationStateManager, ignoring request to use telephony features..."
            )
        else:
            if not self.conversation_state_manager.get_to_phone():
                logger.error(
                    "OnsaiGPTAgent requires a to phone number, ignoring request to use phone number"
                )
            else:
                onsai_payload.phone_number = (
                    self.conversation_state_manager.get_to_phone()
                )

        logger.info(f"Sending request to OnsaiGPT: {onsai_payload.model_dump_json()}")
        response = await self.onsai_client.post(
            url="/completion",
            json=onsai_payload.model_dump(),
        )
        response.raise_for_status()
        json = response.json()
        onsai_response = OnsaiGPTOutput.model_validate(json)

        using_input_streaming_synthesizer = (
            self.conversation_state_manager.using_input_streaming_synthesizer()
        )

        ResponseClass = (
            StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
        )

        message: SSMLMessage | BaseMessage = (
            SSMLMessage(
                ssml=onsai_response.bot_response, text=onsai_response.bot_response
            )
            if self.agent_config.send_raw_ssml
            else BaseMessage(text=onsai_response.bot_response)
        )

        yield ResponseClass(
            message=message,
            is_interruptible=True,
        )

        if onsai_response.end_conversation:
            # TODO: wait for conversation state manager to finish
            # await self.terminate()
            pass



    async def handle_generate_response(
        self,
        transcription: Transcription,
        agent_input: AgentInput,
    ) -> bool:
        return True


    async def terminate(self):
        await self.conversation_state_manager.terminate_conversation()
        return await super().terminate()

from typing import AsyncGenerator, TypeVar

import httpx
import sentry_sdk
from loguru import logger
from pydantic import BaseModel, Field

from vocode.streaming.action.abstract_factory import AbstractActionFactory
from vocode.streaming.action.default_factory import DefaultActionFactory
from vocode.streaming.agent.base_agent import (
    AgentResponseMessage,
    GeneratedResponse,
    RespondAgent,
    StreamedResponse,
)
from vocode.streaming.agent.streaming_utils import (
    collate_response_async,
    stream_response_async,
)
from vocode.streaming.models.actions import EndOfTurn
from vocode.streaming.models.agent import OnsaiGPTAgentConfig
from vocode.streaming.models.message import BaseMessage, LLMToken
from vocode.streaming.utils.state_manager import TwilioPhoneConversationStateManager
from vocode.streaming.vector_db.factory import VectorDBFactory
from vocode.utils.sentry_utils import CustomSentrySpans, sentry_create_span


class OnsaiGPTOutput(BaseModel):
    bot_response: str = Field(..., description="<speak>You can park in the Garage for free. How else may I help you?</speak>")
    end_conversation: bool = Field(False, description="True if the conversation should end")
    conversation_id: str = Field(..., description="the conversations id, f38e26dd-16d4-4ada-a77e-e09ad9b1a2e6")


class OnsaiGPTAgentInput(BaseModel):
    user_input: str = Field(..., description="Where can I park?")
    phone_number: str|None = Field(None, description="+4917643806827")
    conversation_id: str = Field(..., description="the conversations id, f38e26dd-16d4-4ada-a77e-e09ad9b1a2e6")


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

        if not isinstance(self.conversation_state_manager, TwilioPhoneConversationStateManager):
             logger.error("OnsaiGPTAgent requires TwilioPhoneConversationStateManager, ignoring request to use telephony features...")
        else:
            if not self.conversation_state_manager.get_to_phone():
                logger.error("OnsaiGPTAgent requires a to phone number, ignoring request to use phone number")
            else:
                onsai_payload.phone_number = self.conversation_state_manager.get_to_phone()


        first_sentence_total_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.LLM_FIRST_SENTENCE_TOTAL
        )

        ttft_span = sentry_create_span(
            sentry_callable=sentry_sdk.start_span, op=CustomSentrySpans.TIME_TO_FIRST_TOKEN
        )

        logger.info(f"Sending request to Onsai GPT: {onsai_payload.model_dump_json()}")
        response = await self.onsai_client.post(
            url="/completion",
            json=onsai_payload.model_dump(),
        )
        response.raise_for_status()
        json = await response.json()
        onsai_response = OnsaiGPTOutput.model_validate(json)

        response_generator = collate_response_async
        using_input_streaming_synthesizer = (
            self.conversation_state_manager.using_input_streaming_synthesizer()
        )

        if using_input_streaming_synthesizer:
            response_generator = stream_response_async

        # TODO: we can yield the `bot_response` directly if we want to stream the response
        # otherwise we can yield the `GeneratedResponse` object which will collate and yield the bot_response
        async for message in response_generator(
            conversation_id=conversation_id,
            gen=self.token_generator(
                onsai_response,
            ),
            get_functions=False,
            sentry_span=ttft_span,
        ):
            if first_sentence_total_span:
                first_sentence_total_span.finish()

            ResponseClass = (
                StreamedResponse if using_input_streaming_synthesizer else GeneratedResponse
            )
            MessageType = LLMToken if using_input_streaming_synthesizer else BaseMessage
            if isinstance(message, str):
                yield ResponseClass(
                    message=MessageType(text=message),
                    is_interruptible=True,
                )
            else:
                yield ResponseClass(
                    message=message,
                    is_interruptible=True,
                )

        # produce speech before teminating the conversation
        if onsai_response.end_conversation:
            self.agent_responses_consumer.consume_nonblocking(
                self.interruptible_event_factory.create_interruptible_agent_response_event(
                    AgentResponseMessage(message=EndOfTurn()),
                    is_interruptible=False,
                )
            )
            await self.conversation_state_manager.terminate_conversation()


    async def terminate(self):
        return await super().terminate()


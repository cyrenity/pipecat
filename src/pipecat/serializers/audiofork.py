#
# Copyright (c) 2024–2025, Mustafa
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Serializer for handling raw audio frames from mod_audio_fork.

This module provides functions to encode and decode audio data
transmitted by FreeSWITCH's mod_audio_fork over WebSocket, typically
in little-endian 16-bit PCM format. It can also handle conversion to
and from base64 or JSON payloads when needed.
"""

import base64
import dataclasses
import json
from typing import Any, Dict, Optional, Union

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    StartFrame,
    TextFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class AudioForkSerializer(FrameSerializer):
    """Serializer compatible with mod_audio_fork FreeSWITCH module.

    This serializer converts between Pipecat frames and the format expected
    by the mod_audio_fork module, which uses JSON text frames and raw binary
    audio frames.
    """

    class InputParams(BaseModel):
        """Configuration parameters for AudioForkSerializer.

        Parameters:
            freeswitch_sample_rate: Sample rate used by FreeSWITCH, defaults to 16000 Hz.
            sample_rate: Optional override for pipeline input sample rate.
        """

        freeswitch_sample_rate: int = 16000
        sample_rate: Optional[int] = None

    def __init__(self, call_uuid: str, stream_id: str, params: Optional[InputParams] = None):
        """Initialize the AudioForkSerializer.

        Args:
            call_uuid: The associated FreeSWITCH Call UUID (optional, not used in this implementation).
            stream_id: The associated FreeSWITCH Media Stream ID (optional, not used in this implementation).
            params: Configuration parameters.
        """
        self.call_uuid = call_uuid
        self.stream_id = stream_id
        self._params = params or AudioForkSerializer.InputParams()

        self._freeswitch_sample_rate = self._params.freeswitch_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()

    @property
    def type(self) -> FrameSerializerType:
        """Get the serializer type.

        Returns:
            Mixed type as we handle both text and binary frames.
        """
        return FrameSerializerType.BINARY

    async def setup(self, frame: StartFrame):
        """Sets up the serializer with pipeline configuration.

        Args:
            frame: The StartFrame containing pipeline configuration.
        """
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> Union[str, bytes, None]:
        """Serialize a frame to audio_fork compatible format.

        Args:
            frame: The frame to serialize.

        Returns:
            Serialized frame as string (for JSON) or bytes (for audio),
            or None if frame type is not serializable.
        """
        # Handle text frames

        if isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "call_uuid": self.call_uuid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: FreeSWITCH outputs PCM audio, but we need to resample to match requested sample_rate
            serialized_data = await self._output_resampler.resample(
                data, frame.sample_rate, self._freeswitch_sample_rate
            )
            if serialized_data is None or len(serialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            return serialized_data

        elif isinstance(frame, TextFrame):
            return json.dumps({"type": "text", "data": frame.text})
        elif isinstance(frame, (EndFrame, CancelFrame)):
            return json.dumps({"type": "disconnect"})

        # logger.warning(f"Frame type {type(frame)} is not serializable")
        return None

    async def deserialize(self, data: Union[str, bytes]) -> Optional[Frame]:
        """Deserialize mod_audio_fork data to a frame.

        Args:
            data: String (JSON) or bytes (audio) to deserialize.

        Returns:
            Deserialized frame instance, or None if deserialization fails.
        """
        # Handle binary data (audio)
        if isinstance(data, bytes):
            # Add the required sample_rate and num_channels parameters
            deserialized_data = await self._input_resampler.resample(
                data,
                self._freeswitch_sample_rate,
                self._sample_rate,
            )
            if deserialized_data is None or len(deserialized_data) == 0:
                # Ignoring in case we don't have audio
                return None

            # Input: FreeSWITCH takes PCM data, so just resample to match sample_rate
            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1,  # Assuming mono audio from FreeSWITCH
                sample_rate=self._sample_rate,  # Use the configured pipeline input rate
            )
            return audio_frame

        # Handle text data (JSON messages)
        try:
            # Try to parse as JSON
            json_data = json.loads(data)

            # Check if it's a structured message with type
            if isinstance(json_data, dict) and "type" in json_data:
                msg_type = json_data["type"]

                # Handle metadata message
                if msg_type == "dtmf":
                    digit = json_data["digit"]
                    try:
                        # Convert string to enum value
                        return InputDTMFFrame(KeypadEntry(digit))
                    except ValueError:
                        # Handle case where string doesn't match any enum value
                        logger.info(f"Invalid DTMF digit: {digit}")
                        return None
                # Handle other message types
                return InputTransportMessageFrame(message=json_data)

            # Simple text message
            return TextFrame(text=data)

        except json.JSONDecodeError:
            # Not JSON, treat as plain text
            return TextFrame(text=data)
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None

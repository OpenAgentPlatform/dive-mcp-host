from dataclasses import asdict, dataclass
from itertools import batched
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from dive_mcp_host.host.helpers.filter import filter_empty_dict
from dive_mcp_host.host.store.base import StoreManagerProtocol

logger = getLogger(__name__)


IMAGES_KEY = "images"
DOCUMENTS_KEY = "documents"


@dataclass(slots=True)
class ImageUrlMsg:
    """Structure for sending image with url."""

    url: str
    type: Literal["image"] = "image"

    @classmethod
    def create(cls, url: str) -> dict[str, str]:
        """Create the message."""
        return asdict(cls(url=url))


@dataclass(slots=True)
class ImageInfoMsg:
    """Structure for sending Image info."""

    text: str
    type: Literal["text"] = "text"

    @classmethod
    def create(
        cls, path: str, file_name: str, url: str | None = None
    ) -> dict[str, str]:
        """Create image msg with local file and url info."""
        if url:
            content = f"image {file_name}, local file path: {path}, url: {url}"
        else:
            content = f"image {file_name}, local file path: {path}"
        return asdict(cls(text=content))


@dataclass(slots=True)
class ImageBase64Msg:
    """Structure for sending image with inline base64."""

    base64: str
    mime_type: str
    type: Literal["image"] = "image"

    @classmethod
    def create(cls, inline_base64: str) -> dict[str, str | dict[str, str]]:
        """Create the message.

        Arguments:
            inline_base64: 'data:image/jpeg;base64,{base64_image}'
        """
        mime = inline_base64.removeprefix("data:").split(";")[0]
        base64_data = inline_base64.rsplit(",", maxsplit=1)[-1]
        return asdict(cls(base64=base64_data, mime_type=mime))


@dataclass(slots=True)
class DocumentPDFBase64Msg:
    """Structure for sending document with base64."""

    base64: str
    type: Literal["file"] = "file"
    mime_type: Literal["application/pdf"] = "application/pdf"

    @classmethod
    def create(cls, base64_data: str) -> dict[str, str]:
        """Create document base64 msg."""
        return asdict(cls(base64=base64_data))


@dataclass(slots=True)
class DocumentTextContentMsg:
    """Structure for sending text document."""

    text: str
    type: Literal["text"] = "text"

    @classmethod
    def create(cls, content: str, file_name: str) -> dict[str, str]:
        """Create document base64 msg."""
        return asdict(cls(text=f"content for document {file_name}, {content}"))


@dataclass(slots=True)
class DocumentInfoMsg:
    """Document info message."""

    text: str
    type: Literal["text"] = "text"

    @classmethod
    def create(
        cls, path: str, file_name: str, url: str | None = None
    ) -> dict[str, str]:
        """Create document msg with local file and url info."""
        if url:
            content = f"document {file_name}, local file path: {path}, url: {url}"
        else:
            content = f"document {file_name}, local file path: {path}"
        return asdict(cls(text=content))


OAP_MIN_COUNT = 2


class FileMsgConverter:
    """Structures image, document to their appropreate message format."""

    def __init__(self, model_provider: str, store: StoreManagerProtocol) -> None:
        """Initialize the FileMsg object."""
        self._model_provider = model_provider
        self._store = store

    async def _gen_document_msg(
        self,
        local_path: str,
        file_name: str,
        url: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate document message according to the provider."""
        # Normal text file
        if self._store.is_text(local_path):
            content = await self._store.get_document_text(local_path)
            return [
                DocumentTextContentMsg.create(content=content, file_name=file_name)
                if content
                else {},
                DocumentInfoMsg.create(path=local_path, url=url, file_name=file_name),
            ]

        base64_content, content_type = await self._store.get_document(local_path)

        # PDF
        # https://python.langchain.com/docs/how_to/multimodal_inputs/#documents-from-base64-data
        if content_type == "application/pdf" and self._model_provider in {
            "ChatOpenAI",
            "ChatAnthropic",
            "ChatGoogleGenerativeAI",
        }:
            return [
                DocumentPDFBase64Msg.create(base64_data=base64_content)
                if base64_content
                else {},
                DocumentInfoMsg.create(path=local_path, url=url, file_name=file_name),
            ]

        # Other file types or providers, simply pass the document info message
        return [DocumentInfoMsg.create(path=local_path, url=url, file_name=file_name)]

    async def _gen_image_msg_oap(
        self,
        local_path: str,
        file_name: str,
        url: str,
    ) -> list[dict[str, Any]]:
        """Generate image message according to the provider."""
        # Providers that supports passing image url.
        # https://python.langchain.com/docs/how_to/multimodal_inputs/#images-from-a-url
        if self._model_provider in {
            "ChatOpenAI",
            "ChatAnthropic",
            "ChatGoogleGenerativeAI",
        }:
            return [
                ImageUrlMsg.create(url=url),
                ImageInfoMsg.create(path=local_path, url=url, file_name=file_name),
            ]

        # Others
        inline_base64 = await self._store.get_image(local_path)
        return [
            ImageBase64Msg.create(inline_base64=inline_base64) if inline_base64 else {},
            ImageInfoMsg.create(path=local_path, url=url, file_name=file_name),
        ]

    async def _gen_image_msg(
        self, local_path: str, file_name: str
    ) -> list[dict[str, Any]]:
        inline_base64 = await self._store.get_image(local_path)
        return [
            ImageBase64Msg.create(inline_base64=inline_base64) if inline_base64 else {},
            ImageInfoMsg.create(path=local_path, file_name=file_name),
        ]

    def _is_using_oap(self, inpt: list[str]) -> bool:
        return (
            len(inpt) >= OAP_MIN_COUNT
            and len(inpt) % OAP_MIN_COUNT == 0
            and self._store.is_local_file(inpt[0])
            and self._store.is_url(inpt[1])
        )

    def _get_file_name(self, local_path: str) -> str:
        """Extract the original name from cache file path."""
        return Path(local_path).name.split("-", 1)[-1]

    async def _image_msgs(self, images: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if self._is_using_oap(images):
            # If the user is using OAP, the list would be [local_file, url ... etc]
            for local_path, url in batched(images, 2):
                file_name = self._get_file_name(local_path)
                msgs = await self._gen_image_msg_oap(
                    local_path=local_path, url=url, file_name=file_name
                )
                result.extend(msgs)
        else:
            # When not using OAP, the list sould all be local_file paths
            for local_path in images:
                file_name = self._get_file_name(local_path)
                msgs = await self._gen_image_msg(
                    local_path=local_path, file_name=file_name
                )
                result.extend(msgs)

        return result

    async def _document_msgs(self, documents: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if self._is_using_oap(documents):
            # If the user is using OAP, the list would be [local_file, url ... etc]
            for local_path, url in batched(documents, 2):
                file_name = self._get_file_name(local_path)
                msgs = await self._gen_document_msg(
                    local_path=local_path, url=url, file_name=file_name
                )
                result.extend(msgs)
        else:
            # When not using OAP, the list sould all be local_file paths
            for local_path in documents:
                file_name = self._get_file_name(local_path)
                msgs = await self._gen_document_msg(
                    local_path=local_path, file_name=file_name
                )
                result.extend(msgs)

        return result

    async def _structure_msgs(self, human_message: HumanMessage) -> HumanMessage:
        result = human_message

        if result.additional_kwargs:
            assert isinstance(result.content, list), (
                "HumanMessage content must be a list"
            )

            if images := result.additional_kwargs.get(IMAGES_KEY):
                image_msgs = await self._image_msgs(images)
                filtered = filter_empty_dict(image_msgs)
                logger.debug("got image_msgs: %s", len(filtered))
                result.content.extend(image_msgs)

            if documents := result.additional_kwargs.get(DOCUMENTS_KEY):
                document_msgs = await self._document_msgs(documents)
                filtered = filter_empty_dict(document_msgs)
                logger.debug("got document_msgs: %s", len(filtered))
                result.content.extend(document_msgs)

            result = HumanMessage(
                content=result.content,
                additional_kwargs=result.additional_kwargs,
            )

        logger.debug("file msg converter output: %s", result)

        return result

    async def process(
        self, inpt: ChatPromptValue | list[BaseMessage]
    ) -> list[BaseMessage]:
        """Structures image, document to their appropreate message format."""
        messages = inpt.to_messages() if isinstance(inpt, ChatPromptValue) else inpt

        ret = []
        for message in messages:
            if isinstance(message, HumanMessage):
                msg = await self._structure_msgs(message)
                ret.append(msg)
            else:
                ret.append(message)
        return ret

    @property
    def runnable(self) -> RunnableLambda:
        """Get the Runnable."""
        return RunnableLambda(self.process)

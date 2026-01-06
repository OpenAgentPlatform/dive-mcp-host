from datetime import datetime

from langchain_core.messages import ToolCall
from sqlalchemy import (
    CHAR,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB as PGJSONB
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Users(Base):
    """Users model.

    Attributes:
        id: User ID or fingerprint, depending on the prefix.
    """

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    user_type: Mapped[str | None] = mapped_column(CHAR(10))

    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        passive_deletes=True,
        uselist=True,
    )


# sqlite> PRAGMA table_info("chats");
# +-----+------------+------+---------+------------+----+
# | cid |    name    | type | notnull | dflt_value | pk |
# +-----+------------+------+---------+------------+----+
# | 0   | id         | TEXT | 1       |            | 1  |
# | 1   | title      | TEXT | 1       |            | 0  |
# | 2   | created_at | TEXT | 1       |            | 0  |
# +-----+------------+------+---------+------------+----+


class Chat(Base):
    """Chat model.

    Attributes:
        id: Chat ID.
        title: Chat title.
        created_at: Chat creation timestamp.
        updated_at: Chat update timestamp.
        starred_at: Chat star timestamp.
        user_id: User ID or fingerprint, depending on the prefix.
    """

    __tablename__ = "chats"
    __table_args__ = (Index("idx_chats_user_id", "user_id", postgresql_using="hash"),)
    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    title: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite"),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite")
    )
    starred_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite")
    )
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat",
        passive_deletes=True,
        uselist=True,
    )
    user: Mapped["Users"] = relationship(
        foreign_keys=user_id,
        back_populates="chats",
        passive_deletes=True,
    )


# sqlite> PRAGMA table_info("messages");
# +-----+------------+---------+---------+------------+----+
# | cid |    name    |  type   | notnull | dflt_value | pk |
# +-----+------------+---------+---------+------------+----+
# | 0   | id         | INTEGER | 1       |            | 1  |
# | 1   | content    | TEXT    | 1       |            | 0  |
# | 2   | role       | TEXT    | 1       |            | 0  |
# | 3   | chat_id    | TEXT    | 1       |            | 0  |
# | 4   | message_id | TEXT    | 1       |            | 0  |
# | 5   | created_at | TEXT    | 1       |            | 0  |
# | 6   | files      | TEXT    | 1       |            | 0  |
# +-----+------------+---------+---------+------------+----+


class Message(Base):
    """Message model.

    Attributes:
        id: Message ID.
        created_at: Message creation timestamp.
        content: Message content.
        role: Message role.
        chat_id: Chat ID.
        message_id: Message ID.
        files: Message files.
        tool_calls: Message tool calls.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("messages_message_id_index", "message_id", postgresql_using="hash"),
        Index("idx_messages_chat_id", "chat_id", postgresql_using="hash"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True).with_variant(Text(), "sqlite"),
    )
    content: Mapped[str] = mapped_column(Text())
    role: Mapped[str] = mapped_column(Text())
    chat_id: Mapped[str] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    message_id: Mapped[str] = mapped_column(Text(), unique=True)
    files: Mapped[str] = mapped_column(Text())
    tool_calls: Mapped[list[ToolCall] | None] = mapped_column(
        PGJSONB().with_variant(SQLiteJSON(), "sqlite"), default=[]
    )

    chat: Mapped["Chat"] = relationship(
        foreign_keys=chat_id,
        back_populates="messages",
        passive_deletes=True,
    )
    resource_usage: Mapped["ResourceUsage"] = relationship(
        back_populates="message",
        passive_deletes=True,
    )


class ResourceUsage(Base):
    """Resource usage model.

    Attributes:
        id: Resource usage ID.
        message_id: Message ID.
        model: Model name.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        user_token: User input tokens.
        custom_prompt_token: Custom prompt tokens.
        system_prompt_token: System prompt tokens.
        time_to_first_token: Time to first token in seconds.
        tokens_per_second: Tokens per second.
        total_run_time: Total run time.
    """

    __tablename__ = "resource_usage"
    __table_args__ = (Index("idx_resource_usage_message_id", "message_id"),)
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.message_id", ondelete="CASCADE"),
    )
    model: Mapped[str] = mapped_column(Text())
    total_input_tokens: Mapped[int] = mapped_column(BigInteger())
    total_output_tokens: Mapped[int] = mapped_column(BigInteger())
    user_token: Mapped[int] = mapped_column(BigInteger(), default=0)
    custom_prompt_token: Mapped[int] = mapped_column(BigInteger(), default=0)
    system_prompt_token: Mapped[int] = mapped_column(BigInteger(), default=0)
    time_to_first_token: Mapped[float] = mapped_column(Float(), default=0.0)
    tokens_per_second: Mapped[float] = mapped_column(Float(), default=0.0)
    total_run_time: Mapped[float] = mapped_column(Float())

    message: Mapped["Message"] = relationship(
        foreign_keys=message_id,
        back_populates="resource_usage",
        passive_deletes=True,
    )


class OAuth(Base):
    """Oauth Credentials model.

    Attributes:
        id: primary key.
        user_id: User ID.
        name: mcp server name.
        access_token: Access token.
        refresh_token: Refresh token.
        expire: Expire time.
        scope: Scope.
        client_id: Client ID.
        client_secret: Client secret.
        client_info: Client info.
        oauth_metadata: OAuth server metadata.
    """

    __tablename__ = "oauth_credentials"
    __table_args__ = (
        Index("idx_oauth_user_id", "user_id", postgresql_using="hash"),
        UniqueConstraint("user_id", "name", name="uq_oauth_user_id_name"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # name unique constraint with user_id
    name: Mapped[str] = mapped_column(Text())
    access_token: Mapped[str | None] = mapped_column(Text())
    refresh_token: Mapped[str | None] = mapped_column(Text())
    expire: Mapped[datetime | None] = mapped_column(DateTime())
    scope: Mapped[str | None] = mapped_column(Text())
    client_id: Mapped[str | None] = mapped_column(Text())
    client_secret: Mapped[str | None] = mapped_column(Text())
    client_info: Mapped[dict | None] = mapped_column(
        PGJSONB().with_variant(SQLiteJSON(), "sqlite")
    )
    oauth_metadata: Mapped[dict | None] = mapped_column(
        PGJSONB().with_variant(SQLiteJSON(), "sqlite")
    )

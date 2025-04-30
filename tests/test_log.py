import pytest

from dive_mcp_host.host.conf import LogConfig
from dive_mcp_host.host.tools.log import (
    LogBuffer,
    LogBufferNotFoundError,
    LogEvent,
    LogManager,
    LogMsg,
)


@pytest.mark.asyncio
async def test_log_buffer() -> None:
    """Test the log buffer."""
    log_buffer = LogBuffer(size=5)

    first_five = LogMsg(
        event=LogEvent.STATUS_CHANGE,
        body="first_five",
        mcp_server_name="test",
    )
    for _ in range(5):
        await log_buffer.push_log(first_five)

    assert log_buffer.get_logs() == [first_five] * 5

    # Log rotation
    last_two = LogMsg(
        event=LogEvent.STATUS_CHANGE,
        body="last_two",
        mcp_server_name="test",
    )
    for _ in range(2):
        await log_buffer.push_log(last_two)

    assert log_buffer.get_logs() == [first_five] * 3 + [last_two] * 2


@pytest.mark.asyncio
async def test_log_buffer_listener() -> None:
    """Test the log buffer listener."""
    # load 10 logs
    log_buffer = LogBuffer()
    for i in range(10):
        await log_buffer.push_log(
            LogMsg(
                event=LogEvent.STATUS_CHANGE,
                body=f"test {i}",
                mcp_server_name="test",
            )
        )

    result = []

    async def listener(msg: LogMsg) -> None:
        result.append(msg)

    async with log_buffer.add_listener(listener):
        # listener should have received all 10 logs
        assert len(result) == 10

        await log_buffer.push_log(
            LogMsg(
                event=LogEvent.STATUS_CHANGE,
                body="new log",
                mcp_server_name="test",
            )
        )

        # listener should have received the new log
        assert len(result) == 11

    assert len(log_buffer._listeners) == 0


@pytest.mark.asyncio
async def test_multiple_log_buffer_listeners() -> None:
    """Test multiple listeners on the log buffer."""
    log_buffer = LogBuffer()

    # Create some initial logs
    for i in range(5):
        await log_buffer.push_log(
            LogMsg(
                event=LogEvent.STATUS_CHANGE,
                body=f"initial {i}",
                mcp_server_name="test",
            )
        )

    # Prepare result collectors for each listener
    results_1 = []
    results_2 = []

    async def listener_1(msg: LogMsg) -> None:
        results_1.append(msg)

    async def listener_2(msg: LogMsg) -> None:
        results_2.append(msg)

    # Add both listeners
    async with log_buffer.add_listener(listener_1):
        # First listener should have received initial logs
        assert len(results_1) == 5

        # Add second listener
        async with log_buffer.add_listener(listener_2):
            # Second listener should also have received initial logs
            assert len(results_2) == 5

            # Add new logs that both listeners should receive
            for i in range(3):
                await log_buffer.push_log(
                    LogMsg(
                        event=LogEvent.STATUS_CHANGE,
                        body=f"new {i}",
                        mcp_server_name="test",
                    )
                )

            # Both listeners should have received the new logs
            assert len(results_1) == 8
            assert len(results_2) == 8

            # Verify the content of the logs
            for i in range(3):
                assert results_1[5 + i].body == f"new {i}"
                assert results_2[5 + i].body == f"new {i}"

        # After second listener context exits, only first listener should receive new logs  # noqa: E501, W505
        await log_buffer.push_log(
            LogMsg(
                event=LogEvent.STATUS_CHANGE,
                body="final log",
                mcp_server_name="test",
            )
        )

        assert len(results_1) == 9
        assert len(results_2) == 8  # Still 8, didn't receive the final log
        assert results_1[8].body == "final log"


@pytest.mark.asyncio
async def test_log_manager(log_config: LogConfig) -> None:
    """Test the LogManager functionality."""
    # Create a temporary directory for logs
    tmp_path = log_config.log_dir
    tmp_path.mkdir(exist_ok=True)
    try:
        # Create a log manager and a buffer
        log_manager = LogManager(log_dir=tmp_path)
        log_buffer = LogBuffer(name="test_buffer")

        # Register the buffer with the manager
        async with log_manager.register_buffer(log_buffer):
            # Push some logs to the buffer
            await log_buffer.push_log(
                LogMsg(
                    event=LogEvent.STATUS_CHANGE,
                    body="test log",
                    mcp_server_name="test_buffer",
                )
            )

            # Log file should be created
            log_file = tmp_path / "test_buffer.log"
            assert log_file.exists()

            # The buffer should be registered in the manager
            assert "test_buffer" in log_manager._buffers

        # After context exit, the buffer should be removed
        assert "test_buffer" not in log_manager._buffers
    finally:
        # Clean up test directory
        import shutil

        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_manager_listen_log(log_config: LogConfig) -> None:
    """Test listening to logs through the LogManager."""
    tmp_path = log_config.log_dir
    tmp_path.mkdir(exist_ok=True)
    try:
        log_manager = LogManager(log_dir=tmp_path)
        log_buffer = LogBuffer(name="test_listen")

        captured_logs = []

        async def test_listener(msg: LogMsg) -> None:
            captured_logs.append(msg)

        # Register the buffer with the manager
        async with log_manager.register_buffer(log_buffer):
            # Push a log before attaching listener
            test_log = LogMsg(
                event=LogEvent.STATUS_CHANGE,
                body="before listener",
                mcp_server_name="test_listen",
            )
            await log_buffer.push_log(test_log)

            # Listen to logs
            async with log_manager.listen_log("test_listen", test_listener):
                # Should receive the existing log
                assert len(captured_logs) == 1
                assert captured_logs[0].body == "before listener"

                # Push a new log
                new_log = LogMsg(
                    event=LogEvent.STATUS_CHANGE,
                    body="after listener",
                    mcp_server_name="test_listen",
                )
                await log_buffer.push_log(new_log)

                # Should receive the new log
                assert len(captured_logs) == 2
                assert captured_logs[1].body == "after listener"
    finally:
        import shutil

        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_manager_buffer_not_found(log_config: LogConfig) -> None:
    """Test error when trying to listen to a non-existent buffer."""
    tmp_path = log_config.log_dir
    tmp_path.mkdir(exist_ok=True)
    try:
        log_manager = LogManager(log_dir=tmp_path)

        async def test_listener(msg: LogMsg) -> None:
            pass

        # Try to listen to a buffer that doesn't exist
        with pytest.raises(LogBufferNotFoundError):
            async with log_manager.listen_log("nonexistent", test_listener):
                pass
    finally:
        import shutil

        shutil.rmtree(tmp_path, ignore_errors=True)

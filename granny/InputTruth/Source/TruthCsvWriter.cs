using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace Celeste.Mod.InputTruth;

internal sealed class TruthCsvWriter : IDisposable
{
    private const int FlushIntervalRows = 60;
    private const int InitialLineBufferSize = 512;
    private const int StreamBufferSize = 16 * 1024;
    private const string Header =
        "frame_idx,left,right,up,down,jump,dash,grab," +
        "input_active,room_id,pos_x,pos_y,speed_x,speed_y," +
        "dash_count,stamina,on_ground,death\n";

    private readonly StreamWriter _writer;
    private char[] _lineBuffer = new char[InitialLineBufferSize];
    private int _rowsSinceFlush;
    private bool _disposed;

    public TruthCsvWriter(string path)
    {
        FileStream stream = new(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            StreamBufferSize,
            FileOptions.SequentialScan);

        _writer = new StreamWriter(
            stream,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            StreamBufferSize,
            leaveOpen: false);
        _writer.Write(Header);
    }

    public void WriteRow(in InputFrameState frame)
    {
        EnsureLineCapacity(frame.RoomId.Length);
        int position = 0;
        Append(ref position, frame.FrameIndex);
        Append(ref position, ',');
        Append(ref position, frame.Left);
        Append(ref position, ',');
        Append(ref position, frame.Right);
        Append(ref position, ',');
        Append(ref position, frame.Up);
        Append(ref position, ',');
        Append(ref position, frame.Down);
        Append(ref position, ',');
        Append(ref position, frame.Jump);
        Append(ref position, ',');
        Append(ref position, frame.Dash);
        Append(ref position, ',');
        Append(ref position, frame.Grab);
        Append(ref position, ',');
        Append(ref position, frame.InputActive);
        Append(ref position, ',');
        AppendCsvString(ref position, frame.RoomId);
        Append(ref position, ',');
        Append(ref position, frame.PositionX);
        Append(ref position, ',');
        Append(ref position, frame.PositionY);
        Append(ref position, ',');
        Append(ref position, frame.SpeedX);
        Append(ref position, ',');
        Append(ref position, frame.SpeedY);
        Append(ref position, ',');
        Append(ref position, frame.DashCount);
        Append(ref position, ',');
        Append(ref position, frame.Stamina);
        Append(ref position, ',');
        Append(ref position, frame.OnGround);
        Append(ref position, ',');
        Append(ref position, frame.Death);
        Append(ref position, '\n');

        _writer.Write(_lineBuffer, 0, position);

        _rowsSinceFlush++;
        if (_rowsSinceFlush >= FlushIntervalRows)
            Flush();
    }

    public void Flush()
    {
        if (_disposed)
            return;

        _writer.Flush();
        _rowsSinceFlush = 0;
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _writer.Dispose();
    }

    private void Append(ref int position, char value)
    {
        _lineBuffer[position++] = value;
    }

    private void Append(ref int position, bool value)
    {
        _lineBuffer[position++] = value ? '1' : '0';
    }

    private void Append(ref int position, long value)
    {
        if (!value.TryFormat(
                _lineBuffer.AsSpan(position),
                out int written,
                provider: CultureInfo.InvariantCulture))
        {
            throw new InvalidOperationException("CSV line buffer is too small.");
        }

        position += written;
    }

    private void Append(ref int position, int value)
    {
        if (!value.TryFormat(
                _lineBuffer.AsSpan(position),
                out int written,
                provider: CultureInfo.InvariantCulture))
        {
            throw new InvalidOperationException("CSV line buffer is too small.");
        }

        position += written;
    }

    private void Append(ref int position, float value)
    {
        if (!value.TryFormat(
                _lineBuffer.AsSpan(position),
                out int written,
                "R",
                CultureInfo.InvariantCulture))
        {
            throw new InvalidOperationException("CSV line buffer is too small.");
        }

        position += written;
    }

    private void AppendCsvString(ref int position, string value)
    {
        bool quote = false;
        for (int i = 0; i < value.Length; i++)
        {
            char current = value[i];
            if (current == ',' || current == '"' || current == '\r' || current == '\n')
            {
                quote = true;
                break;
            }
        }

        if (!quote)
        {
            value.AsSpan().CopyTo(_lineBuffer.AsSpan(position));
            position += value.Length;
            return;
        }

        Append(ref position, '"');
        for (int i = 0; i < value.Length; i++)
        {
            char current = value[i];
            Append(ref position, current);
            if (current == '"')
                Append(ref position, '"');
        }
        Append(ref position, '"');
    }

    private void EnsureLineCapacity(int roomIdLength)
    {
        const int FixedFieldCapacity = 512;
        if (roomIdLength > (int.MaxValue - FixedFieldCapacity) / 2)
            throw new InvalidOperationException("Room id is too long for the CSV buffer.");

        int required = FixedFieldCapacity + roomIdLength * 2;
        if (_lineBuffer.Length >= required)
            return;

        int capacity = _lineBuffer.Length;
        while (capacity < required && capacity <= int.MaxValue / 2)
            capacity *= 2;

        if (capacity < required)
            capacity = required;

        Array.Resize(ref _lineBuffer, capacity);
    }
}

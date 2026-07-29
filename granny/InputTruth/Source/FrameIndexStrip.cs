using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Monocle;

namespace Celeste.Mod.InputTruth;

public static class FrameIndexStrip
{
    public const int PayloadBits = 24;
    public const int ChecksumBits = 4;
    public const int PatternLength = 2 + PayloadBits + ChecksumBits;
    public const int PayloadMask = (1 << PayloadBits) - 1;

    // Pattern() is called on the game thread. Reusing this non-interned buffer is
    // what keeps the required string API allocation-free in the steady state.
    // Callers must consume the returned pattern before their next Pattern() call.
    private static readonly string PatternBuffer = new('0', PatternLength);

    public static unsafe string Pattern(int idx)
    {
        uint value = (uint)idx & PayloadMask;
        uint checksum = 0;
        uint remaining = value;

        for (int nibble = 0; nibble < PayloadBits / ChecksumBits; nibble++)
        {
            checksum ^= remaining & 0xFu;
            remaining >>= ChecksumBits;
        }

        fixed (char* cells = PatternBuffer)
        {
            cells[0] = '1';
            cells[1] = '0';

            for (int bit = PayloadBits - 1, cell = 2; bit >= 0; bit--, cell++)
                cells[cell] = ((value >> bit) & 1u) != 0 ? '1' : '0';

            for (int bit = ChecksumBits - 1, cell = 2 + PayloadBits;
                 bit >= 0;
                 bit--, cell++)
            {
                cells[cell] = ((checksum >> bit) & 1u) != 0 ? '1' : '0';
            }
        }

        return PatternBuffer;
    }

    internal static void Render(string pattern)
    {
        // The first RenderCore can fire before Celeste's content load has
        // initialized Monocle.Draw; rendering then would crash the game.
        if (Draw.SpriteBatch is null || Draw.Pixel is null)
            return;

        Draw.SpriteBatch.Begin(
            SpriteSortMode.Deferred,
            BlendState.Opaque,
            SamplerState.PointClamp,
            DepthStencilState.None,
            RasterizerState.CullNone,
            effect: null,
            Engine.ScreenMatrix);

        Draw.Rect(
            InputTruthModule.FrameIndexMaskX,
            InputTruthModule.FrameIndexMaskY,
            InputTruthModule.FrameIndexMaskWidth,
            InputTruthModule.FrameIndexMaskHeight,
            Color.Black);

        int cellX = InputTruthModule.FrameIndexMaskX + InputTruthModule.FrameIndexQuietZone;
        int cellY = InputTruthModule.FrameIndexMaskY + InputTruthModule.FrameIndexQuietZone;

        for (int cell = 0; cell < PatternLength; cell++)
        {
            if (pattern[cell] == '1')
            {
                Draw.Rect(
                    cellX + cell * InputTruthModule.FrameIndexCellSize,
                    cellY,
                    InputTruthModule.FrameIndexCellSize,
                    InputTruthModule.FrameIndexCellSize,
                    Color.White);
            }
        }

        Draw.SpriteBatch.End();
    }
}

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Monocle;

namespace Celeste.Mod.InputTruth;

public static class InputOverlay
{
    public const int CellCount = InputOverlayGeometry.CellCount;
    public const int QuietZone = InputOverlayGeometry.QuietZone;
    public const int CellPitch = InputOverlayGeometry.CellPitch;
    public const int CellInsetX = InputOverlayGeometry.CellInsetX;
    public const int CellWidth = InputOverlayGeometry.CellWidth;
    public const int CellHeight = InputOverlayGeometry.CellHeight;

    private static readonly Color KeyUpColor = new(0x28, 0x28, 0x28, 0xff);

    public static int CellX(int cell)
    {
        return InputOverlayGeometry.CellX(cell);
    }

    public static int CellY(int cell)
    {
        return InputOverlayGeometry.CellY(cell);
    }

    public static Rectangle CellRect(int cell)
    {
        return new Rectangle(CellX(cell), CellY(cell), CellWidth, CellHeight);
    }

    internal static void Render(in InputFrameState frame)
    {
        // Match the strip's early-render guard: Celeste can call RenderCore
        // before Monocle.Draw has finished loading its shared resources.
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
            InputTruthModule.InputOverlayMaskX,
            InputTruthModule.InputOverlayMaskY,
            InputTruthModule.InputOverlayMaskWidth,
            InputTruthModule.InputOverlayMaskHeight,
            Color.Black);

        int cellX = InputTruthModule.InputOverlayMaskX + QuietZone + CellInsetX;
        int cellY = InputTruthModule.InputOverlayMaskY + QuietZone;

        for (int cell = 0; cell < CellCount; cell++)
        {
            Draw.Rect(
                cellX + cell * CellPitch,
                cellY,
                CellWidth,
                CellHeight,
                frame.IsKeyDown(cell) ? Color.White : KeyUpColor);
        }

        Draw.SpriteBatch.End();
    }
}

using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Monocle;

namespace Celeste.Mod.InputTruth;

/// <summary>
/// A deliberately WILD-STYLE input overlay: translucent, labelled, and drawn
/// over live game content, imitating the HUDs speedrunners composite into
/// their videos.
///
/// This is a TEST TARGET, not a truth source. Truth is the CSV row, and the
/// opaque <see cref="InputOverlay"/> remains the exact, machine-readable
/// instrument that E4 validated at macro-F1 1.0. This overlay exists because
/// harvested speedrun HUDs are alpha-blended over moving pixels, so a cell's
/// "released" appearance tracks whatever is behind it and no fixed threshold
/// decodes it. Recording this alongside engine truth is the only way to score
/// a translucent decoder against ground truth we own.
///
/// Placement is the right edge, chosen so it cannot collide with the
/// frame-index strip (top-left) or the opaque overlay (bottom-left): three
/// overlays can then coexist in one calibration session.
///
/// Off by default. Any session that enables it declares the rect in meta.json
/// so the builder masks it exactly like every other answer key — an
/// unmasked input display is the one bug that arrives disguised as good news.
/// </summary>
public static class WildOverlay
{
    // Logical 1920x1080 game-window coordinates, matching the other overlays.
    public const int PanelX = 1472;
    public const int PanelY = 904;
    public const int PanelWidth = 432;
    public const int PanelHeight = 160;

    public const int Columns = 4;
    public const int CellWidth = 96;
    public const int CellHeight = 64;
    public const int CellGap = 8;
    public const int PanelPad = 12;

    // Alpha levels picked to sit in the awkward middle the wild HUDs occupy:
    // high enough to read, low enough that the background bleeds through and
    // defeats a global threshold.
    private const float PanelAlpha = 0.35f;
    private const float CellUpAlpha = 0.28f;
    private const float CellDownAlpha = 0.82f;
    private const float LabelScale = 0.32f;

    private static readonly string[] Labels =
    {
        "Left", "Right", "Up", "Down", "Jump", "Dash", "Grab",
    };

    public static Rectangle PanelRect =>
        new(PanelX, PanelY, PanelWidth, PanelHeight);

    public static Rectangle CellRect(int cell)
    {
        int column = cell % Columns;
        int row = cell / Columns;
        return new Rectangle(
            PanelX + PanelPad + column * (CellWidth + CellGap),
            PanelY + PanelPad + row * (CellHeight + CellGap),
            CellWidth,
            CellHeight);
    }

    internal static void Render(in InputFrameState frame)
    {
        // Same early-render guard as the strip: RenderCore can run before
        // Monocle's shared resources finish loading.
        if (Draw.SpriteBatch is null || Draw.Pixel is null)
            return;

        Draw.SpriteBatch.Begin(
            SpriteSortMode.Deferred,
            // AlphaBlend, NOT Opaque: translucency is the whole point of this
            // overlay. XNA's AlphaBlend expects premultiplied colour, which is
            // what `Color * alpha` produces.
            BlendState.AlphaBlend,
            SamplerState.PointClamp,
            DepthStencilState.None,
            RasterizerState.CullNone,
            effect: null,
            Engine.ScreenMatrix);

        Draw.Rect(PanelX, PanelY, PanelWidth, PanelHeight,
            Color.Black * PanelAlpha);

        for (int cell = 0; cell < InputOverlay.CellCount; cell++)
        {
            Rectangle rect = CellRect(cell);
            bool down = frame.IsKeyDown(cell);
            Draw.Rect(
                rect.X, rect.Y, rect.Width, rect.Height,
                down ? Color.White * CellDownAlpha : Color.Gray * CellUpAlpha);

            if (ActiveFont.Font is not null)
            {
                ActiveFont.Draw(
                    Labels[cell],
                    new Vector2(rect.Center.X, rect.Center.Y),
                    new Vector2(0.5f, 0.5f),
                    Vector2.One * LabelScale,
                    down ? Color.Black : Color.White * 0.75f);
            }
        }

        Draw.SpriteBatch.End();
    }
}

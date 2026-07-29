using System;

namespace Celeste.Mod.InputTruth;

public static class InputOverlayGeometry
{
    public const int CellCount = 7;
    public const int QuietZone = 8;
    public const int CellPitch = 56;
    public const int CellInsetX = 8;
    public const int CellWidth = 40;
    public const int CellHeight = 32;

    public static int CellX(int cell)
    {
        ValidateCell(cell);
        return InputTruthModule.InputOverlayMaskX +
            QuietZone +
            CellInsetX +
            cell * CellPitch;
    }

    public static int CellY(int cell)
    {
        ValidateCell(cell);
        return InputTruthModule.InputOverlayMaskY + QuietZone;
    }

    private static void ValidateCell(int cell)
    {
        if ((uint)cell >= CellCount)
            throw new ArgumentOutOfRangeException(nameof(cell));
    }
}
